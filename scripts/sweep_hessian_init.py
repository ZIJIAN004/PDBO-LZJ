"""Sweep g(x) and controlled initial Hessian levels for Max-Cut.

For a quadratic objective with Hessian A and initial constraint curvature
d = g''(x), curvature initialization chooses coordinate-wise duals so that

    A + diag(y * d) = A + (1 + r) * (-lambda_min(A)) * I.

The relative level r therefore has a direct interpretation: r > 0 is on the
convex side, r = 0 is the PSD-singular boundary, -1 < r < 0 is nonconvex, and
r = -1 is the exact y = 0 baseline.  The latter deliberately uses constant
initialization so it remains available for g whose second derivative is zero
or negative and hence cannot support curvature matching.

Example (run from the repository root):
    python scripts/sweep_hessian_init.py --gset_ids 1 \
        --g quad entropy sin --levels 0.1 0 -0.1 -0.5 -1 \
        --seeds 0 1 2 --batch 100 --max_iters 5000 \
        --out sweep_hessian_init.csv
"""

import argparse
import csv
import os
import sys
import time

import jax
import numpy as np

# Allow running as "python scripts/sweep_hessian_init.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbo import PDBOSolver, generate_max_cut, parse_gset  # noqa: E402
from pdbo.curvature import quadratic_hessian, smallest_eigenvalue  # noqa: E402
from solver_jax import G_TYPES  # noqa: E402


DEFAULT_LEVELS = (0.1, 0.0, -0.1, -0.5, -1.0)

FIELDNAMES = (
    "instance",
    "g",
    "level",
    "level_label",
    "seed",
    "objective_hessian_lambda_min",
    "objective_curvature",
    "target_initial_lambda_min",
    "curvature_shift",
    "curvature_min",
    "curvature_max",
    "dual_min",
    "dual_max",
    "dual_mean",
    "eigensolver",
    "eigen_residual",
    "boundary_residual",
    "cut",
    "objective",
    "converged",
    "binary_conv",
    "obj_plateaued",
    "final_integrality",
    "last_improve_iter",
    "iters",
    "batch",
    "lr_x",
    "lr_y",
    "primal_init",
    "g_normalize",
    "init_time_s",
    "time_s",
    "total_time_s",
    "stop_reason",
    "status",
    "error",
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep g(x) and relative initial Hessian levels for Max-Cut."
    )
    parser.add_argument(
        "--gset_ids",
        type=int,
        nargs="+",
        default=[1],
        help="Gset instance ids; the .txt files must exist in ./instance/Gset/",
    )
    parser.add_argument(
        "--g",
        dest="g_types",
        nargs="+",
        default=list(G_TYPES),
        help="constraint functions to sweep; unknown or curvature-ineligible choices are recorded",
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="relative Hessian levels r (default: convex, boundary, light/medium nonconvex, y=0)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--g_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rescale every g to value range [-1, 0] (default on)",
    )
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--lr_x", type=float, default=0.025)
    parser.add_argument("--lr_y", type=float, default=0.025)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument(
        "--primal_init", choices=["uniform", "half", "binary"], default="uniform"
    )
    parser.add_argument(
        "--curvature_tol",
        type=float,
        default=1e-12,
        help="minimum admissible g'' value for matched curvature initialization",
    )
    parser.add_argument("--eig_tol", type=float, default=1e-8)
    parser.add_argument(
        "--integrality_tol",
        type=float,
        default=1e-3,
        help="minimum final mean x*(1-x) across the batch; one trajectory must be binary",
    )
    parser.add_argument(
        "--plateau_window",
        type=int,
        default=None,
        help="minimum iterations since the last incumbent improvement (default: 10%% of budget)",
    )
    parser.add_argument("--out", default="sweep_hessian_init.csv")
    return parser


def _level_label(level):
    if np.isclose(level, -1.0, rtol=0.0, atol=1e-12):
        return "zero_dual"
    if np.isclose(level, 0.0, rtol=0.0, atol=1e-12):
        return "boundary"
    if level > 0.0:
        return "convex"
    if level >= -0.25:
        return "light_nonconvex"
    return "medium_nonconvex"


def _empty_row(instance, g_type, level, seed, args):
    row = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "instance": instance,
            "g": g_type,
            "level": level,
            "level_label": _level_label(level),
            "seed": seed,
            "iters": args.max_iters,
            "batch": args.batch,
            "lr_x": args.lr_x,
            "lr_y": args.lr_y,
            "primal_init": args.primal_init,
            "g_normalize": int(args.g_normalize),
        }
    )
    return row


def _g_curvature_bounds(solver):
    """Evaluate g'' at every initial primal coordinate using JAX autodiff."""

    g_second = jax.grad(jax.grad(solver._make_g()))
    curvature = np.asarray(jax.vmap(jax.vmap(g_second))(solver.primal), dtype=np.float64)
    if not np.isfinite(curvature).all():
        raise ValueError("g'' produced non-finite initial curvature")
    return float(curvature.min()), float(curvature.max())


def _error_text(exc):
    return f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ")


def _make_solver(data, g_type, level, seed, objective_lambda_min, args):
    # r=-1 is exactly y=0 and needs no division by g''.  Keeping it out of
    # curvature mode makes the baseline valid for vshape, huber, and partial g.
    zero_dual = np.isclose(level, -1.0, rtol=0.0, atol=1e-12)
    return PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        optimizer_type="rmsprop",
        batch_size=args.batch,
        primal_lr=args.lr_x,
        dual_lr=args.lr_y,
        dual_init=0.0,
        dual_init_mode="constant" if zero_dual else "curvature",
        hessian_init_level=level,
        curvature_tol=args.curvature_tol,
        eig_tol=args.eig_tol,
        # This value was computed immediately above from this instance's Hessian.
        trusted_objective_hessian_lambda_min=objective_lambda_min,
        g_type=g_type,
        g_normalize=args.g_normalize,
        max_iters=args.max_iters,
        primal_init=args.primal_init,
        seed=seed,
        verbose=False,
    )


def _record_initialization(row, solver, objective_lambda_min, level):
    dual = np.asarray(solver.dual, dtype=np.float64)
    row.update(
        {
            "objective_hessian_lambda_min": objective_lambda_min,
            "objective_curvature": -objective_lambda_min,
            "dual_min": float(dual.min()),
            "dual_max": float(dual.max()),
            "dual_mean": float(dual.mean()),
        }
    )

    diagnostics = solver.initial_curvature_diagnostics
    if diagnostics is None:
        curvature_min, curvature_max = _g_curvature_bounds(solver)
        row.update(
            {
                "target_initial_lambda_min": objective_lambda_min,
                "curvature_shift": 0.0,
                "curvature_min": curvature_min,
                "curvature_max": curvature_max,
                "eigensolver": "constant_zero_dual",
                "eigen_residual": "",
                "boundary_residual": "",
            }
        )
        return

    row.update(
        {
            "objective_hessian_lambda_min": diagnostics.lambda_min,
            "objective_curvature": diagnostics.objective_curvature,
            "target_initial_lambda_min": diagnostics.target_min_eigenvalue,
            "curvature_shift": diagnostics.curvature_shift,
            "curvature_min": diagnostics.curvature_min,
            "curvature_max": diagnostics.curvature_max,
            "eigensolver": diagnostics.eigensolver,
            "eigen_residual": (
                diagnostics.eigen_residual if np.isfinite(diagnostics.eigen_residual) else ""
            ),
            "boundary_residual": (
                diagnostics.boundary_residual if np.isfinite(diagnostics.boundary_residual) else ""
            ),
        }
    )

    expected_target = level * diagnostics.objective_curvature
    if not np.isclose(
        diagnostics.target_min_eigenvalue,
        expected_target,
        rtol=1e-7,
        atol=max(args_epsilon(diagnostics.objective_curvature), 1e-10),
    ):
        raise RuntimeError("curvature initializer did not report the requested Hessian level")


def args_epsilon(scale):
    """Scale-aware absolute tolerance for a reported spectral target."""

    return 1e-9 * max(1.0, abs(float(scale)))


def _run_one(instance, data, g_type, level, seed, objective_lambda_min, args):
    row = _empty_row(instance, g_type, level, seed, args)
    objective_curvature = -objective_lambda_min
    row.update(
        {
            "objective_hessian_lambda_min": objective_lambda_min,
            "objective_curvature": objective_curvature,
            "target_initial_lambda_min": level * objective_curvature,
        }
    )
    init_start = time.perf_counter()
    try:
        if level < -1.0 and not np.isclose(level, -1.0, rtol=0.0, atol=1e-12):
            raise ValueError("relative Hessian level must be at least -1")
        solver = _make_solver(data, g_type, level, seed, objective_lambda_min, args)
        _record_initialization(row, solver, objective_lambda_min, level)
        row["init_time_s"] = round(time.perf_counter() - init_start, 6)
    except Exception as exc:
        row["init_time_s"] = round(time.perf_counter() - init_start, 6)
        row["status"] = "unsupported_g" if g_type not in G_TYPES else "unsupported_curvature"
        row["error"] = _error_text(exc)
        return row

    optimize_start = time.perf_counter()
    try:
        result = solver.optimize()
        elapsed = time.perf_counter() - optimize_start
        total_elapsed = time.perf_counter() - init_start

        integrality = solver.final_integrality_min
        last_improvement = solver.last_improvement_step
        binary_conv = integrality < args.integrality_tol
        plateau_window = (
            args.plateau_window
            if args.plateau_window is not None
            else max(50, args.max_iters // 10)
        )
        obj_plateaued = (args.max_iters - 1 - last_improvement) >= plateau_window
        converged = binary_conv and obj_plateaued

        row.update(
            {
                "cut": -float(result.objective),
                "objective": float(result.objective),
                "converged": int(converged),
                "binary_conv": int(binary_conv),
                "obj_plateaued": int(obj_plateaued),
                "final_integrality": round(integrality, 8),
                "last_improve_iter": last_improvement,
                "time_s": round(elapsed, 6),
                "total_time_s": round(total_elapsed, 6),
                "stop_reason": result.stop_reason,
                "status": "ok" if converged else "not_converged",
                "error": "",
            }
        )
    except Exception as exc:
        row["time_s"] = round(time.perf_counter() - optimize_start, 6)
        row["total_time_s"] = round(time.perf_counter() - init_start, 6)
        row["status"] = "error"
        row["error"] = _error_text(exc)
    return row


def main():
    args = build_parser().parse_args()
    if not args.levels:
        print("No Hessian levels requested.", file=sys.stderr)
        return 1

    instances = {}
    for gid in args.gset_ids:
        path = f"./instance/Gset/G{gid}.txt"
        if not os.path.exists(path):
            print(f"[skip] missing instance file: {path}", file=sys.stderr)
            continue
        graph = parse_gset(str(gid))
        data = generate_max_cut(graph)
        hessian = quadratic_hessian(
            data["Q_indices"], data["Q_values"], data["num_vars"]
        )
        objective_lambda_min = smallest_eigenvalue(hessian, tol=args.eig_tol)
        instances[gid] = (data, objective_lambda_min)

    if not instances:
        print("No usable instances found. Put the G*.txt files in ./instance/Gset/.", file=sys.stderr)
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        for gid, (data, objective_lambda_min) in instances.items():
            instance = f"G{gid}"
            for g_type in args.g_types:
                for level in args.levels:
                    for seed in args.seeds:
                        row = _run_one(
                            instance,
                            data,
                            g_type,
                            level,
                            seed,
                            objective_lambda_min,
                            args,
                        )
                        writer.writerow(row)
                        output.flush()
                        rows.append(row)

                        if row["status"] in {"ok", "not_converged"}:
                            print(
                                f"{instance:<6} g={g_type:<12} r={level:>6g} "
                                f"seed={seed:<3} cut={row['cut']:>9.1f} "
                                f"lambda0={row['target_initial_lambda_min']:>10.4g} "
                                f"[{row['status']}]"
                            )
                        else:
                            print(
                                f"{instance:<6} g={g_type:<12} r={level:>6g} "
                                f"seed={seed:<3} [{row['status']}] {row['error']}",
                                file=sys.stderr,
                            )

    completed = sum(row["status"] in {"ok", "not_converged"} for row in rows)
    converged = sum(row["status"] == "ok" for row in rows)
    unsupported = sum(str(row["status"]).startswith("unsupported") for row in rows)
    failed = len(rows) - completed - unsupported
    print(
        f"\nWrote {len(rows)} rows to {args.out}: {converged} converged, "
        f"{completed - converged} not converged, {unsupported} unsupported, {failed} errors."
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
