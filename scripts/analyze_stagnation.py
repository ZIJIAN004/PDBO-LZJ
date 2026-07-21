"""Diagnose why paper-aligned Max-Cut PDBO runs stop improving.

The script never perturbs or restarts a trajectory. It records inexpensive batch
statistics at a fixed interval and computes exact projected minimum curvature only
for a few representative trajectories after the global incumbent has plateaued.

Example:
    python scripts/analyze_stagnation.py \
        --gset_ids 67 70 72 77 81 --seeds 0 1 2 \
        --batch 100 --max_iters 5000 --out_prefix results/stagnation_g67_g81
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

# Allow running as "python scripts/analyze_stagnation.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbo import PDBOSolver, generate_max_cut, parse_gset  # noqa: E402
from pdbo.curvature import quadratic_hessian  # noqa: E402


TRACE_FIELDS = (
    "instance",
    "seed",
    "step",
    "global_cut",
    "global_last_improve_step",
    "global_stall_steps",
    "batch_cut_best",
    "batch_cut_mean",
    "batch_cut_std",
    "unique_rounded",
    "sampled_hamming_mean",
    "row_stalled_count",
    "centered_row_count",
    "movement_rms_mean",
    "movement_rms_median",
    "movement_rms_per_step_mean",
    "gradient_rms_mean",
    "gradient_rms_median",
    "fractionality_mean",
    "fractionality_min",
    "center_distance_mean",
    "free_fraction_mean",
    "dual_mean",
    "dual_min",
    "dual_max",
)

EVENT_FIELDS = (
    "instance",
    "seed",
    "event_index",
    "event_reason",
    "probe_role",
    "step",
    "global_cut",
    "global_stall_steps",
    "row_index",
    "row_cut",
    "row_last_improve_step",
    "row_stall_steps",
    "movement_rms",
    "movement_rms_per_step",
    "gradient_rms",
    "fractionality",
    "center_distance",
    "free_fraction",
    "free_variables",
    "lambda_min_projected",
    "eigen_residual",
    "eigensolver_status",
    "location",
    "classification",
)

RUN_FIELDS = (
    "instance",
    "seed",
    "status",
    "error",
    "n",
    "m",
    "batch",
    "optimizer",
    "lr_x",
    "lr_y",
    "dual_init",
    "max_iters",
    "state_every",
    "stall_window",
    "final_cut",
    "last_improve_iter",
    "final_integrality_mean",
    "final_integrality_min",
    "curvature_events",
    "event_classifications",
    "stop_reason",
    "time_s",
)


@dataclass(frozen=True)
class CurvatureResult:
    lambda_min: float
    residual: float
    free_variables: int
    status: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify PDBO incumbent plateaus without changing the solver trajectory."
    )
    parser.add_argument("--gset_ids", type=int, nargs="+", default=[67, 70, 72, 77, 81])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr_x", type=float, default=0.025)
    parser.add_argument("--lr_y", type=float, default=0.025)
    parser.add_argument("--dual_init", type=float, default=6.0)
    parser.add_argument("--optimizer", choices=["sgd", "rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--check_every", type=int, default=1)
    parser.add_argument("--state_every", type=int, default=50)
    parser.add_argument("--stall_window", type=int, default=300)
    parser.add_argument("--event_cooldown", type=int, default=750)
    parser.add_argument("--max_curvature_events", type=int, default=2)
    parser.add_argument("--curvature_probes", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--free_margin", type=float, default=0.05)
    parser.add_argument("--binary_tol", type=float, default=1e-3)
    parser.add_argument("--free_fraction_tol", type=float, default=0.01)
    parser.add_argument("--center_tol", type=float, default=0.1)
    parser.add_argument("--movement_tol", type=float, default=1e-5)
    parser.add_argument("--gradient_tol", type=float, default=1e-3)
    parser.add_argument("--curvature_tol", type=float, default=1e-3)
    parser.add_argument("--eig_tol", type=float, default=1e-5)
    parser.add_argument("--eig_maxiter", type=int, default=2000)
    parser.add_argument("--hamming_pairs", type=int, default=256)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--out_prefix", default="results/stagnation_analysis")
    return parser


def classify_stagnation(
    *,
    fractionality: float,
    free_fraction: float,
    movement_rms_per_step: float,
    gradient_rms: float,
    lambda_min: float,
    binary_tol: float,
    free_fraction_tol: float,
    movement_tol: float,
    gradient_tol: float,
    curvature_tol: float,
) -> str:
    """Return a conservative label for one plateaued trajectory snapshot."""
    if fractionality <= binary_tol or free_fraction <= free_fraction_tol:
        return "near_binary_discrete_basin"
    if movement_rms_per_step > movement_tol:
        return "continuous_state_still_moving"
    if gradient_rms > gradient_tol:
        return "optimizer_or_projection_limited"
    if not np.isfinite(lambda_min):
        return "unclassified_no_curvature"
    if lambda_min < -curvature_tol:
        return "negative_curvature_stall"
    if abs(lambda_min) <= curvature_tol:
        return "flat_fractional_stall"
    return "positive_curvature_fractional_basin"


def projected_min_curvature(
    objective_hessian: sparse.csr_matrix,
    primal: np.ndarray,
    dual: np.ndarray,
    *,
    free_margin: float,
    eig_tol: float,
    eig_maxiter: int,
    seed: int,
) -> CurvatureResult:
    """Compute the smallest Hessian eigenvalue on currently free coordinates."""
    x = np.asarray(primal, dtype=np.float64)
    y = np.asarray(dual, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("primal and dual must be same-length one-dimensional arrays")
    if objective_hessian.shape != (x.size, x.size):
        raise ValueError("objective_hessian shape does not match primal")
    if not (0.0 <= free_margin < 0.5):
        raise ValueError("free_margin must lie in [0, 0.5)")

    free = (x > free_margin) & (x < 1.0 - free_margin)
    free_indices = np.flatnonzero(free)
    dimension = int(free_indices.size)
    if dimension == 0:
        return CurvatureResult(float("nan"), float("nan"), 0, "no_free_variables")

    matrix = objective_hessian[free_indices][:, free_indices].astype(np.float64).tocsr()
    # For g(x) = x^2 - x, g_second(x) = 2.
    matrix = matrix + sparse.diags(2.0 * y[free_indices], format="csr")
    if dimension <= 64:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
        value = float(values[0])
        vector = vectors[:, 0]
        status = "dense"
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=dimension)
        v0 /= np.linalg.norm(v0)
        try:
            values, vectors = eigsh(
                matrix,
                k=1,
                which="SA",
                tol=eig_tol,
                maxiter=eig_maxiter,
                v0=v0,
            )
            value = float(values[0])
            vector = vectors[:, 0]
            status = "eigsh"
        except ArpackNoConvergence as exc:
            if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
                return CurvatureResult(float("nan"), float("nan"), dimension, "no_convergence")
            value = float(exc.eigenvalues[0])
            vector = exc.eigenvectors[:, 0]
            status = "eigsh_partial"

    matrix_action = matrix @ vector
    scale = max(1.0, abs(value), float(np.linalg.norm(matrix_action)))
    residual = float(np.linalg.norm(matrix_action - value * vector) / scale)
    return CurvatureResult(value, residual, dimension, status)


class StagnationAnalyzer:
    """State callback that writes lightweight traces and sparse curvature events."""

    def __init__(
        self,
        *,
        instance: str,
        seed: int,
        data: dict,
        objective_hessian: sparse.csr_matrix,
        args: argparse.Namespace,
        trace_writer: csv.DictWriter,
        event_writer: csv.DictWriter,
        trace_file,
        event_file,
    ):
        self.instance = instance
        self.seed = seed
        self.args = args
        self.objective_hessian = objective_hessian
        self.trace_writer = trace_writer
        self.event_writer = event_writer
        self.trace_file = trace_file
        self.event_file = event_file
        self.previous_primal: Optional[jax.Array] = None
        self.previous_step = -1
        self.row_best_objective: Optional[np.ndarray] = None
        self.row_last_improve: Optional[np.ndarray] = None
        self.event_count = 0
        self.next_event_step = 0
        self.classifications: Counter[str] = Counter()

        rows = jnp.asarray(data["Q_indices"][0])
        cols = jnp.asarray(data["Q_indices"][1])
        values = jnp.asarray(data["Q_values"])
        linear = jnp.asarray(data["c"])
        free_margin = float(args.free_margin)

        @jax.jit
        def rounded_objectives(primal):
            rounded = jnp.rint(primal)
            quadratic = jnp.sum(
                values[jnp.newaxis, :]
                * rounded[:, rows]
                * rounded[:, cols],
                axis=1,
            )
            return quadratic + rounded @ linear

        @jax.jit
        def state_metrics(primal, dual, gradient, previous_primal):
            movement = jnp.sqrt(jnp.mean((primal - previous_primal) ** 2, axis=1))
            gradient_rms = jnp.sqrt(jnp.mean(gradient ** 2, axis=1))
            fractionality = jnp.mean(primal * (1.0 - primal), axis=1)
            center_distance = jnp.mean(jnp.abs(primal - 0.5), axis=1)
            free_fraction = jnp.mean(
                (primal > free_margin) & (primal < 1.0 - free_margin),
                axis=1,
            )
            return (
                rounded_objectives(primal),
                movement,
                gradient_rms,
                fractionality,
                center_distance,
                free_fraction,
                jnp.asarray([dual.mean(), dual.min(), dual.max()]),
            )

        self._rounded_objectives = rounded_objectives
        self._state_metrics = state_metrics

        rng = np.random.default_rng(seed + 19_003)
        pair_count = min(args.hamming_pairs, args.batch * max(args.batch - 1, 1) // 2)
        first = rng.integers(0, args.batch, size=pair_count)
        second = rng.integers(0, args.batch, size=pair_count)
        same = first == second
        while np.any(same) and args.batch > 1:
            second[same] = rng.integers(0, args.batch, size=int(same.sum()))
            same = first == second
        self._hamming_first = first
        self._hamming_second = second

    def set_initial_state(self, primal: jax.Array) -> None:
        self.previous_primal = jnp.array(primal)
        initial_objectives = np.asarray(jax.device_get(self._rounded_objectives(primal)))
        self.row_best_objective = initial_objectives.astype(np.float64)
        self.row_last_improve = np.full(initial_objectives.shape, -1, dtype=np.int64)

    def __call__(
        self,
        step: int,
        primal: jax.Array,
        dual: jax.Array,
        gradient: jax.Array,
        global_objective: jax.Array,
        incumbent: jax.Array,
        global_last_improve_step: int,
    ) -> None:
        del incumbent
        if self.previous_primal is None:
            self.set_initial_state(primal)
        delta_steps = max(step - self.previous_step, 1)
        metrics = jax.device_get(
            self._state_metrics(primal, dual, gradient, self.previous_primal)
        )
        (
            row_objectives,
            movement,
            gradient_rms,
            fractionality,
            center_distance,
            free_fraction,
            dual_stats,
        ) = (np.asarray(value) for value in metrics)
        movement_per_step = movement / float(delta_steps)

        improved = row_objectives < self.row_best_objective
        self.row_best_objective = np.minimum(self.row_best_objective, row_objectives)
        self.row_last_improve[improved] = step
        row_stall = step - self.row_last_improve

        rounded = np.asarray(jax.device_get(jnp.rint(primal)), dtype=np.uint8)
        packed = np.packbits(rounded, axis=1)
        unique_rounded = int(np.unique(packed, axis=0).shape[0])
        if self._hamming_first.size:
            sampled_hamming = float(
                np.mean(
                    rounded[self._hamming_first]
                    != rounded[self._hamming_second]
                )
            )
        else:
            sampled_hamming = 0.0

        global_stall = int(step - global_last_improve_step)
        trace_row = {
            "instance": self.instance,
            "seed": self.seed,
            "step": step,
            "global_cut": -float(global_objective),
            "global_last_improve_step": global_last_improve_step,
            "global_stall_steps": global_stall,
            "batch_cut_best": -float(np.min(row_objectives)),
            "batch_cut_mean": -float(np.mean(row_objectives)),
            "batch_cut_std": float(np.std(row_objectives)),
            "unique_rounded": unique_rounded,
            "sampled_hamming_mean": sampled_hamming,
            "row_stalled_count": int(np.sum(row_stall >= self.args.stall_window)),
            "centered_row_count": int(np.sum(center_distance <= self.args.center_tol)),
            "movement_rms_mean": float(np.mean(movement)),
            "movement_rms_median": float(np.median(movement)),
            "movement_rms_per_step_mean": float(np.mean(movement_per_step)),
            "gradient_rms_mean": float(np.mean(gradient_rms)),
            "gradient_rms_median": float(np.median(gradient_rms)),
            "fractionality_mean": float(np.mean(fractionality)),
            "fractionality_min": float(np.min(fractionality)),
            "center_distance_mean": float(np.mean(center_distance)),
            "free_fraction_mean": float(np.mean(free_fraction)),
            "dual_mean": float(dual_stats[0]),
            "dual_min": float(dual_stats[1]),
            "dual_max": float(dual_stats[2]),
        }
        self.trace_writer.writerow(trace_row)
        self.trace_file.flush()

        event_due = (
            global_stall >= self.args.stall_window
            and step >= self.next_event_step
            and self.event_count < self.args.max_curvature_events
        )
        if event_due:
            self._record_curvature_event(
                step=step,
                global_cut=-float(global_objective),
                global_stall=global_stall,
                primal=primal,
                dual=dual,
                row_objectives=row_objectives,
                row_stall=row_stall,
                movement=movement,
                movement_per_step=movement_per_step,
                gradient_rms=gradient_rms,
                fractionality=fractionality,
                center_distance=center_distance,
                free_fraction=free_fraction,
            )

        self.previous_primal = jnp.array(primal)
        self.previous_step = step

    def _probe_indices(
        self,
        *,
        row_objectives: np.ndarray,
        row_stall: np.ndarray,
        movement_per_step: np.ndarray,
        gradient_rms: np.ndarray,
        fractionality: np.ndarray,
        center_distance: np.ndarray,
    ) -> list[tuple[str, int]]:
        stalled = np.flatnonzero(row_stall >= self.args.stall_window)
        candidates = stalled if stalled.size else np.arange(row_objectives.size)
        selected: list[tuple[str, int]] = []

        best_index = int(candidates[np.argmin(row_objectives[candidates])])
        selected.append(("best_current", best_index))

        fractional = candidates[fractionality[candidates] > self.args.binary_tol]
        if fractional.size:
            order = np.lexsort((gradient_rms[fractional], movement_per_step[fractional]))
            selected.append(("most_stationary_fractional", int(fractional[order[0]])))

        if self.args.curvature_probes >= 3:
            centered_index = int(candidates[np.argmin(center_distance[candidates])])
            selected.append(("most_centered", centered_index))

        deduplicated = []
        seen = set()
        for role, index in selected:
            if index not in seen:
                deduplicated.append((role, index))
                seen.add(index)
            if len(deduplicated) >= self.args.curvature_probes:
                break
        return deduplicated

    def _record_curvature_event(
        self,
        *,
        step: int,
        global_cut: float,
        global_stall: int,
        primal: jax.Array,
        dual: jax.Array,
        row_objectives: np.ndarray,
        row_stall: np.ndarray,
        movement: np.ndarray,
        movement_per_step: np.ndarray,
        gradient_rms: np.ndarray,
        fractionality: np.ndarray,
        center_distance: np.ndarray,
        free_fraction: np.ndarray,
    ) -> None:
        probes = self._probe_indices(
            row_objectives=row_objectives,
            row_stall=row_stall,
            movement_per_step=movement_per_step,
            gradient_rms=gradient_rms,
            fractionality=fractionality,
            center_distance=center_distance,
        )
        self.event_count += 1
        self.next_event_step = step + self.args.event_cooldown

        for probe_offset, (role, row_index) in enumerate(probes):
            row_primal = np.asarray(jax.device_get(primal[row_index]), dtype=np.float64)
            row_dual = np.asarray(jax.device_get(dual[row_index]), dtype=np.float64)
            curvature = projected_min_curvature(
                self.objective_hessian,
                row_primal,
                row_dual,
                free_margin=self.args.free_margin,
                eig_tol=self.args.eig_tol,
                eig_maxiter=self.args.eig_maxiter,
                seed=self.seed * 100_003 + self.event_count * 101 + probe_offset,
            )
            classification = classify_stagnation(
                fractionality=float(fractionality[row_index]),
                free_fraction=float(free_fraction[row_index]),
                movement_rms_per_step=float(movement_per_step[row_index]),
                gradient_rms=float(gradient_rms[row_index]),
                lambda_min=curvature.lambda_min,
                binary_tol=self.args.binary_tol,
                free_fraction_tol=self.args.free_fraction_tol,
                movement_tol=self.args.movement_tol,
                gradient_tol=self.args.gradient_tol,
                curvature_tol=self.args.curvature_tol,
            )
            location = (
                "near_center"
                if center_distance[row_index] <= self.args.center_tol
                else "away_from_center"
            )
            event_row = {
                "instance": self.instance,
                "seed": self.seed,
                "event_index": self.event_count,
                "event_reason": "global_incumbent_plateau",
                "probe_role": role,
                "step": step,
                "global_cut": global_cut,
                "global_stall_steps": global_stall,
                "row_index": row_index,
                "row_cut": -float(row_objectives[row_index]),
                "row_last_improve_step": int(self.row_last_improve[row_index]),
                "row_stall_steps": int(row_stall[row_index]),
                "movement_rms": float(movement[row_index]),
                "movement_rms_per_step": float(movement_per_step[row_index]),
                "gradient_rms": float(gradient_rms[row_index]),
                "fractionality": float(fractionality[row_index]),
                "center_distance": float(center_distance[row_index]),
                "free_fraction": float(free_fraction[row_index]),
                "free_variables": curvature.free_variables,
                "lambda_min_projected": curvature.lambda_min,
                "eigen_residual": curvature.residual,
                "eigensolver_status": curvature.status,
                "location": location,
                "classification": classification,
            }
            self.event_writer.writerow(event_row)
            self.event_file.flush()
            self.classifications[classification] += 1
            print(
                f"{self.instance:<4} seed={self.seed:<3} step={step:<5} role={role:<26} "
                f"cut={-row_objectives[row_index]:>8.1f} frac={fractionality[row_index]:.4g} "
                f"move={movement_per_step[row_index]:.3g} grad={gradient_rms[row_index]:.3g} "
                f"lambda={curvature.lambda_min:.5g} class={classification}"
            )


def _iter_instances(ids: Iterable[int], allow_missing: bool):
    missing = []
    for gid in ids:
        path = f"./instance/Gset/G{gid}.txt"
        if not os.path.exists(path):
            missing.append(path)
            continue
        graph = parse_gset(str(gid))
        yield gid, generate_max_cut(graph)
    if missing and not allow_missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"missing requested Gset files: {joined}. "
            "Run scripts/download_gset.py or pass --allow_missing."
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch < 1 or args.max_iters < 1:
        raise ValueError("batch and max_iters must be positive")
    if args.check_every < 1 or args.state_every < 1 or args.stall_window < 1:
        raise ValueError("check_every, state_every, and stall_window must be positive")
    if args.event_cooldown < 1 or args.max_curvature_events < 0:
        raise ValueError("event_cooldown must be positive and max_curvature_events non-negative")
    if not (0.0 <= args.free_margin < 0.5):
        raise ValueError("free_margin must lie in [0, 0.5)")
    if args.hamming_pairs < 0:
        raise ValueError("hamming_pairs must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    try:
        _validate_args(args)
        instances = list(_iter_instances(args.gset_ids, args.allow_missing))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not instances:
        print("No usable instances found.", file=sys.stderr)
        return 1

    prefix = os.path.abspath(args.out_prefix)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    trace_path = prefix + "_trace.csv"
    event_path = prefix + "_events.csv"
    run_path = prefix + "_runs.csv"
    print("Configuration:", json.dumps(vars(args), sort_keys=True))

    failed = 0
    with (
        open(trace_path, "w", newline="") as trace_file,
        open(event_path, "w", newline="") as event_file,
        open(run_path, "w", newline="") as run_file,
    ):
        trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
        event_writer = csv.DictWriter(event_file, fieldnames=EVENT_FIELDS)
        run_writer = csv.DictWriter(run_file, fieldnames=RUN_FIELDS)
        trace_writer.writeheader()
        event_writer.writeheader()
        run_writer.writeheader()

        for gid, data in instances:
            instance = f"G{gid}"
            objective_hessian = quadratic_hessian(
                data["Q_indices"], data["Q_values"], data["num_vars"]
            )
            for seed in args.seeds:
                run_row = {field: "" for field in RUN_FIELDS}
                run_row.update(
                    {
                        "instance": instance,
                        "seed": seed,
                        "n": data["num_vars"],
                        "m": data["num_edges"],
                        "batch": args.batch,
                        "optimizer": args.optimizer,
                        "lr_x": args.lr_x,
                        "lr_y": args.lr_y,
                        "dual_init": args.dual_init,
                        "max_iters": args.max_iters,
                        "state_every": args.state_every,
                        "stall_window": args.stall_window,
                    }
                )
                started = time.perf_counter()
                analyzer = StagnationAnalyzer(
                    instance=instance,
                    seed=seed,
                    data=data,
                    objective_hessian=objective_hessian,
                    args=args,
                    trace_writer=trace_writer,
                    event_writer=event_writer,
                    trace_file=trace_file,
                    event_file=event_file,
                )
                try:
                    solver = PDBOSolver(
                        n_vars=data["num_vars"],
                        objective_type="quadratic",
                        Q_indices=data["Q_indices"],
                        Q_values=data["Q_values"],
                        c=data["c"],
                        optimizer_type=args.optimizer,
                        batch_size=args.batch,
                        primal_lr=args.lr_x,
                        dual_lr=args.lr_y,
                        dual_init=args.dual_init,
                        dual_init_mode="constant",
                        g_type="quad",
                        g_normalize=False,
                        max_iters=args.max_iters,
                        primal_init="uniform",
                        rounding_samples=0,
                        check_every=args.check_every,
                        state_callback=analyzer,
                        state_callback_every=args.state_every,
                        perturbation=False,
                        seed=seed,
                        verbose=False,
                    )
                    analyzer.set_initial_state(solver.primal)
                    result = solver.optimize()
                    run_row.update(
                        {
                            "status": "ok",
                            "error": "",
                            "final_cut": -float(result.objective),
                            "last_improve_iter": solver.last_improvement_step,
                            "final_integrality_mean": solver.final_integrality,
                            "final_integrality_min": solver.final_integrality_min,
                            "curvature_events": analyzer.event_count,
                            "event_classifications": json.dumps(
                                dict(analyzer.classifications), sort_keys=True
                            ),
                            "stop_reason": result.stop_reason,
                            "time_s": round(time.perf_counter() - started, 6),
                        }
                    )
                    print(
                        f"{instance:<4} seed={seed:<3} final={-result.objective:>8.1f} "
                        f"last_iter={solver.last_improvement_step:<5} "
                        f"integrality={solver.final_integrality:.4g} events={analyzer.event_count}"
                    )
                except Exception as exc:
                    failed += 1
                    run_row.update(
                        {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}".replace("\n", " "),
                            "curvature_events": analyzer.event_count,
                            "event_classifications": json.dumps(
                                dict(analyzer.classifications), sort_keys=True
                            ),
                            "time_s": round(time.perf_counter() - started, 6),
                        }
                    )
                    print(
                        f"{instance} seed={seed} [{run_row['error']}]",
                        file=sys.stderr,
                    )
                run_writer.writerow(run_row)
                run_file.flush()

    print(f"Trace:  {trace_path}")
    print(f"Events: {event_path}")
    print(f"Runs:   {run_path}")
    print(f"failures={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()))
    sys.exit(main())
