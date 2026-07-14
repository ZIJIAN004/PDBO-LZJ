"""Controlled sweep over the constraint function g(x) and the initial dual magnitude y-bar.

Task is fixed to Max-Cut. Every factor other than (g_type, dual_init, seed) is held
constant so the experiment isolates the effect of g and y-bar:

    fixed: task=maxcut, instance set, optimizer(rmsprop), batch B, primal_lr a,
           dual_lr b, max_iters T, primal_init, no perturbation / rounding / refine.
    varied: g_type (13 forms: 10 convex + 2 partially convex + 1 linear control),
            dual_init (y-bar) over a grid, seed.

Budget is by iteration count (max_iters), matching the paper's setting.

The Max-Cut QUBO is min f(x) = -cut(x), so we report cut = -objective (larger is better).

Each run is tagged with a convergence verdict (binary settling + objective plateau).
Runs that did not converge within the budget are excluded from the best-y-bar
selection, so a still-climbing run is never compared against a settled one.

Example (run from the repository root):
    python scripts/sweep_g_y.py --gset_ids 1 70 81 \
        --ybar 0 1 2 4 6 8 10 --seeds 0 1 2 \
        --batch 100 --lr_x 0.025 --lr_y 0.025 --max_iters 5000 --out sweep.csv
"""
import argparse
import csv
import os
import sys
import time

# allow running as "python scripts/sweep_g_y.py" from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbo import PDBOSolver, generate_max_cut, parse_gset  # noqa: E402
from solver_jax import G_TYPES  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Sweep g(x) and y-bar for Max-Cut (fixed everything else).")
    p.add_argument("--gset_ids", type=int, nargs="+", default=[1],
                   help="Gset instance ids; the .txt files must exist in ./instance/Gset/")
    p.add_argument("--g", dest="g_types", nargs="+", default=list(G_TYPES),
                   choices=list(G_TYPES),
                   help="constraint functions to sweep (10 convex + 2 partially convex + 1 linear control)")
    p.add_argument("--g_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="rescale every g to value range [-1, 0] to compare shape not scale "
                        "(default on; use --no-g_normalize for the raw literature forms)")
    p.add_argument("--ybar", type=float, nargs="+", default=[0, 1, 2, 4, 6, 8, 10],
                   help="initial dual magnitude grid (paper notation y-bar; solver arg dual_init)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # --- fixed algorithm hyper-parameters (do not vary within one experiment) ---
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--lr_x", type=float, default=0.025)
    p.add_argument("--lr_y", type=float, default=0.025)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--primal_init", choices=["uniform", "half", "binary"], default="uniform")
    # --- convergence gating: runs that did not converge within the budget are not
    #     comparable to converged ones and are excluded from the best-ybar selection ---
    p.add_argument("--integrality_tol", type=float, default=1e-3,
                   help="max final fractionality x*(1-x) of the best init to count as binary-converged")
    p.add_argument("--plateau_window", type=int, default=None,
                   help="objective is 'plateaued' if the last improvement was this many iters "
                        "before the budget end (default: 10%% of max_iters)")
    p.add_argument("--out", default="sweep_g_y.csv")
    return p


def main():
    args = build_parser().parse_args()

    # Load every requested instance once and cache its QUBO data.
    instances = {}
    for gid in args.gset_ids:
        path = f"./instance/Gset/G{gid}.txt"
        if not os.path.exists(path):
            print(f"[skip] missing instance file: {path}", file=sys.stderr)
            continue
        graph = parse_gset(str(gid))
        instances[gid] = generate_max_cut(graph)
    if not instances:
        print("No usable instances found. Put the G*.txt files in ./instance/Gset/.", file=sys.stderr)
        return 1

    plateau_window = args.plateau_window if args.plateau_window is not None else max(50, args.max_iters // 10)

    rows = []
    for gid, data in instances.items():
        for g_type in args.g_types:
            for ybar in args.ybar:
                cuts = []
                convs = []
                for seed in args.seeds:
                    solver = PDBOSolver(
                        n_vars=data["num_vars"],
                        objective_type="quadratic",
                        Q_indices=data["Q_indices"],
                        Q_values=data["Q_values"],
                        c=data["c"],
                        optimizer_type="rmsprop",
                        batch_size=args.batch,
                        primal_lr=args.lr_x,
                        dual_lr=args.lr_y,
                        dual_init=ybar,
                        g_type=g_type,
                        g_normalize=args.g_normalize,
                        max_iters=args.max_iters,
                        primal_init=args.primal_init,
                        seed=seed,
                        verbose=False,
                    )
                    t0 = time.perf_counter()
                    result = solver.optimize()
                    elapsed = time.perf_counter() - t0
                    cut = -float(result.objective)  # Max-Cut QUBO: min f = -cut

                    integ = solver.final_integrality_min
                    last_imp = solver.last_improvement_step
                    binary_conv = integ < args.integrality_tol
                    obj_plateaued = (args.max_iters - 1 - last_imp) >= plateau_window
                    converged = binary_conv and obj_plateaued
                    cuts.append(cut)
                    convs.append(converged)
                    rows.append({
                        "instance": f"G{gid}", "g": g_type, "ybar": ybar, "seed": seed,
                        "cut": cut, "objective": float(result.objective),
                        "converged": int(converged), "binary_conv": int(binary_conv),
                        "obj_plateaued": int(obj_plateaued),
                        "final_integrality": round(integ, 6), "last_improve_iter": last_imp,
                        "iters": args.max_iters, "batch": args.batch,
                        "lr_x": args.lr_x, "lr_y": args.lr_y,
                        "g_normalize": int(args.g_normalize), "time_s": round(elapsed, 3),
                    })
                mean_cut = sum(cuts) / len(cuts)
                n_conv = sum(convs)
                tag = "OK" if n_conv == len(convs) else ("PART" if n_conv else "NOCONV")
                print(f"G{gid:<4} g={g_type:<12} ybar={ybar:<5} "
                      f"cut mean={mean_cut:8.1f} best={max(cuts):.0f}  "
                      f"conv={n_conv}/{len(convs)} [{tag}]")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Aggregate per (instance, g, ybar): mean cut + how many seeds converged.
    agg = {}
    for r in rows:
        key = (r["instance"], r["g"], r["ybar"])
        d = agg.setdefault(key, {"cuts": [], "conv": 0, "n": 0})
        d["cuts"].append(r["cut"])
        d["conv"] += r["converged"]
        d["n"] += 1

    # Headline comparison: best y-bar per (instance, g), CONVERGED CELLS ONLY.
    # A cell qualifies only if every seed converged, so we never compare a run that
    # was still climbing at the budget against one that had settled.
    print("\n=== best y-bar per (instance, g), among fully-converged cells only ===")
    best = {}
    non_converged = []
    for (inst, g, ybar), d in agg.items():
        mean_cut = sum(d["cuts"]) / len(d["cuts"])
        if d["conv"] == d["n"]:
            k = (inst, g)
            if k not in best or mean_cut > best[k][1]:
                best[k] = (ybar, mean_cut)
        else:
            non_converged.append((inst, g, ybar, d["conv"], d["n"]))

    all_gs = {(r["instance"], r["g"]) for r in rows}
    for key in sorted(all_gs):
        inst, g = key
        if key in best:
            ybar, m = best[key]
            print(f"{inst:<6} g={g:<12} best ybar={ybar:<5} mean cut={m:8.1f}")
        else:
            print(f"{inst:<6} g={g:<12} -- no fully-converged y-bar within budget --")

    if non_converged:
        print(f"\n[!] {len(non_converged)} (instance,g,ybar) cells did NOT fully converge "
              f"(excluded from best-ybar). Increase --max_iters or inspect the CSV:")
        for inst, g, ybar, c, n in sorted(non_converged):
            print(f"    {inst} g={g} ybar={ybar}: {c}/{n} seeds converged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
