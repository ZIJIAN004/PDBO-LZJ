"""Compare stagnation-triggered reopening strategies on Max-Cut.

Every mode uses the paper-aligned PDBO settings and the same random seed. The
trajectory is unchanged until a late incumbent plateau triggers an intervention.

Example:
    python scripts/compare_reopening_strategies.py \
        --gset_ids 67 70 72 77 81 --seeds 0 1 2 \
        --batch 100 --max_iters 5000 \
        --out_prefix results/reopening_g67_g81
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Iterable

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbo import PDBOSolver, generate_max_cut, parse_gset  # noqa: E402


MODES = (
    "baseline",
    "random_kick",
    "dual_zero_once",
    "dual_zero_hold",
    "dual_positive_hold",
    "random_branch",
    "gain_branch",
)

RUN_FIELDS = (
    "instance",
    "mode",
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
    "trigger_patience",
    "trigger_min_step",
    "event_cooldown",
    "max_events",
    "active_batch_fraction",
    "variables_per_row",
    "hold_steps",
    "branch_strength",
    "positive_dual",
    "final_cut",
    "last_improve_iter",
    "final_integrality_mean",
    "final_integrality_min",
    "final_improving_one_flips",
    "final_best_one_flip_cut_gain",
    "event_count",
    "first_event_iter",
    "cut_at_first_event",
    "cut_improvement_after_first_event",
    "first_event_best_one_flip_cut_gain",
    "first_event_rows_with_improving_flip_fraction",
    "stop_reason",
    "time_s",
)

EVENT_FIELDS = (
    "instance",
    "mode",
    "seed",
    "event_index",
    "step",
    "cut_before",
    "final_cut",
    "cut_improvement_after_event",
    "selected_rows",
    "variables_per_row",
    "selected_coordinates",
    "dual_target",
    "hold_steps",
    "best_one_flip_cut_gain",
    "improving_flips_per_row_mean",
    "rows_with_improving_flip_fraction",
    "selected_cut_gain_mean",
    "selected_dual_before_mean",
)

SUMMARY_FIELDS = (
    "instance",
    "mode",
    "ok_runs",
    "cut_mean",
    "cut_std",
    "cut_best",
    "delta_vs_baseline_mean",
    "delta_vs_baseline_std",
    "wins_vs_baseline",
    "ties_vs_baseline",
    "losses_vs_baseline",
    "post_event_improvement_rate",
    "one_flip_local_optimum_rate",
    "last_improve_iter_mean",
    "time_s_mean",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare selective dual reopening and branch strategies for PDBO Max-Cut."
    )
    parser.add_argument("--gset_ids", type=int, nargs="+", default=[67, 70, 72, 77, 81])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr_x", type=float, default=0.025)
    parser.add_argument("--lr_y", type=float, default=0.025)
    parser.add_argument("--dual_init", type=float, default=6.0)
    parser.add_argument("--optimizer", choices=["sgd", "rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--check_every", type=int, default=1)
    parser.add_argument("--trigger_patience", type=int, default=500)
    parser.add_argument("--trigger_min_step", type=int, default=2500)
    parser.add_argument("--event_cooldown", type=int, default=750)
    parser.add_argument("--max_events", type=int, default=2)
    parser.add_argument("--active_batch_fraction", type=float, default=0.5)
    parser.add_argument("--variables_per_row", type=int, default=8)
    parser.add_argument("--hold_steps", type=int, default=100)
    parser.add_argument("--branch_strength", type=float, default=0.1)
    parser.add_argument("--positive_dual", type=float, default=0.5)
    parser.add_argument("--flip_tolerance", type=float, default=1e-6)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_prefix", default="results/reopening_g67_g81")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch < 1 or args.max_iters < 1 or args.check_every < 1:
        raise ValueError("batch, max_iters, and check_every must be positive")
    if args.trigger_patience < 1 or args.trigger_min_step < 0:
        raise ValueError("trigger_patience must be positive and trigger_min_step non-negative")
    if args.event_cooldown < 1 or args.max_events < 0:
        raise ValueError("event_cooldown must be positive and max_events non-negative")
    if not (0.0 < args.active_batch_fraction <= 1.0):
        raise ValueError("active_batch_fraction must lie in (0, 1]")
    if args.variables_per_row < 1 or args.hold_steps < 1:
        raise ValueError("variables_per_row and hold_steps must be positive")
    if not (0.0 <= args.branch_strength <= 0.5):
        raise ValueError("branch_strength must lie in [0, 0.5]")
    if args.positive_dual <= 0.0 or not np.isfinite(args.positive_dual):
        raise ValueError("positive_dual must be positive and finite")
    if args.flip_tolerance < 0.0:
        raise ValueError("flip_tolerance must be non-negative")


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


def final_flip_audit(data: dict, bits: np.ndarray, tolerance: float) -> tuple[int, float]:
    """Return improving one-flip count and best cut gain for a binary QUBO point."""
    x = np.asarray(bits, dtype=np.float64)
    q = data["Q_sparse"].tocsr().astype(np.float64)
    gradient = q.dot(x) + q.T.dot(x) + np.asarray(data["c"], dtype=np.float64)
    direction = 1.0 - 2.0 * x
    objective_gains = direction * gradient + q.diagonal()
    improving = int(np.count_nonzero(objective_gains < -tolerance))
    best_cut_gain = -float(np.min(objective_gains))
    return improving, best_cut_gain


def _empty_run_row(instance: str, mode: str, seed: int, data: dict, args) -> dict:
    row = {field: "" for field in RUN_FIELDS}
    row.update(
        {
            "instance": instance,
            "mode": mode,
            "seed": seed,
            "n": data["num_vars"],
            "m": data["num_edges"],
            "batch": args.batch,
            "optimizer": args.optimizer,
            "lr_x": args.lr_x,
            "lr_y": args.lr_y,
            "dual_init": args.dual_init,
            "max_iters": args.max_iters,
            "trigger_patience": args.trigger_patience,
            "trigger_min_step": args.trigger_min_step,
            "event_cooldown": args.event_cooldown,
            "max_events": args.max_events,
            "active_batch_fraction": args.active_batch_fraction,
            "variables_per_row": args.variables_per_row,
            "hold_steps": args.hold_steps,
            "branch_strength": args.branch_strength,
            "positive_dual": args.positive_dual,
        }
    )
    return row


def run_one(instance: str, data: dict, mode: str, seed: int, args) -> tuple[dict, list[dict]]:
    row = _empty_run_row(instance, mode, seed, data, args)
    started = time.perf_counter()
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
            perturbation=False,
            reopening_mode="none" if mode == "baseline" else mode,
            reopening_patience=args.trigger_patience,
            reopening_min_step=args.trigger_min_step,
            reopening_cooldown=args.event_cooldown,
            reopening_max_events=args.max_events,
            reopening_fraction=args.active_batch_fraction,
            reopening_variables=args.variables_per_row,
            reopening_hold_steps=args.hold_steps,
            reopening_strength=args.branch_strength,
            reopening_dual_value=0.0,
            reopening_positive_value=args.positive_dual,
            reopening_reset_optimizer=True,
            seed=seed,
            verbose=False,
        )
        result = solver.optimize()
        final_cut = -float(result.objective)
        improving_flips, best_flip_cut_gain = final_flip_audit(
            data,
            result.incumbent,
            args.flip_tolerance,
        )
        events = []
        for event in solver.reopening_events:
            cut_before = -float(event["objective_before"])
            events.append(
                {
                    "instance": instance,
                    "mode": mode,
                    "seed": seed,
                    "event_index": event["event_index"],
                    "step": event["step"],
                    "cut_before": cut_before,
                    "final_cut": final_cut,
                    "cut_improvement_after_event": final_cut - cut_before,
                    "selected_rows": event["selected_rows"],
                    "variables_per_row": event["variables_per_row"],
                    "selected_coordinates": event["selected_coordinates"],
                    "dual_target": event["dual_target"],
                    "hold_steps": event["hold_steps"],
                    "best_one_flip_cut_gain": -event["best_one_flip_objective_gain"],
                    "improving_flips_per_row_mean": event["improving_flips_per_row_mean"],
                    "rows_with_improving_flip_fraction": event[
                        "rows_with_improving_flip_fraction"
                    ],
                    "selected_cut_gain_mean": -event["selected_objective_gain_mean"],
                    "selected_dual_before_mean": event["selected_dual_before_mean"],
                }
            )

        first_event = events[0] if events else None
        row.update(
            {
                "status": "ok",
                "error": "",
                "final_cut": final_cut,
                "last_improve_iter": solver.last_improvement_step,
                "final_integrality_mean": solver.final_integrality,
                "final_integrality_min": solver.final_integrality_min,
                "final_improving_one_flips": improving_flips,
                "final_best_one_flip_cut_gain": best_flip_cut_gain,
                "event_count": len(events),
                "first_event_iter": first_event["step"] if first_event else "",
                "cut_at_first_event": first_event["cut_before"] if first_event else "",
                "cut_improvement_after_first_event": (
                    first_event["cut_improvement_after_event"] if first_event else ""
                ),
                "first_event_best_one_flip_cut_gain": (
                    first_event["best_one_flip_cut_gain"] if first_event else ""
                ),
                "first_event_rows_with_improving_flip_fraction": (
                    first_event["rows_with_improving_flip_fraction"] if first_event else ""
                ),
                "stop_reason": result.stop_reason,
                "time_s": round(time.perf_counter() - started, 6),
            }
        )
        return row, events
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " "),
                "time_s": round(time.perf_counter() - started, 6),
            }
        )
        return row, []


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else float("nan")


def _mean_std(values: list[float]) -> tuple[float | str, float | str]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return "", ""
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return float(finite.mean()), std


def _latest_rows(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["instance"], row["mode"], str(row["seed"]))] = row
    return list(latest.values())


def summarize(rows: list[dict], tolerance: float) -> list[dict]:
    ok_rows = [row for row in _latest_rows(rows) if row.get("status") == "ok"]
    baseline = {
        (row["instance"], str(row["seed"])): _float(row, "final_cut")
        for row in ok_rows
        if row["mode"] == "baseline"
    }
    grouped = defaultdict(list)
    for row in ok_rows:
        grouped[(row["instance"], row["mode"])].append(row)
        grouped[("ALL", row["mode"])].append(row)

    output = []
    for (instance, mode), group in sorted(grouped.items()):
        cuts = [_float(row, "final_cut") for row in group]
        deltas = []
        for row in group:
            key = (row["instance"], str(row["seed"]))
            if key in baseline:
                deltas.append(_float(row, "final_cut") - baseline[key])
        cut_mean, cut_std = _mean_std(cuts)
        delta_mean, delta_std = _mean_std(deltas)
        post_event = [
            _float(row, "cut_improvement_after_first_event")
            for row in group
            if np.isfinite(_float(row, "cut_improvement_after_first_event"))
        ]
        local_optima = [
            _float(row, "final_improving_one_flips") == 0.0 for row in group
        ]
        last_mean, _ = _mean_std([_float(row, "last_improve_iter") for row in group])
        time_mean, _ = _mean_std([_float(row, "time_s") for row in group])
        output.append(
            {
                "instance": instance,
                "mode": mode,
                "ok_runs": len(group),
                "cut_mean": "" if instance == "ALL" else cut_mean,
                "cut_std": "" if instance == "ALL" else cut_std,
                "cut_best": "" if instance == "ALL" else max(cuts),
                "delta_vs_baseline_mean": delta_mean,
                "delta_vs_baseline_std": delta_std,
                "wins_vs_baseline": sum(delta > tolerance for delta in deltas),
                "ties_vs_baseline": sum(abs(delta) <= tolerance for delta in deltas),
                "losses_vs_baseline": sum(delta < -tolerance for delta in deltas),
                "post_event_improvement_rate": (
                    float(np.mean(np.asarray(post_event) > tolerance)) if post_event else ""
                ),
                "one_flip_local_optimum_rate": float(np.mean(local_optima)),
                "last_improve_iter_mean": last_mean,
                "time_s_mean": time_mean,
            }
        )
    return output


def _read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def validate_resume_rows(rows: list[dict], args: argparse.Namespace) -> None:
    expected = {
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr_x": args.lr_x,
        "lr_y": args.lr_y,
        "dual_init": args.dual_init,
        "max_iters": args.max_iters,
        "trigger_patience": args.trigger_patience,
        "trigger_min_step": args.trigger_min_step,
        "event_cooldown": args.event_cooldown,
        "max_events": args.max_events,
        "active_batch_fraction": args.active_batch_fraction,
        "variables_per_row": args.variables_per_row,
        "hold_steps": args.hold_steps,
        "branch_strength": args.branch_strength,
        "positive_dual": args.positive_dual,
    }
    text_fields = {"optimizer"}
    for row in rows:
        for field, value in expected.items():
            observed = row.get(field, "")
            matches = (
                observed == str(value)
                if field in text_fields
                else observed != "" and np.isclose(float(observed), float(value))
            )
            if not matches:
                raise ValueError(
                    f"cannot resume: existing {field}={observed!r}, requested {value!r}. "
                    "Use a new out_prefix for a different configuration."
                )


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
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    run_path = prefix + "_runs.csv"
    event_path = prefix + "_events.csv"
    summary_path = prefix + "_summary.csv"
    config_path = prefix + "_config.json"

    previous_rows = _read_rows(run_path) if args.resume else []
    try:
        validate_resume_rows(previous_rows, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    completed = {
        (row["instance"], row["mode"], str(row["seed"]))
        for row in _latest_rows(previous_rows)
        if row.get("status") == "ok"
    }
    all_rows = list(previous_rows)
    file_mode = "a" if args.resume and os.path.exists(run_path) else "w"
    event_mode = "a" if args.resume and os.path.exists(event_path) else "w"

    with open(config_path, "w") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)
    print("Configuration:", json.dumps(vars(args), sort_keys=True))
    print("JAX devices:", jax.devices())

    failed = 0
    with (
        open(run_path, file_mode, newline="") as run_file,
        open(event_path, event_mode, newline="") as event_file,
    ):
        run_writer = csv.DictWriter(run_file, fieldnames=RUN_FIELDS)
        event_writer = csv.DictWriter(event_file, fieldnames=EVENT_FIELDS)
        if file_mode == "w":
            run_writer.writeheader()
        if event_mode == "w":
            event_writer.writeheader()

        for gid, data in instances:
            instance = f"G{gid}"
            for seed in args.seeds:
                for mode in args.modes:
                    key = (instance, mode, str(seed))
                    if key in completed:
                        print(f"{instance:<4} seed={seed:<3} mode={mode:<20} [resume: skipped]")
                        continue
                    row, events = run_one(instance, data, mode, seed, args)
                    all_rows.append(row)
                    run_writer.writerow(row)
                    run_file.flush()
                    for event in events:
                        event_writer.writerow(event)
                    event_file.flush()
                    if row["status"] == "ok":
                        delta = row["cut_improvement_after_first_event"]
                        delta_text = f" event_delta={float(delta):+.1f}" if delta != "" else ""
                        print(
                            f"{instance:<4} seed={seed:<3} mode={mode:<20} "
                            f"cut={float(row['final_cut']):>9.1f} "
                            f"best_iter={int(row['last_improve_iter']):<5} "
                            f"events={int(row['event_count'])}{delta_text}"
                        )
                    else:
                        failed += 1
                        print(
                            f"{instance} seed={seed} mode={mode} [{row['error']}]",
                            file=sys.stderr,
                        )

    summary = summarize(all_rows, args.flip_tolerance)
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    print("\nOverall paired comparison against baseline:")
    for row in summary:
        if row["instance"] != "ALL":
            continue
        delta = row["delta_vs_baseline_mean"]
        delta_text = f"{float(delta):+.3f}" if delta != "" else "n/a"
        print(
            f"  {row['mode']:<20} mean_delta={delta_text:>9} "
            f"W/T/L={row['wins_vs_baseline']}/{row['ties_vs_baseline']}/{row['losses_vs_baseline']}"
        )
    print(f"Runs: {run_path}")
    print(f"Events: {event_path}")
    print(f"Summary: {summary_path}")
    print(f"Completed with {failed} failed run(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
