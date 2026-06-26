"""Run PDBO on QOBLIB LABS instances with the original PUBO objective."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdbo import LABSPuboSolver, evaluate_labs_bits  # noqa: E402


QOBLIB_HEADER = [
    "Problem",
    "Submitter",
    "Date",
    "Reference",
    "Best Objective Value",
    "Optimality Bound",
    "Modeling Approach",
    "# Decision Variables",
    "# Binary Variables",
    "# Integer Variables",
    "# Continuous Variables",
    "# Non-Zero Coefficients",
    "Coefficients Type",
    "Coefficients Range",
    "Workflow",
    "Algorithm Type",
    "# Runs",
    "# Feasible Runs",
    "# Successful Runs",
    "Success Threshold",
    "Hardware Specifications",
    "Total Runtime",
    "CPU Runtime",
    "GPU Runtime",
    "QPU Runtime",
    "Other HW Runtime",
    "Remarks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qoblib-root", type=Path, default=Path("/home/lwb/project/PDBO_LABS/QOBLIB/02-labs"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-start", type=int, default=2)
    parser.add_argument("--n-end", type=int, default=100)
    parser.add_argument("--timelimit", type=float, default=180.0)
    parser.add_argument("--max-iters", type=int, default=1_000_000_000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--primal-lr", type=float, default=0.03)
    parser.add_argument("--dual-lr", type=float, default=0.03)
    parser.add_argument("--dual-init", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--primal-init", choices=["uniform", "half", "binary"], default="uniform")
    parser.add_argument("--submitter", default="PDBO")
    parser.add_argument("--reference", default="PDBO LABS PUBO run")
    parser.add_argument("--resume", action="store_true", help="Skip instances with all expected output files.")
    parser.add_argument("--reverse", action="store_true", help="Run instances from n-end down to n-start.")
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def instance_names(n_start: int, n_end: int, reverse: bool = False) -> Iterable[tuple[int, str]]:
    ns = range(n_end, n_start - 1, -1) if reverse else range(n_start, n_end + 1)
    for n in ns:
        yield n, f"labs{n:03d}"


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "NVIDIA GPU; JAX"
    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "NVIDIA GPU"
    return f"{first}; JAX"


def write_solution(path: Path, energy: int, bits: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Energy: {energy}\n")
        f.write("# Source: PDBO\n")
        for bit in bits.astype(int).tolist():
            f.write(f"{bit}\n")


def write_time_series(path: Path, timings: list[float], values: list[float], runtime: float, final_value: int) -> None:
    series = [
        {"Time": round(float(t), 6), "Incumbent": int(round(float(v)))}
        for t, v in zip(timings, values)
    ]
    if not series or series[-1]["Incumbent"] != final_value or series[-1]["Time"] < runtime:
        series.append({"Time": round(float(runtime), 6), "Incumbent": int(final_value)})
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump([series], f, separators=(",", ":"))


def write_summary(path: Path, args: argparse.Namespace, n: int, name: str, energy: int, runtime: float, hardware: str) -> None:
    remarks = (
        f"model=pubo; primal_init={args.primal_init}; dual_init={args.dual_init}; "
        f"primal_lr={args.primal_lr}; dual_lr={args.dual_lr}; batch={args.batch_size}; "
        f"seed={args.seed}; optimizer={args.optimizer}; timelimit={args.timelimit:g}; no local search"
    )
    row = [
        name,
        args.submitter,
        date.today().isoformat(),
        args.reference,
        energy,
        "N/A",
        "PUBO",
        n,
        n,
        0,
        0,
        "N/A",
        "integer",
        "N/A",
        "PDBO",
        "heuristic",
        1,
        1,
        1,
        "N/A",
        hardware,
        round(float(runtime), 6),
        0,
        round(float(runtime), 6),
        0,
        0,
        remarks,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(QOBLIB_HEADER)
        writer.writerow(row)


def check_solution(checker: Path, n: int, solution_file: Path, expected_energy: int) -> str:
    result = subprocess.run(
        [str(checker), str(n), str(solution_file)],
        check=False,
        text=True,
        capture_output=True,
    )
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"E\(S\)=(-?\d+)", output)
    if match is None:
        raise RuntimeError(f"checker did not report an energy for n={n}: {output}")
    checker_energy = int(match.group(1))
    if checker_energy != expected_energy:
        raise RuntimeError(f"energy mismatch for n={n}: solver={expected_energy}, checker={checker_energy}")
    if result.returncode != 0 and "NOT OPTIMAL" not in output:
        raise RuntimeError(f"checker failed for n={n}: {output}")
    return output


def output_complete(instance_dir: Path, name: str) -> bool:
    return (
        (instance_dir / f"{name}_solution.sol").is_file()
        and (instance_dir / f"{name}_summary.csv").is_file()
        and (instance_dir / f"{name}_objective_time_series.json.gz").is_file()
    )


def run_one(args: argparse.Namespace, n: int, name: str, submission_root: Path, hardware: str) -> dict[str, object]:
    instance_dir = submission_root / name
    if args.resume and output_complete(instance_dir, name):
        return {"Problem": name, "Status": "skipped"}

    instance_dir.mkdir(parents=True, exist_ok=True)
    solver = LABSPuboSolver(
        n_vars=n,
        optimizer_type=args.optimizer,
        batch_size=args.batch_size,
        primal_lr=args.primal_lr,
        dual_lr=args.dual_lr,
        dual_init=args.dual_init,
        max_iters=args.max_iters,
        timelimit=args.timelimit,
        seed=args.seed,
        verbose=args.verbose_solver,
        primal_init=args.primal_init,
    )
    solver.optimize()

    bits = np.asarray(solver.incumbent, dtype=np.int32)
    energy = int(evaluate_labs_bits(bits))
    runtime = float(solver.solving_time)

    solution_file = instance_dir / f"{name}_solution.sol"
    summary_file = instance_dir / f"{name}_summary.csv"
    series_file = instance_dir / f"{name}_objective_time_series.json.gz"

    write_solution(solution_file, energy, bits)
    write_time_series(series_file, solver.timing_record, solver.objVal_record, runtime, energy)
    write_summary(summary_file, args, n, name, energy, runtime, hardware)

    checker_output = check_solution(args.qoblib_root / "check/target/release/check_labs", n, solution_file, energy)
    return {
        "Problem": name,
        "Status": "ok",
        "Energy": energy,
        "Runtime": round(runtime, 6),
        "Checker": checker_output,
    }


def append_manifest(path: Path, result: dict[str, object]) -> None:
    exists = path.is_file()
    fieldnames = ["Problem", "Status", "Energy", "Runtime", "Checker"]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: result.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        args.output_root = args.qoblib_root / "submissions" / f"{date.today():%Y%m%d}_PDBO_PUBO"
    checker = args.qoblib_root / "check/target/release/check_labs"
    if not checker.is_file():
        raise FileNotFoundError(f"checker not found: {checker}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "manifest.csv"
    hardware = gpu_name()

    print(f"output_root={args.output_root}", flush=True)
    print(
        "params="
        f"timelimit={args.timelimit}, max_iters={args.max_iters}, batch={args.batch_size}, "
        f"primal_lr={args.primal_lr}, dual_lr={args.dual_lr}, dual_init={args.dual_init}, seed={args.seed}",
        flush=True,
    )
    for n, name in instance_names(args.n_start, args.n_end, args.reverse):
        print(f"[start] {name}", flush=True)
        result = run_one(args, n, name, args.output_root, hardware)
        append_manifest(manifest, result)
        if result["Status"] == "skipped":
            print(f"[skip]  {name}", flush=True)
        else:
            print(f"[done]  {name} energy={result['Energy']} runtime={result['Runtime']} checker={result['Checker']}", flush=True)


if __name__ == "__main__":
    main()
