"""Run the G8-G10 spectral-subspace patch with file-backed logs.

This small launcher is intentionally quiet: it redirects the child process'
stdout/stderr to files before the experiment script emits anything, so the
experiment can keep running after an interactive session is interrupted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    out_csv = repo / "spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.csv"
    out_log = repo / "spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.out.log"
    err_log = repo / "spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.err.log"

    cmd = [
        sys.executable,
        "scripts/compare_spectral_initializations.py",
        "--gset_ids",
        "8",
        "9",
        "10",
        "--seeds",
        "0",
        "1",
        "2",
        "--modes",
        "spectral_subspace_random",
        "--batch",
        "100",
        "--max_iters",
        "5000",
        "--check_every",
        "10",
        "--optimizer",
        "rmsprop",
        "--init_radius",
        "0.5",
        "--dual_init_mode",
        "curvature",
        "--hessian_level",
        "0",
        "--subspace_dim",
        "16",
        "--subspace_power_min",
        "0.5",
        "--subspace_power_max",
        "1.5",
        "--out",
        str(out_csv),
    ]

    with out_log.open("w", encoding="utf-8") as stdout, err_log.open("w", encoding="utf-8") as stderr:
        return subprocess.call(cmd, cwd=repo, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    raise SystemExit(main())
