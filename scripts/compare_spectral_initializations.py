"""Compare PDBO Max-Cut runs under controlled primal initializations.

The experiment compares four initial settings while keeping the solver
controls fixed:

    random            uniform samples around 1/2
    random_nonconvex  uniform samples around 1/2 with a separate Hessian level
    min_eig           only the minimum-eigenvalue eigenvector v1 is present
    spectral_ranked   descending coefficients on v1, v2, ...
    spectral_subspace_random
                      randomized directions inside the low-eigenvalue subspace
    spectral_single_random
                      one randomly selected low-eigenvalue vector per batch member
    spectral_subset_random
                      a few randomly selected low-eigenvalue vectors per batch member
    hybrid_random_spectral
                      part uniform random, part randomized low-spectral-subspace

Example:
    python scripts/compare_spectral_initializations.py --gset_ids 1 2 3 4 5 6 7 8 9 10 \
        --seeds 0 1 2 --batch 32 --max_iters 3000 --out spectral_init_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

# Allow running as "python scripts/compare_spectral_initializations.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdbo import PDBOSolver, generate_max_cut, parse_gset  # noqa: E402
from pdbo.curvature import quadratic_hessian  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402


MODES = (
    "random",
    "random_nonconvex",
    "min_eig",
    "spectral_ranked",
    "spectral_subspace_random",
    "spectral_single_random",
    "spectral_subset_random",
    "hybrid_random_spectral",
)

FIELDNAMES = (
    "instance",
    "mode",
    "seed",
    "status",
    "error",
    "n",
    "m",
    "batch",
    "unique_initializations",
    "optimizer",
    "lr_x",
    "lr_y",
    "max_iters",
    "g",
    "dual_init_mode",
    "dual_init",
    "hessian_level",
    "lambda_min",
    "lambda_second",
    "objective_curvature",
    "target_initial_lambda_min",
    "dual_init_expected",
    "dual_min",
    "dual_max",
    "init_radius",
    "miu_rms",
    "energy_v1_mean",
    "energy_v2_mean",
    "energy_low_basis_mean",
    "initial_cut_best",
    "initial_cut_mean",
    "final_cut",
    "final_objective",
    "improvement_from_initial_best",
    "final_integrality_min",
    "final_integrality_mean",
    "last_improve_iter",
    "stop_reason",
    "time_s",
)


@dataclass(frozen=True)
class SpectralData:
    values: np.ndarray
    vectors: np.ndarray

    @property
    def lambda_min(self) -> float:
        return float(self.values[0])

    @property
    def lambda_second(self) -> float:
        return float(self.values[1]) if self.values.size > 1 else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare random, pure-v1, and ranked spectral PDBO initializations."
    )
    parser.add_argument("--gset_ids", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--lr_x", type=float, default=0.025)
    parser.add_argument("--lr_y", type=float, default=0.025)
    parser.add_argument("--optimizer", choices=["sgd", "rmsprop", "adam"], default="sgd")
    parser.add_argument(
        "--dual_init_mode",
        choices=["constant", "curvature"],
        default="curvature",
        help="default y initialization mode for all modes unless overridden",
    )
    parser.add_argument(
        "--dual_init",
        type=float,
        default=0.0,
        help="default constant y value when dual_init_mode is constant",
    )
    parser.add_argument(
        "--hessian_level",
        type=float,
        default=-0.1,
        help=(
            "default relative initial Hessian level r. "
            "r > 0 is convex, r = 0 is the PSD boundary, -1 < r < 0 is nonconvex"
        ),
    )
    parser.add_argument(
        "--random_hessian_level",
        type=float,
        default=None,
        help="override r for random initialization, e.g. 0.1 for a convex paper-style baseline",
    )
    parser.add_argument(
        "--random_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for random initialization",
    )
    parser.add_argument(
        "--random_dual_init",
        type=float,
        default=None,
        help="override constant y value for random initialization",
    )
    parser.add_argument(
        "--random_nonconvex_hessian_level",
        type=float,
        default=None,
        help="override r for the separate nonconvex random initialization mode",
    )
    parser.add_argument(
        "--random_nonconvex_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for random_nonconvex initialization",
    )
    parser.add_argument(
        "--random_nonconvex_dual_init",
        type=float,
        default=None,
        help="override constant y value for random_nonconvex initialization",
    )
    parser.add_argument(
        "--min_eig_hessian_level",
        type=float,
        default=None,
        help="override r for pure minimum-eigenvector initialization",
    )
    parser.add_argument(
        "--min_eig_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for pure minimum-eigenvector initialization",
    )
    parser.add_argument(
        "--min_eig_dual_init",
        type=float,
        default=None,
        help="override constant y value for pure minimum-eigenvector initialization",
    )
    parser.add_argument(
        "--spectral_ranked_hessian_level",
        type=float,
        default=None,
        help="override r for ranked spectral-mixture initialization",
    )
    parser.add_argument(
        "--spectral_ranked_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for ranked spectral-mixture initialization",
    )
    parser.add_argument(
        "--spectral_ranked_dual_init",
        type=float,
        default=None,
        help="override constant y value for ranked spectral-mixture initialization",
    )
    parser.add_argument(
        "--spectral_subspace_random_hessian_level",
        type=float,
        default=None,
        help="override r for randomized low-spectral-subspace initialization",
    )
    parser.add_argument(
        "--spectral_subspace_random_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for randomized low-spectral-subspace initialization",
    )
    parser.add_argument(
        "--spectral_subspace_random_dual_init",
        type=float,
        default=None,
        help="override constant y value for randomized low-spectral-subspace initialization",
    )
    parser.add_argument(
        "--hybrid_random_spectral_hessian_level",
        type=float,
        default=None,
        help="override r for hybrid random/spectral initialization",
    )
    parser.add_argument(
        "--hybrid_random_spectral_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for hybrid random/spectral initialization",
    )
    parser.add_argument(
        "--hybrid_random_spectral_dual_init",
        type=float,
        default=None,
        help="override constant y value for hybrid random/spectral initialization",
    )
    parser.add_argument(
        "--spectral_single_random_hessian_level",
        type=float,
        default=None,
        help="override r for single-random-eigenvector initialization",
    )
    parser.add_argument(
        "--spectral_single_random_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for single-random-eigenvector initialization",
    )
    parser.add_argument(
        "--spectral_single_random_dual_init",
        type=float,
        default=None,
        help="override constant y value for single-random-eigenvector initialization",
    )
    parser.add_argument(
        "--spectral_subset_random_hessian_level",
        type=float,
        default=None,
        help="override r for few-random-eigenvectors initialization",
    )
    parser.add_argument(
        "--spectral_subset_random_dual_init_mode",
        choices=["constant", "curvature"],
        default=None,
        help="override y initialization mode for few-random-eigenvectors initialization",
    )
    parser.add_argument(
        "--spectral_subset_random_dual_init",
        type=float,
        default=None,
        help="override constant y value for few-random-eigenvectors initialization",
    )
    parser.add_argument(
        "--init_radius",
        type=float,
        default=0.45,
        help="max coordinate deviation from 1/2 for all initialization modes",
    )
    parser.add_argument(
        "--mixture_power",
        type=float,
        default=1.0,
        help="ranked spectral coefficient magnitude is 1 / rank**power",
    )
    parser.add_argument(
        "--mixture_signs",
        choices=["random", "positive"],
        default="random",
        help="sign pattern for ranked spectral coefficients across the batch",
    )
    parser.add_argument(
        "--subspace_dim",
        type=int,
        default=32,
        help="number of lowest-eigenvalue vectors used by spectral_subspace_random",
    )
    parser.add_argument(
        "--subspace_power_min",
        type=float,
        default=0.5,
        help="minimum sampled decay exponent p for |alpha_k| = 1 / k**p",
    )
    parser.add_argument(
        "--subspace_power_max",
        type=float,
        default=1.5,
        help="maximum sampled decay exponent p for |alpha_k| = 1 / k**p",
    )
    parser.add_argument(
        "--hybrid_spectral_fraction",
        type=float,
        default=0.5,
        help="fraction of hybrid batch initialized from the randomized spectral subspace",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=4,
        help="number of eigenvectors sampled by spectral_subset_random",
    )
    parser.add_argument(
        "--rounding_samples",
        type=int,
        default=0,
        help="optional final randomized rounding samples per relaxed trajectory",
    )
    parser.add_argument("--check_every", type=int, default=10)
    parser.add_argument("--eig_basis", type=int, default=32)
    parser.add_argument(
        "--spectral_basis",
        type=int,
        default=32,
        help="number of smallest Hessian eigenvectors to compute/cache",
    )
    parser.add_argument(
        "--spectral_cache_dir",
        default="results/spectral_cache",
        help="directory for cached low-eigenpair .npz files",
    )
    parser.add_argument(
        "--no_spectral_cache",
        action="store_true",
        help="recompute spectral data instead of reading/writing the cache",
    )
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--out", default="spectral_init_compare.csv")
    return parser


def _as_row_template(instance: str, mode: str, seed: int, args) -> dict:
    row = {name: "" for name in FIELDNAMES}
    row.update(
        {
            "instance": instance,
            "mode": mode,
            "seed": seed,
            "batch": args.batch,
            "optimizer": args.optimizer,
            "lr_x": args.lr_x,
            "lr_y": args.lr_y,
            "max_iters": args.max_iters,
            "g": "quad",
            "dual_init_mode": dual_init_mode_for_mode(mode, args),
            "dual_init": dual_init_for_mode(mode, args),
            "hessian_level": hessian_level_for_mode(mode, args),
            "init_radius": args.init_radius,
        }
    )
    return row


def hessian_level_for_mode(mode: str, args) -> float:
    overrides = {
        "random": args.random_hessian_level,
        "random_nonconvex": args.random_nonconvex_hessian_level,
        "min_eig": args.min_eig_hessian_level,
        "spectral_ranked": args.spectral_ranked_hessian_level,
        "spectral_subspace_random": args.spectral_subspace_random_hessian_level,
        "hybrid_random_spectral": args.hybrid_random_spectral_hessian_level,
        "spectral_single_random": args.spectral_single_random_hessian_level,
        "spectral_subset_random": args.spectral_subset_random_hessian_level,
    }
    value = overrides[mode]
    return args.hessian_level if value is None else value


def dual_init_mode_for_mode(mode: str, args) -> str:
    overrides = {
        "random": args.random_dual_init_mode,
        "random_nonconvex": args.random_nonconvex_dual_init_mode,
        "min_eig": args.min_eig_dual_init_mode,
        "spectral_ranked": args.spectral_ranked_dual_init_mode,
        "spectral_subspace_random": args.spectral_subspace_random_dual_init_mode,
        "hybrid_random_spectral": args.hybrid_random_spectral_dual_init_mode,
        "spectral_single_random": args.spectral_single_random_dual_init_mode,
        "spectral_subset_random": args.spectral_subset_random_dual_init_mode,
    }
    value = overrides[mode]
    return args.dual_init_mode if value is None else value


def dual_init_for_mode(mode: str, args) -> float:
    overrides = {
        "random": args.random_dual_init,
        "random_nonconvex": args.random_nonconvex_dual_init,
        "min_eig": args.min_eig_dual_init,
        "spectral_ranked": args.spectral_ranked_dual_init,
        "spectral_subspace_random": args.spectral_subspace_random_dual_init,
        "hybrid_random_spectral": args.hybrid_random_spectral_dual_init,
        "spectral_single_random": args.spectral_single_random_dual_init,
        "spectral_subset_random": args.spectral_subset_random_dual_init,
    }
    value = overrides[mode]
    return args.dual_init if value is None else value


def required_spectral_basis(args, n: int) -> int:
    if args.spectral_basis is not None:
        k = args.spectral_basis
    else:
        k = max(2, args.eig_basis, args.subspace_dim, args.subset_size)
        if "spectral_ranked" in args.modes:
            k = max(k, args.subspace_dim)
    if k < 2:
        raise ValueError("--spectral_basis must be at least 2")
    return min(k, n)


def _spectral_cache_path(instance: str, data: dict, k: int, args) -> str:
    safe_instance = instance.replace(os.sep, "_")
    filename = f"{safe_instance}_n{data['num_vars']}_m{data['num_edges']}_k{k}.npz"
    return os.path.join(args.spectral_cache_dir, filename)


def load_spectral_data(data: dict, instance: str, args) -> SpectralData:
    k = required_spectral_basis(args, data["num_vars"])
    cache_path = _spectral_cache_path(instance, data, k, args)
    if not args.no_spectral_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        values = cached["values"].astype(np.float64)
        vectors = cached["vectors"].astype(np.float64)
        if values.shape == (k,) and vectors.shape == (data["num_vars"], k):
            print(f"{instance}: loaded {k} cached Hessian eigenpairs from {cache_path}")
            return SpectralData(values=values, vectors=vectors)
        print(f"{instance}: ignoring incompatible spectral cache at {cache_path}")

    hessian = quadratic_hessian(data["Q_indices"], data["Q_values"], data["num_vars"])
    n = hessian.shape[0]
    if k >= n or n <= 256:
        dense = hessian.toarray()
        values, vectors = np.linalg.eigh(dense)
        values = values[:k]
        vectors = vectors[:, :k]
    else:
        values, vectors = eigsh(hessian, k=k, which="SA", tol=1e-8)
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]

    values = values.astype(np.float64)
    vectors = vectors.astype(np.float64)
    if not args.no_spectral_cache:
        os.makedirs(args.spectral_cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, values=values, vectors=vectors)
        print(f"{instance}: saved {k} Hessian eigenpairs to {cache_path}")
    return SpectralData(values=values, vectors=vectors)


def spectral_rank_coefficients(n: int, power: float) -> np.ndarray:
    if power < 0.0:
        raise ValueError("mixture_power must be non-negative")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return 1.0 / np.power(ranks, power)


def _scale_direction_to_box(direction: np.ndarray, radius: float) -> np.ndarray:
    max_abs = np.max(np.abs(direction), axis=1, keepdims=True)
    if np.any(max_abs <= 0.0) or not np.isfinite(max_abs).all():
        raise ValueError("initial direction contains a zero or non-finite vector")
    return 0.5 + radius * direction / max_abs


def make_initial_primal(
    mode: str,
    spectral: SpectralData,
    *,
    batch: int,
    seed: int,
    radius: float,
    mixture_power: float = 1.0,
    mixture_signs: str = "random",
    subspace_dim: int = 16,
    subspace_power_min: float = 0.5,
    subspace_power_max: float = 1.5,
    hybrid_spectral_fraction: float = 0.5,
    subset_size: int = 4,
) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"unknown initialization mode: {mode}")
    if batch < 1:
        raise ValueError("batch must be positive")
    if not (0.0 <= radius <= 0.5):
        raise ValueError("init_radius must lie in [0, 0.5]")

    rng = np.random.default_rng(seed)
    n = spectral.vectors.shape[0]
    if mode in {"random", "random_nonconvex"}:
        # Match PDBOSolver's default uniform initialization as closely as possible:
        # it splits PRNGKey(seed) once and uses the subkey for the primal batch.
        _, subkey = jax.random.split(jax.random.PRNGKey(seed))
        primal = jax.random.uniform(
            subkey,
            (batch, n),
            minval=0.5 - radius,
            maxval=0.5 + radius,
            dtype=jnp.float32,
        )
        return np.asarray(primal, dtype=np.float32)

    if mode == "hybrid_random_spectral":
        if not (0.0 <= hybrid_spectral_fraction <= 1.0):
            raise ValueError("hybrid_spectral_fraction must lie in [0, 1]")
        spectral_batch = int(round(batch * hybrid_spectral_fraction))
        random_batch = batch - spectral_batch
        parts = []
        if random_batch > 0:
            parts.append(
                make_initial_primal(
                    "random",
                    spectral,
                    batch=random_batch,
                    seed=seed,
                    radius=radius,
                )
            )
        if spectral_batch > 0:
            parts.append(
                make_initial_primal(
                    "spectral_subspace_random",
                    spectral,
                    batch=spectral_batch,
                    seed=seed + 1_000_003,
                    radius=radius,
                    subspace_dim=subspace_dim,
                    subspace_power_min=subspace_power_min,
                    subspace_power_max=subspace_power_max,
                )
            )
        return np.concatenate(parts, axis=0).astype(np.float32)

    if mode == "spectral_single_random":
        if subspace_dim < 1:
            raise ValueError("subspace_dim must be positive")
        k = min(subspace_dim, spectral.vectors.shape[1])
        choices = rng.integers(0, k, size=batch)
        signs = rng.choice(np.array([-1.0, 1.0]), size=batch)
        directions = spectral.vectors[:, choices].T * signs[:, np.newaxis]
        return _scale_direction_to_box(directions, radius).astype(np.float32)

    if mode == "min_eig":
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, 1))
        directions = signs * spectral.vectors[:, 0][np.newaxis, :]
        return _scale_direction_to_box(directions, radius).astype(np.float32)

    if mode == "spectral_ranked":
        k = spectral.vectors.shape[1]
        coeff = spectral_rank_coefficients(k, mixture_power)
        coeffs = np.broadcast_to(coeff, (batch, k)).copy()
        if mixture_signs == "random":
            coeffs *= rng.choice(np.array([-1.0, 1.0]), size=(batch, k))
        elif mixture_signs != "positive":
            raise ValueError("mixture_signs must be 'random' or 'positive'")
        directions = coeffs @ spectral.vectors[:, :k].T
        return _scale_direction_to_box(directions, radius).astype(np.float32)

    if subspace_dim < 1:
        raise ValueError("subspace_dim must be positive")
    if subspace_power_min <= 0.0 or subspace_power_max <= 0.0:
        raise ValueError("subspace decay powers must be positive")
    if subspace_power_min > subspace_power_max:
        raise ValueError("subspace_power_min must be <= subspace_power_max")

    k = min(subspace_dim, spectral.vectors.shape[1])
    if mode == "spectral_subset_random":
        if subset_size < 1:
            raise ValueError("subset_size must be positive")
        m = min(subset_size, k)
        coeffs = np.zeros((batch, k), dtype=np.float64)
        for row in range(batch):
            chosen = np.sort(rng.choice(k, size=m, replace=False))
            powers = rng.uniform(subspace_power_min, subspace_power_max)
            coeffs[row, chosen] = 1.0 / np.power(chosen + 1.0, powers)
            coeffs[row, chosen] *= rng.choice(np.array([-1.0, 1.0]), size=m)
        directions = coeffs @ spectral.vectors[:, :k].T
        return _scale_direction_to_box(directions, radius).astype(np.float32)

    ranks = np.arange(1, k + 1, dtype=np.float64)
    powers = rng.uniform(subspace_power_min, subspace_power_max, size=(batch, 1))
    coeffs = np.zeros((batch, k), dtype=np.float64)
    coeffs[:, :k] = 1.0 / np.power(ranks[np.newaxis, :], powers)
    coeffs[:, :k] *= rng.choice(np.array([-1.0, 1.0]), size=(batch, k))
    directions = coeffs @ spectral.vectors[:, :k].T
    return _scale_direction_to_box(directions, radius).astype(np.float32)


def spectral_energy_diagnostics(
    primal: np.ndarray,
    spectral: SpectralData,
    *,
    low_basis: int,
) -> dict:
    miu = np.asarray(primal, dtype=np.float64) - 0.5
    coeffs = miu @ spectral.vectors
    energy = np.sum(coeffs * coeffs, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(energy[:, np.newaxis] > 0.0, (coeffs * coeffs) / energy[:, np.newaxis], 0.0)
    k = min(max(int(low_basis), 1), spectral.vectors.shape[1])
    return {
        "miu_rms": float(np.sqrt(np.mean(miu * miu))),
        "energy_v1_mean": float(frac[:, 0].mean()),
        "energy_v2_mean": float(frac[:, 1].mean()) if frac.shape[1] > 1 else float("nan"),
        "energy_low_basis_mean": float(frac[:, :k].sum(axis=1).mean()),
    }


def maxcut_objective_values(data: dict, candidates: np.ndarray) -> np.ndarray:
    rows = data["Q_indices"][0]
    cols = data["Q_indices"][1]
    q_values = data["Q_values"]
    linear = data["c"]
    x = np.asarray(candidates, dtype=np.float64)
    quad = np.sum(x[:, rows] * q_values[np.newaxis, :] * x[:, cols], axis=1)
    return quad + x @ linear


def _count_unique_rows(values: np.ndarray) -> int:
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals=10)
    return int(np.unique(rounded, axis=0).shape[0])


def run_one(instance: str, data: dict, spectral: SpectralData, mode: str, seed: int, args) -> dict:
    row = _as_row_template(instance, mode, seed, args)
    n = data["num_vars"]
    hessian_level = hessian_level_for_mode(mode, args)
    dual_init_mode = dual_init_mode_for_mode(mode, args)
    dual_init = dual_init_for_mode(mode, args)
    objective_curvature = -spectral.lambda_min
    if dual_init_mode == "curvature":
        target_initial_lambda_min = hessian_level * objective_curvature
        expected_dual = (1.0 + hessian_level) * objective_curvature / 2.0
    else:
        target_initial_lambda_min = spectral.lambda_min + 2.0 * dual_init
        expected_dual = dual_init
    row.update(
        {
            "n": n,
            "m": data["num_edges"],
            "lambda_min": spectral.lambda_min,
            "lambda_second": spectral.lambda_second,
            "objective_curvature": objective_curvature,
            "target_initial_lambda_min": target_initial_lambda_min,
            "dual_init_expected": expected_dual,
        }
    )

    started = time.perf_counter()
    try:
        primal0 = make_initial_primal(
            mode,
            spectral,
            batch=args.batch,
            seed=seed,
            radius=args.init_radius,
            mixture_power=args.mixture_power,
            mixture_signs=args.mixture_signs,
            subspace_dim=args.subspace_dim,
            subspace_power_min=args.subspace_power_min,
            subspace_power_max=args.subspace_power_max,
            hybrid_spectral_fraction=args.hybrid_spectral_fraction,
            subset_size=args.subset_size,
        )
        rounded0 = np.rint(primal0).astype(np.float64)
        initial_objectives = maxcut_objective_values(data, rounded0)
        initial_best_objective = float(initial_objectives.min())
        diagnostics = spectral_energy_diagnostics(primal0, spectral, low_basis=args.eig_basis)

        solver = PDBOSolver(
            n_vars=n,
            objective_type="quadratic",
            Q_indices=data["Q_indices"],
            Q_values=data["Q_values"],
            c=data["c"],
            optimizer_type=args.optimizer,
            batch_size=args.batch,
            primal_lr=args.lr_x,
            dual_lr=args.lr_y,
            dual_init=dual_init,
            dual_init_mode=dual_init_mode,
            hessian_init_level=hessian_level,
            trusted_objective_hessian_lambda_min=spectral.lambda_min,
            g_type="quad",
            g_normalize=False,
            max_iters=args.max_iters,
            primal_init="half",
            primal_initial_values=primal0,
            rounding_samples=args.rounding_samples,
            check_every=args.check_every,
            seed=seed,
            verbose=False,
        )
        result = solver.optimize()
        final_cut = -float(result.objective)
        row.update(
            {
                "status": "ok",
                "error": "",
                "unique_initializations": _count_unique_rows(primal0),
                "dual_min": solver.initial_dual_min,
                "dual_max": solver.initial_dual_max,
                "initial_cut_best": -initial_best_objective,
                "initial_cut_mean": float((-initial_objectives).mean()),
                "final_cut": final_cut,
                "final_objective": float(result.objective),
                "improvement_from_initial_best": final_cut - (-initial_best_objective),
                "final_integrality_min": solver.final_integrality_min,
                "final_integrality_mean": solver.final_integrality,
                "last_improve_iter": solver.last_improvement_step,
                "stop_reason": result.stop_reason,
                "time_s": round(time.perf_counter() - started, 6),
                **diagnostics,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " "),
                "time_s": round(time.perf_counter() - started, 6),
            }
        )
    return row


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


def main() -> int:
    args = build_parser().parse_args()
    for name, value in (
        ("--hessian_level", args.hessian_level),
        ("--random_hessian_level", args.random_hessian_level),
        ("--random_nonconvex_hessian_level", args.random_nonconvex_hessian_level),
        ("--min_eig_hessian_level", args.min_eig_hessian_level),
        ("--spectral_ranked_hessian_level", args.spectral_ranked_hessian_level),
        ("--spectral_subspace_random_hessian_level", args.spectral_subspace_random_hessian_level),
        ("--hybrid_random_spectral_hessian_level", args.hybrid_random_spectral_hessian_level),
        ("--spectral_single_random_hessian_level", args.spectral_single_random_hessian_level),
        ("--spectral_subset_random_hessian_level", args.spectral_subset_random_hessian_level),
    ):
        if value is not None and value < -1.0:
            print(f"{name} must be at least -1", file=sys.stderr)
            return 2
    if args.subspace_dim < 1:
        print("--subspace_dim must be positive", file=sys.stderr)
        return 2
    if args.subspace_power_min <= 0.0 or args.subspace_power_max <= 0.0:
        print("--subspace_power_min and --subspace_power_max must be positive", file=sys.stderr)
        return 2
    if args.subspace_power_min > args.subspace_power_max:
        print("--subspace_power_min must be <= --subspace_power_max", file=sys.stderr)
        return 2
    if not (0.0 <= args.hybrid_spectral_fraction <= 1.0):
        print("--hybrid_spectral_fraction must be in [0, 1]", file=sys.stderr)
        return 2
    if args.subset_size < 1:
        print("--subset_size must be positive", file=sys.stderr)
        return 2

    output_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_dir, exist_ok=True)

    try:
        instances = list(_iter_instances(args.gset_ids, args.allow_missing))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not instances:
        print("No usable instances found.", file=sys.stderr)
        return 1

    rows = []
    with open(args.out, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        for gid, data in instances:
            instance = f"G{gid}"
            k = required_spectral_basis(args, data["num_vars"])
            print(f"{instance}: loading/computing {k} smallest Hessian eigenpairs ...")
            spectral = load_spectral_data(data, instance, args)
            for seed in args.seeds:
                for mode in args.modes:
                    row = run_one(instance, data, spectral, mode, seed, args)
                    writer.writerow(row)
                    output.flush()
                    rows.append(row)
                    if row["status"] == "ok":
                        print(
                            f"{instance:<4} seed={seed:<3} mode={mode:<15} "
                            f"init={row['initial_cut_best']:>8.1f} final={row['final_cut']:>8.1f} "
                            f"v1={row['energy_v1_mean']:.3f}"
                        )
                    else:
                        print(
                            f"{instance:<4} seed={seed:<3} mode={mode:<15} [{row['error']}]",
                            file=sys.stderr,
                        )

    failed = sum(row["status"] != "ok" for row in rows)
    print(f"\nWrote {len(rows)} rows to {args.out}; failures={failed}.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    # Make sure any first-run JAX backend information appears before the sweep log.
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()))
    sys.exit(main())
