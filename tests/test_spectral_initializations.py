import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pdbo import PDBOSolver, generate_max_cut, random_graph


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_spectral_initializations.py"
SPEC = importlib.util.spec_from_file_location("compare_spectral_initializations", SCRIPT_PATH)
spectral_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spectral_script)


def _tiny_spectral_data():
    values = np.array([-3.0, -1.0, 2.0, 4.0], dtype=np.float64)
    vectors = np.eye(4, dtype=np.float64)
    return spectral_script.SpectralData(values=values, vectors=vectors)


def test_custom_primal_initialization_broadcasts_and_sgd_smoke():
    data = generate_max_cut(random_graph(n=8, d=3, seed=0))
    primal0 = np.linspace(0.1, 0.9, data["num_vars"], dtype=np.float32)

    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        optimizer_type="sgd",
        batch_size=3,
        primal_lr=0.01,
        dual_lr=0.01,
        primal_initial_values=primal0,
        max_iters=2,
        check_every=2,
        verbose=False,
    )

    np.testing.assert_allclose(np.asarray(solver.primal), np.broadcast_to(primal0, (3, primal0.size)))
    result = solver.optimize()
    assert result.incumbent.shape == (data["num_vars"],)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.array([-0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ],
)
def test_custom_primal_initialization_rejects_invalid_values(bad_values):
    data = generate_max_cut(random_graph(n=4, d=2, seed=0))
    with pytest.raises((TypeError, ValueError)):
        PDBOSolver(
            n_vars=data["num_vars"],
            objective_type="quadratic",
            Q_indices=data["Q_indices"],
            Q_values=data["Q_values"],
            c=data["c"],
            batch_size=2,
            primal_initial_values=bad_values,
            max_iters=1,
            verbose=False,
        )


def test_min_eig_initialization_has_only_v1_energy():
    spectral = _tiny_spectral_data()
    primal = spectral_script.make_initial_primal(
        "min_eig",
        spectral,
        batch=4,
        seed=0,
        radius=0.4,
    )

    assert np.all(primal >= 0.0)
    assert np.all(primal <= 1.0)
    diagnostics = spectral_script.spectral_energy_diagnostics(primal, spectral, low_basis=4)
    assert diagnostics["energy_v1_mean"] == pytest.approx(1.0)
    assert diagnostics["energy_v2_mean"] == pytest.approx(0.0)


def test_ranked_spectral_coefficients_descend_and_initialization_stays_in_box():
    spectral = _tiny_spectral_data()
    coeffs = spectral_script.spectral_rank_coefficients(6, power=1.0)
    assert np.all(np.diff(coeffs) < 0.0)

    primal = spectral_script.make_initial_primal(
        "spectral_ranked",
        spectral,
        batch=3,
        seed=1,
        radius=0.45,
        mixture_power=1.0,
        mixture_signs="positive",
    )

    assert np.all(primal >= 0.0)
    assert np.all(primal <= 1.0)
    diagnostics = spectral_script.spectral_energy_diagnostics(primal, spectral, low_basis=4)
    assert diagnostics["energy_v1_mean"] > diagnostics["energy_v2_mean"]


def test_randomized_spectral_subspace_preserves_descending_magnitudes_and_batch_diversity():
    spectral = _tiny_spectral_data()
    primal = spectral_script.make_initial_primal(
        "spectral_subspace_random",
        spectral,
        batch=8,
        seed=4,
        radius=0.45,
        subspace_dim=4,
        subspace_power_min=0.5,
        subspace_power_max=1.5,
    )

    assert np.all(primal >= 0.0)
    assert np.all(primal <= 1.0)

    centered_abs = np.abs(primal.astype(np.float64) - 0.5)
    assert np.all(centered_abs[:, 0] > centered_abs[:, 1])
    assert np.all(centered_abs[:, 1] > centered_abs[:, 2])
    assert np.all(centered_abs[:, 2] > centered_abs[:, 3])
    assert np.unique(np.round(primal, decimals=8), axis=0).shape[0] > 2


def test_hybrid_random_spectral_combines_random_and_spectral_batches():
    spectral = _tiny_spectral_data()
    primal = spectral_script.make_initial_primal(
        "hybrid_random_spectral",
        spectral,
        batch=10,
        seed=5,
        radius=0.45,
        subspace_dim=4,
        subspace_power_min=0.5,
        subspace_power_max=1.5,
        hybrid_spectral_fraction=0.4,
    )

    assert primal.shape == (10, 4)
    assert np.all(primal >= 0.0)
    assert np.all(primal <= 1.0)
    assert np.unique(np.round(primal, decimals=8), axis=0).shape[0] > 6


def test_subset_random_handles_cached_basis_larger_than_subset():
    spectral = _tiny_spectral_data()
    primal = spectral_script.make_initial_primal(
        "spectral_subset_random",
        spectral,
        batch=6,
        seed=7,
        radius=0.45,
        subspace_dim=4,
        subset_size=2,
    )

    assert primal.shape == (6, 4)
    assert np.all(primal >= 0.0)
    assert np.all(primal <= 1.0)
    assert np.unique(np.round(primal, decimals=8), axis=0).shape[0] > 2
