import numpy as np
import pytest

from pdbo import PDBOSolver, generate_max_cut, random_graph
from scripts.compare_reopening_strategies import final_flip_audit


def _zero_qubo(n=4):
    indices = np.asarray([[0], [1]], dtype=np.int32)
    values = np.asarray([0.0], dtype=np.float32)
    return {
        "num_vars": n,
        "Q_indices": indices,
        "Q_values": values,
        "c": np.zeros(n, dtype=np.float32),
    }


def test_final_flip_audit_matches_brute_force():
    data = generate_max_cut(random_graph(n=8, d=2, seed=4))
    bits = np.asarray([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.float64)
    q = data["Q_sparse"].tocsr()

    def objective(x):
        return float(x @ q.dot(x) + data["c"] @ x)

    base = objective(bits)
    brute_gains = []
    for index in range(bits.size):
        candidate = bits.copy()
        candidate[index] = 1.0 - candidate[index]
        brute_gains.append(objective(candidate) - base)

    improving, best_cut_gain = final_flip_audit(data, bits, tolerance=1e-6)
    brute_gains = np.asarray(brute_gains)
    assert improving == int(np.count_nonzero(brute_gains < -1e-6))
    assert best_cut_gain == pytest.approx(-float(brute_gains.min()))


def test_dual_zero_hold_releases_only_selected_batch_entries():
    data = _zero_qubo()
    initial = np.asarray(
        [
            [0.2, 0.3, 0.7, 0.8],
            [0.8, 0.7, 0.3, 0.2],
            [0.2, 0.7, 0.3, 0.8],
            [0.8, 0.3, 0.7, 0.2],
        ],
        dtype=np.float32,
    )
    solver = PDBOSolver(
        n_vars=4,
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=4,
        primal_initial_values=initial,
        dual_init=-2.0,
        primal_lr=0.01,
        dual_lr=0.01,
        max_iters=4,
        check_every=1,
        reopening_mode="dual_zero_hold",
        reopening_patience=1,
        reopening_min_step=1,
        reopening_cooldown=10,
        reopening_max_events=1,
        reopening_fraction=0.5,
        reopening_variables=2,
        reopening_hold_steps=10,
        seed=0,
        verbose=False,
    )
    solver.optimize()

    assert solver.reopening_count == 1
    assert solver.reopening_events[0]["step"] == 1
    assert solver.reopening_events[0]["selected_coordinates"] == 4
    dual = np.asarray(solver.dual)
    assert np.count_nonzero(np.isclose(dual, 0.0)) == 4
    assert np.all(dual[:2] < -2.0)


@pytest.mark.parametrize(
    "mode",
    [
        "random_kick",
        "dual_zero_once",
        "dual_positive_hold",
        "random_branch",
        "gain_branch",
        "global_y_zero_once",
        "global_y_zero_hold",
        "global_y_zero_resetopt",
        "global_soft_recenter",
        "full_random_restart",
    ],
)
def test_reopening_modes_smoke(mode):
    data = _zero_qubo()
    solver = PDBOSolver(
        n_vars=4,
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=2,
        dual_init=-1.0,
        max_iters=3,
        check_every=1,
        reopening_mode=mode,
        reopening_patience=1,
        reopening_min_step=1,
        reopening_cooldown=10,
        reopening_max_events=1,
        reopening_fraction=0.5,
        reopening_variables=1,
        reopening_hold_steps=2,
        seed=2,
        verbose=False,
    )
    result = solver.optimize()
    assert np.isfinite(result.objective)
    assert solver.reopening_count == 1
    assert solver.reopening_events[0]["mode"] == mode


def test_global_soft_recenter_moves_all_coordinates_inside_without_one_flip_selection():
    data = _zero_qubo()
    initial = np.asarray(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    solver = PDBOSolver(
        n_vars=4,
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=2,
        primal_initial_values=initial,
        dual_init=-1.0,
        max_iters=2,
        check_every=1,
        reopening_mode="global_soft_recenter",
        reopening_patience=1,
        reopening_min_step=1,
        reopening_cooldown=10,
        reopening_max_events=1,
        reopening_hold_steps=5,
        reopening_recenter_scale=0.2,
        reopening_noise=0.0,
        seed=1,
        verbose=False,
    )
    solver.optimize()

    np.testing.assert_allclose(
        np.sort(np.unique(np.asarray(solver.primal))),
        np.asarray([0.4, 0.6]),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(solver.dual), 0.0)
    assert solver.reopening_events[0]["selected_coordinates"] == 8


def test_reopening_rejects_custom_objective():
    with pytest.raises(ValueError, match="require a quadratic objective"):
        PDBOSolver(
            n_vars=3,
            objective_type="custom",
            objective_fn=lambda x: x.sum(),
            reopening_mode="random_kick",
        )
