import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from pdbo import PDBOSolver, generate_mis, random_graph


def test_quadratic_solver_smoke():
    graph = random_graph(n=12, d=3, seed=0)
    data = generate_mis(graph, penalty=4)
    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=2,
        max_iters=5,
        check_every=5,
        rounding_samples=2,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    assert result.incumbent.shape == (data["num_vars"],)
    assert result.runtime >= 0.0


def test_custom_solver_smoke():
    def extension(x):
        return jnp.sum((x[:-1] - x[1:]) ** 2) + 0.1 * jnp.sum(x)

    solver = PDBOSolver(
        n_vars=8,
        objective_type="custom",
        objective_fn=extension,
        batch_size=2,
        max_iters=5,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    assert result.incumbent.shape == (8,)


def test_state_callback_runs_at_requested_interval_and_final_step():
    graph = random_graph(n=8, d=2, seed=3)
    data = generate_mis(graph, penalty=4)
    calls = []

    def record_state(step, primal, dual, gradient, objective, incumbent, last_improve_step):
        calls.append(
            (
                step,
                np.asarray(primal).shape,
                np.asarray(dual).shape,
                np.asarray(gradient).shape,
                float(objective),
                np.asarray(incumbent).shape,
                last_improve_step,
            )
        )

    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=3,
        max_iters=5,
        check_every=5,
        state_callback=record_state,
        state_callback_every=2,
        verbose=False,
        seed=0,
    )
    solver.optimize()

    assert [call[0] for call in calls] == [1, 3, 4]
    assert all(call[1] == (3, data["num_vars"]) for call in calls)
    assert all(call[2] == call[1] and call[3] == call[1] for call in calls)
    assert all(call[5] == (data["num_vars"],) for call in calls)
    assert all(isinstance(call[6], int) for call in calls)


def test_state_callback_interval_must_be_positive():
    with pytest.raises(ValueError, match="state_callback_every must be positive"):
        PDBOSolver(n_vars=2, state_callback_every=0)


def test_state_callback_does_not_change_solver_trajectory():
    graph = random_graph(n=10, d=2, seed=5)
    data = generate_mis(graph, penalty=4)
    common = dict(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        optimizer_type="rmsprop",
        batch_size=3,
        max_iters=12,
        check_every=3,
        verbose=False,
        seed=7,
    )
    baseline = PDBOSolver(**common)
    baseline_result = baseline.optimize()

    calls = []

    def observe(step, primal, dual, gradient, objective, incumbent, last_improve_step):
        del primal, dual, gradient, objective, incumbent, last_improve_step
        calls.append(step)

    observed = PDBOSolver(
        **common,
        state_callback=observe,
        state_callback_every=4,
    )
    observed_result = observed.optimize()

    assert calls == [3, 7, 11]
    assert observed_result.objective == baseline_result.objective
    np.testing.assert_array_equal(observed_result.incumbent, baseline_result.incumbent)
    np.testing.assert_array_equal(np.asarray(observed.primal), np.asarray(baseline.primal))
    np.testing.assert_array_equal(np.asarray(observed.dual), np.asarray(baseline.dual))


def test_cli_smoke():
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--task",
            "mis",
            "--graph",
            "reg",
            "--n",
            "12",
            "--d",
            "3",
            "--batch",
            "2",
            "--max_iters",
            "5",
            "--check_every",
            "5",
            "--no-verbose",
            "--refine",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "best=" in proc.stdout
    assert "refined_best=" in proc.stdout
