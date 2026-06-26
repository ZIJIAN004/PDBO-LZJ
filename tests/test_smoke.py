import subprocess
import sys

import jax.numpy as jnp

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
