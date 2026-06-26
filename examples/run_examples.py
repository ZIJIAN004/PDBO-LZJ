"""Small examples for the public PDBO package."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdbo import (
    LABSPuboSolver,
    MaxSatSolver,
    PDBOSolver,
    evaluate_labs_bits,
    generate_max_cut,
    generate_max_sat,
    generate_mis,
    parse_gset,
    random_graph,
)


def run_quadratic_mis():
    graph = random_graph(n=20, d=3, seed=0)
    data = generate_mis(graph, penalty=4)
    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=4,
        max_iters=20,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    print(f"MIS objective={result.objective:.3f}")


def run_gset_maxcut():
    graph = parse_gset(1)
    data = generate_max_cut(graph)
    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=8,
        max_iters=20,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    print(f"Gset G1 Max-Cut QUBO objective={result.objective:.3f}")


def run_maxsat():
    data = generate_max_sat("instance/MAXSAT/3CNF/example.cnf")
    solver = MaxSatSolver(
        n_vars=data["num_vars"],
        CNF=data["CNF"],
        batch_size=4,
        max_iters=5,
        verbose=False,
        seed=0,
    )
    solver.optimize()
    print(f"MAXSAT unsatisfied={float(solver.objVal):.0f}")


def run_labs_pubo():
    solver = LABSPuboSolver(
        n_vars=12,
        batch_size=32,
        max_iters=100,
        primal_lr=0.03,
        dual_lr=0.03,
        dual_init=100,
        verbose=False,
        seed=0,
    )
    solver.optimize()
    energy = evaluate_labs_bits(solver.incumbent)
    print(f"LABS PUBO energy={energy}")


def run_custom_objective():
    def extension(x):
        return jnp.sum((x[:-1] - x[1:]) ** 2) + 0.1 * jnp.sum(x)

    solver = PDBOSolver(
        n_vars=16,
        objective_type="custom",
        objective_fn=extension,
        batch_size=4,
        max_iters=20,
        check_every=5,
        verbose=False,
        seed=0,
    )
    result = solver.optimize()
    print(f"Custom objective={result.objective:.3f}")


if __name__ == "__main__":
    run_quadratic_mis()
    run_gset_maxcut()
    run_maxsat()
    run_labs_pubo()
    run_custom_objective()
