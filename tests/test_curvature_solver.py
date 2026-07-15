import jax
import numpy as np
import pytest

from pdbo import PDBOSolver, generate_max_cut, random_graph
from pdbo.curvature import quadratic_hessian


def _initial_curvature(solver):
    g_second = jax.grad(jax.grad(solver._make_g()))
    return np.asarray(jax.vmap(jax.vmap(g_second))(solver.primal), dtype=np.float64)


@pytest.mark.parametrize("g_type", ["quad", "entropy", "cosh"])
@pytest.mark.parametrize("level", [0.0, -0.2, 0.1])
def test_solver_curvature_init_hits_requested_hessian_level(g_type, level):
    data = generate_max_cut(random_graph(n=10, d=3, seed=0))
    solver = PDBOSolver(
        n_vars=data["num_vars"],
        objective_type="quadratic",
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=3,
        max_iters=1,
        g_type=g_type,
        dual_init_mode="curvature",
        hessian_init_level=level,
        seed=7,
        verbose=False,
    )

    objective_hessian = quadratic_hessian(
        data["Q_indices"], data["Q_values"], data["num_vars"]
    ).toarray()
    curvature = _initial_curvature(solver)
    dual = np.asarray(solver.dual)
    diagnostics = solver.initial_curvature_diagnostics

    np.testing.assert_allclose(
        dual * curvature,
        diagnostics.curvature_shift,
        rtol=2e-5,
        atol=2e-5,
    )
    for sample_dual, sample_curvature in zip(dual, curvature):
        hessian = objective_hessian + np.diag(sample_dual * sample_curvature)
        assert np.linalg.eigvalsh(hessian)[0] == pytest.approx(
            diagnostics.target_min_eigenvalue,
            abs=3e-5,
        )


def test_solver_rejects_zero_constraint_curvature_at_boundary():
    data = generate_max_cut(random_graph(n=10, d=3, seed=0))

    with pytest.raises(ValueError, match="curvature entries"):
        PDBOSolver(
            n_vars=data["num_vars"],
            objective_type="quadratic",
            Q_indices=data["Q_indices"],
            Q_values=data["Q_values"],
            c=data["c"],
            batch_size=1,
            max_iters=1,
            primal_init="half",
            g_type="poly4",
            dual_init_mode="curvature",
            hessian_init_level=0.0,
            verbose=False,
        )


def test_new_initialization_options_preserve_legacy_positional_arguments():
    data = generate_max_cut(random_graph(n=10, d=3, seed=0))

    solver = PDBOSolver(
        data["num_vars"],
        "quadratic",
        None,
        data["Q_indices"],
        data["Q_values"],
        data["c"],
        "rmsprop",
        1,
        0.001,
        1e-8,
        0.001,
        4.0,
        3,
        None,
        0,
        False,
    )

    assert solver.max_iters == 3
    assert solver.dual_init_mode == "constant"
