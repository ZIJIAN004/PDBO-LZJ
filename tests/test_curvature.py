import numpy as np
import pytest
from scipy import sparse

from pdbo.curvature import (
    matched_dual_from_curvature,
    quadratic_hessian,
    scalar_boundary_from_curvature,
    smallest_eigenvalue,
)


def test_quadratic_hessian_from_repository_coo_layout():
    indices = np.array([[0, 0, 1], [0, 1, 2]])
    values = np.array([3.0, -2.0, 4.0])

    hessian = quadratic_hessian(indices, values, n_vars=4)

    expected_q = np.zeros((4, 4))
    expected_q[indices[0], indices[1]] = values
    np.testing.assert_allclose(hessian.toarray(), expected_q + expected_q.T)
    assert sparse.isspmatrix_csr(hessian)


def test_smallest_eigenvalue_supports_dense_and_sparse_paths():
    diagonal = np.linspace(-3.5, 8.0, 100)
    matrix = sparse.diags(diagonal, format="csr")

    assert smallest_eigenvalue(matrix, dense_threshold=16) == pytest.approx(-3.5)
    assert smallest_eigenvalue(matrix[:3, :3], dense_threshold=16) == pytest.approx(-3.5)


def test_scalar_generalized_eigenvalue_is_psd_boundary():
    hessian = sparse.csr_matrix([[-3.0, 1.0], [1.0, 2.0]])
    curvature = np.array([0.5, 4.0])

    y_bar, diagnostics = scalar_boundary_from_curvature(hessian, curvature)
    boundary_hessian = hessian.toarray() + y_bar * np.diag(curvature)
    boundary_eigenvalues = np.linalg.eigvalsh(boundary_hessian)

    assert boundary_eigenvalues[0] == pytest.approx(0.0, abs=1e-11)
    assert np.all(boundary_eigenvalues >= -1e-11)
    assert y_bar > 0.0
    assert diagnostics.mode == "scalar_boundary"
    assert diagnostics.objective_nonconvex
    assert diagnostics.generalized_lambda_min == pytest.approx(-y_bar)
    assert diagnostics.boundary_residual < 1e-10


def test_matched_batch_hits_requested_nonconvex_level():
    hessian = np.array([[-4.0, 1.0], [1.0, 2.0]])
    curvature = np.array([[0.5, 2.0], [5.0, 0.25], [1.0, 8.0]])
    relative_level = -0.25

    dual, diagnostics = matched_dual_from_curvature(
        hessian,
        curvature,
        relative_level=relative_level,
    )

    expected_shift = diagnostics.curvature_shift
    np.testing.assert_allclose(dual * curvature, expected_shift)
    for sample_dual, sample_curvature in zip(dual, curvature):
        total_hessian = hessian + np.diag(sample_dual * sample_curvature)
        assert np.linalg.eigvalsh(total_hessian)[0] == pytest.approx(
            diagnostics.target_min_eigenvalue,
            abs=1e-11,
        )
    assert diagnostics.target_min_eigenvalue < 0.0
    assert diagnostics.target_min_eigenvalue == pytest.approx(
        relative_level * diagnostics.objective_curvature
    )


def test_matched_boundary_is_invariant_to_curvature_scaling():
    hessian = np.array([[-2.0, 0.5], [0.5, 1.0]])
    curvature = np.array([1.0, 3.0])

    dual, diagnostics = matched_dual_from_curvature(hessian, curvature)
    scaled_dual, scaled_diagnostics = matched_dual_from_curvature(hessian, 7.0 * curvature)

    np.testing.assert_allclose(scaled_dual, dual / 7.0)
    np.testing.assert_allclose(scaled_dual * (7.0 * curvature), dual * curvature)
    assert scaled_diagnostics.curvature_shift == pytest.approx(diagnostics.curvature_shift)
    total_hessian = hessian + np.diag(dual * curvature)
    assert np.linalg.eigvalsh(total_hessian)[0] == pytest.approx(0.0, abs=1e-11)


def test_matched_dual_accepts_cached_objective_eigenvalue():
    hessian = np.diag([-3.0, 2.0])
    curvature = np.array([[1.0, 4.0], [2.0, 8.0]])

    dual, diagnostics = matched_dual_from_curvature(
        hessian,
        curvature,
        relative_level=-0.25,
        trusted_objective_lambda_min=-3.0,
    )

    np.testing.assert_allclose(dual * curvature, 2.25)
    assert diagnostics.target_min_eigenvalue == pytest.approx(-0.75)
    assert diagnostics.eigensolver == "trusted_lambda_min"
    assert np.isnan(diagnostics.eigen_residual)
    assert np.isnan(diagnostics.boundary_residual)


def test_matched_dual_rejects_psd_objective_hessian():
    with pytest.raises(ValueError, match="negative objective Hessian eigenvalue"):
        matched_dual_from_curvature(np.diag([1.0, 2.0]), np.ones(2))


def test_zero_shift_returns_exact_zero_dual():
    hessian = sparse.diags([-3.0, 1.0], format="csr")

    dual, diagnostics = matched_dual_from_curvature(
        hessian,
        np.array([[1.0, 2.0], [4.0, 8.0]]),
        relative_level=-1.0,
    )

    np.testing.assert_array_equal(dual, np.zeros_like(dual))
    assert diagnostics.curvature_shift == 0.0
    assert diagnostics.target_min_eigenvalue == pytest.approx(-3.0)


@pytest.mark.parametrize(
    "curvature",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, 1e-14]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_matched_dual_rejects_zero_or_nonfinite_curvature(curvature):
    with pytest.raises(ValueError, match="curvature"):
        matched_dual_from_curvature(np.diag([-1.0, 2.0]), curvature)


def test_matched_dual_rejects_relative_level_below_minus_one():
    with pytest.raises(ValueError, match="relative_level"):
        matched_dual_from_curvature(
            np.diag([-1.0, 2.0]),
            np.ones(2),
            relative_level=-1.0001,
        )
