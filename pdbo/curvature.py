"""Spectral curvature initialization for quadratic PDBO objectives.

For ``f(x) = x.T @ Q @ x + c.T @ x``, the objective Hessian is the
constant symmetric matrix ``A = Q + Q.T``.  This module computes dual
initializations that place

    A + diag(y * g_second(x))

at, or at a prescribed distance from, the positive-semidefinite boundary.
The routines use NumPy/SciPy deliberately: initialization happens outside of
the JAX optimization loop and sparse problem data need not be densified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


ArrayLike = Union[np.ndarray, sparse.spmatrix]


@dataclass(frozen=True)
class CurvatureInitDiagnostics:
    """Numerical details for a spectral dual initialization.

    ``lambda_min`` always refers to the smallest eigenvalue of the objective
    Hessian.  ``generalized_lambda_min`` is populated only for the scalar-dual
    generalized eigenvalue construction.  In matched mode,
    ``curvature_shift`` is the common diagonal shift ``y * g_second``; in
    scalar-boundary mode it is the scalar dual value itself.
    """

    mode: str
    lambda_min: float
    generalized_lambda_min: Optional[float]
    objective_curvature: float
    objective_nonconvex: bool
    curvature_shift: float
    relative_level: Optional[float]
    target_min_eigenvalue: float
    curvature_min: float
    curvature_max: float
    eigensolver: str
    eigen_residual: float
    boundary_residual: float


@dataclass(frozen=True)
class _Eigenpair:
    value: float
    vector: np.ndarray
    method: str
    residual: float


def quadratic_hessian(
    q: ArrayLike,
    q_values: Optional[np.ndarray] = None,
    n_vars: Optional[int] = None,
    *,
    dtype: np.dtype = np.float64,
) -> sparse.csr_matrix:
    """Return the sparse Hessian ``Q + Q.T`` of a quadratic objective.

    Parameters
    ----------
    q:
        A square dense/sparse matrix when ``q_values`` is omitted.  When
        ``q_values`` is provided, this is a COO index array with shape
        ``(2, nnz)`` (the repository's native layout) or ``(nnz, 2)``.
    q_values:
        Values corresponding to COO indices in ``q``.
    n_vars:
        Matrix dimension for COO input.  Supplying it is recommended because
        an index list cannot reveal isolated trailing variables.
    dtype:
        Floating-point output dtype.  Float64 is the default for reliable
        boundary eigenvalue calculations.

    The linear coefficient ``c`` is intentionally absent: it has no effect on
    the Hessian.
    """

    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a real floating-point dtype")

    if q_values is None:
        if sparse.issparse(q):
            q_matrix = sparse.csr_matrix(q, dtype=output_dtype)
        else:
            dense_q = np.asarray(q)
            if dense_q.ndim != 2:
                raise ValueError("q must be a two-dimensional square matrix")
            if np.iscomplexobj(dense_q):
                raise TypeError("q must be real-valued")
            q_matrix = sparse.csr_matrix(dense_q, dtype=output_dtype)

        if q_matrix.ndim != 2 or q_matrix.shape[0] != q_matrix.shape[1]:
            raise ValueError("q must be a square matrix")
        if n_vars is not None and _validate_n_vars(n_vars) != q_matrix.shape[0]:
            raise ValueError("n_vars does not match q.shape")
    else:
        values = np.asarray(q_values)
        if values.ndim != 1:
            raise ValueError("q_values must be one-dimensional")
        if np.iscomplexobj(values):
            raise TypeError("q_values must be real-valued")

        indices = np.asarray(q)
        if indices.ndim != 2:
            raise ValueError("COO indices must have shape (2, nnz) or (nnz, 2)")
        if indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
        elif indices.shape[1] == 2:
            rows, cols = indices[:, 0], indices[:, 1]
        else:
            raise ValueError("COO indices must have shape (2, nnz) or (nnz, 2)")
        if rows.size != values.size:
            raise ValueError("q indices and q_values must contain the same number of entries")
        rows = _validated_indices(rows, "row")
        cols = _validated_indices(cols, "column")

        if n_vars is None:
            if values.size == 0:
                raise ValueError("n_vars is required when COO input is empty")
            dimension = int(max(rows.max(), cols.max())) + 1
        else:
            dimension = _validate_n_vars(n_vars)
        if rows.size and (rows.max() >= dimension or cols.max() >= dimension):
            raise ValueError("COO index is outside the n_vars by n_vars matrix")

        q_matrix = sparse.coo_matrix(
            (values.astype(output_dtype, copy=False), (rows, cols)),
            shape=(dimension, dimension),
            dtype=output_dtype,
        ).tocsr()

    if q_matrix.data.size and not np.isfinite(q_matrix.data).all():
        raise ValueError("q entries must all be finite")

    hessian = (q_matrix + q_matrix.T).tocsr()
    hessian.sum_duplicates()
    hessian.eliminate_zeros()
    return hessian


def smallest_eigenvalue(
    matrix: ArrayLike,
    *,
    tol: float = 1e-10,
    maxiter: Optional[int] = None,
    dense_threshold: int = 64,
) -> float:
    """Compute the algebraically smallest eigenvalue of a real symmetric matrix.

    Small matrices use LAPACK's dense symmetric eigensolver.  Larger matrices
    remain sparse and use ARPACK through :func:`scipy.sparse.linalg.eigsh`.
    """

    return _smallest_eigenpair(
        matrix,
        tol=tol,
        maxiter=maxiter,
        dense_threshold=dense_threshold,
    ).value


def scalar_boundary_from_curvature(
    hessian: ArrayLike,
    curvature: np.ndarray,
    *,
    curvature_tol: float = 1e-12,
    eig_tol: float = 1e-10,
    maxiter: Optional[int] = None,
    dense_threshold: int = 64,
) -> Tuple[float, CurvatureInitDiagnostics]:
    """Find the scalar dual value on the PSD boundary at a fixed primal point.

    With ``D = diag(curvature)`` and ``D`` strictly positive, the returned
    value is

    ``y_bar = -lambda_min(D**(-1/2) @ A @ D**(-1/2))``.

    Consequently, ``A + y_bar * D`` is positive semidefinite and singular up
    to eigensolver accuracy.  This is the generalized-eigenvalue extension of
    the MaxCut construction where ``D`` is a multiple of the identity.
    """

    matrix = _as_symmetric_csr(hessian)
    d = _validated_curvature(curvature, matrix.shape[0], curvature_tol, batch=False)

    inverse_sqrt_d = 1.0 / np.sqrt(d)
    scaling = sparse.diags(inverse_sqrt_d, format="csr")
    generalized_matrix = (scaling @ matrix @ scaling).tocsr()
    generalized_pair = _smallest_eigenpair(
        generalized_matrix,
        tol=eig_tol,
        maxiter=maxiter,
        dense_threshold=dense_threshold,
        assume_valid=True,
    )
    y_bar = -generalized_pair.value

    objective_pair = _smallest_eigenpair(
        matrix,
        tol=eig_tol,
        maxiter=maxiter,
        dense_threshold=dense_threshold,
        assume_valid=True,
    )

    # Transform the generalized eigenvector back to the original coordinates.
    critical_vector = inverse_sqrt_d * generalized_pair.vector
    critical_vector /= np.linalg.norm(critical_vector)
    av = np.asarray(matrix @ critical_vector).reshape(-1)
    dv = d * critical_vector
    boundary_action = av + y_bar * dv
    boundary_scale = max(1.0, np.linalg.norm(av) + abs(y_bar) * np.linalg.norm(dv))
    boundary_residual = float(np.linalg.norm(boundary_action) / boundary_scale)

    diagnostics = CurvatureInitDiagnostics(
        mode="scalar_boundary",
        lambda_min=objective_pair.value,
        generalized_lambda_min=generalized_pair.value,
        objective_curvature=-objective_pair.value,
        objective_nonconvex=bool(objective_pair.value < 0.0),
        curvature_shift=float(y_bar),
        relative_level=None,
        target_min_eigenvalue=0.0,
        curvature_min=float(d.min()),
        curvature_max=float(d.max()),
        eigensolver=generalized_pair.method,
        eigen_residual=generalized_pair.residual,
        boundary_residual=boundary_residual,
    )
    return float(y_bar), diagnostics


def matched_dual_from_curvature(
    hessian: ArrayLike,
    curvature: np.ndarray,
    relative_level: float = 0.0,
    *,
    trusted_objective_lambda_min: Optional[float] = None,
    curvature_tol: float = 1e-12,
    eig_tol: float = 1e-10,
    maxiter: Optional[int] = None,
    dense_threshold: int = 64,
) -> Tuple[np.ndarray, CurvatureInitDiagnostics]:
    """Match per-coordinate duals to a requested relative Hessian level.

    Let ``c = -lambda_min(A)`` and ``r = relative_level``.  For primal
    curvatures ``d`` this routine returns

    ``shift = (1 + r) * c`` and ``y = shift / d``.

    Thus every sample in a batched ``d`` satisfies

    ``A + diag(y * d) = A + shift * I``

    and its smallest eigenvalue is exactly ``r * c`` up to eigensolver
    accuracy.  ``r=0`` is the PSD boundary, ``-1 <= r < 0`` starts on the
    nonconvex side, and ``r>0`` starts inside the positive-definite side when
    the objective Hessian is nonconvex.

    ``trusted_objective_lambda_min`` can reuse a value computed for the same
    Hessian in a large parameter sweep. It is deliberately not recomputed;
    callers are responsible for binding it to the matrix. Residual diagnostics
    are ``NaN`` on this trusted-cache path.
    """

    matrix = _as_symmetric_csr(hessian)
    d = _validated_curvature(curvature, matrix.shape[0], curvature_tol, batch=True)
    r = _validated_relative_level(relative_level)
    if trusted_objective_lambda_min is None:
        pair = _smallest_eigenpair(
            matrix,
            tol=eig_tol,
            maxiter=maxiter,
            dense_threshold=dense_threshold,
            assume_valid=True,
        )
        lambda_min = pair.value
        eigensolver = pair.method
        eigen_residual = pair.residual
    else:
        supplied = np.asarray(trusted_objective_lambda_min)
        if supplied.ndim != 0 or np.iscomplexobj(supplied) or not np.isfinite(supplied):
            raise ValueError("trusted_objective_lambda_min must be a finite real scalar")
        pair = None
        lambda_min = float(supplied)
        eigensolver = "trusted_lambda_min"
        eigen_residual = float("nan")

    objective_curvature = -lambda_min
    if objective_curvature <= 0.0:
        raise ValueError(
            "matched relative curvature requires a negative objective Hessian eigenvalue"
        )
    shift = (1.0 + r) * objective_curvature
    if shift == 0.0:
        dual = np.zeros_like(d, dtype=np.float64)
    else:
        dual = np.asarray(shift / d, dtype=np.float64)
        if not np.isfinite(dual).all():
            raise FloatingPointError("matched dual overflowed; rescale the curvature data")

    target_min_eigenvalue = r * objective_curvature
    if pair is None:
        boundary_residual = float("nan")
    else:
        shifted_action = np.asarray(matrix @ pair.vector).reshape(-1) + shift * pair.vector
        target_action = target_min_eigenvalue * pair.vector
        boundary_scale = max(1.0, np.linalg.norm(shifted_action), abs(target_min_eigenvalue))
        boundary_residual = float(np.linalg.norm(shifted_action - target_action) / boundary_scale)

    diagnostics = CurvatureInitDiagnostics(
        mode="matched",
        lambda_min=lambda_min,
        generalized_lambda_min=None,
        objective_curvature=objective_curvature,
        objective_nonconvex=bool(lambda_min < 0.0),
        curvature_shift=float(shift),
        relative_level=r,
        target_min_eigenvalue=float(target_min_eigenvalue),
        curvature_min=float(d.min()),
        curvature_max=float(d.max()),
        eigensolver=eigensolver,
        eigen_residual=eigen_residual,
        boundary_residual=boundary_residual,
    )
    return dual, diagnostics


def _smallest_eigenpair(
    matrix: ArrayLike,
    *,
    tol: float,
    maxiter: Optional[int],
    dense_threshold: int,
    assume_valid: bool = False,
) -> _Eigenpair:
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    if maxiter is not None and (not isinstance(maxiter, (int, np.integer)) or maxiter <= 0):
        raise ValueError("maxiter must be a positive integer or None")
    if not isinstance(dense_threshold, (int, np.integer)) or dense_threshold < 1:
        raise ValueError("dense_threshold must be a positive integer")

    csr = sparse.csr_matrix(matrix, dtype=np.float64) if assume_valid else _as_symmetric_csr(matrix)
    n = csr.shape[0]
    if n <= dense_threshold:
        eigenvalues, eigenvectors = np.linalg.eigh(csr.toarray())
        value = float(eigenvalues[0])
        vector = np.asarray(eigenvectors[:, 0], dtype=np.float64)
        method = "dense_eigh"
    else:
        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(n)
        try:
            eigenvalues, eigenvectors = eigsh(
                csr,
                k=1,
                which="SA",
                tol=tol,
                maxiter=maxiter,
                v0=v0,
                return_eigenvectors=True,
            )
            value = float(eigenvalues[0])
            vector = np.asarray(eigenvectors[:, 0], dtype=np.float64)
            method = "sparse_eigsh"
        except ArpackNoConvergence as exc:
            if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
                # A dense fallback is useful for modest matrices and avoids
                # turning an ARPACK iteration limit into a brittle API edge.
                if n <= max(4 * dense_threshold, 256):
                    eigenvalues, eigenvectors = np.linalg.eigh(csr.toarray())
                    value = float(eigenvalues[0])
                    vector = np.asarray(eigenvectors[:, 0], dtype=np.float64)
                    method = "dense_eigh_fallback"
                else:
                    raise RuntimeError(
                        "ARPACK did not converge to a smallest eigenpair; "
                        "increase maxiter or relax tol"
                    ) from exc
            else:
                index = int(np.argmin(exc.eigenvalues))
                value = float(exc.eigenvalues[index])
                vector = np.asarray(exc.eigenvectors[:, index], dtype=np.float64)
                method = "sparse_eigsh_partial"

    vector_norm = np.linalg.norm(vector)
    if not np.isfinite(value) or not np.isfinite(vector_norm) or vector_norm == 0.0:
        raise RuntimeError("eigensolver returned a non-finite eigenpair")
    vector = vector / vector_norm
    action = np.asarray(csr @ vector).reshape(-1)
    residual_scale = max(1.0, np.linalg.norm(action), abs(value))
    residual = float(np.linalg.norm(action - value * vector) / residual_scale)
    return _Eigenpair(value=value, vector=vector, method=method, residual=residual)


def _as_symmetric_csr(matrix: ArrayLike) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        if np.iscomplexobj(matrix.data):
            raise TypeError("matrix must be real-valued")
        csr = sparse.csr_matrix(matrix, dtype=np.float64)
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if np.iscomplexobj(dense):
            raise TypeError("matrix must be real-valued")
        csr = sparse.csr_matrix(dense, dtype=np.float64)

    if csr.ndim != 2 or csr.shape[0] != csr.shape[1]:
        raise ValueError("matrix must be square")
    if csr.shape[0] == 0:
        raise ValueError("matrix must not be empty")
    if csr.data.size and not np.isfinite(csr.data).all():
        raise ValueError("matrix entries must all be finite")

    asymmetry = (csr - csr.T).tocsr()
    asymmetry.eliminate_zeros()
    if asymmetry.nnz:
        max_asymmetry = float(np.max(np.abs(asymmetry.data)))
        matrix_scale = max(1.0, float(np.max(np.abs(csr.data))) if csr.data.size else 0.0)
        if max_asymmetry > 1e-10 * matrix_scale:
            raise ValueError("matrix must be symmetric")

    # Remove harmless floating-point skew before passing the matrix to eigsh.
    symmetric = ((csr + csr.T) * 0.5).tocsr()
    symmetric.sum_duplicates()
    symmetric.eliminate_zeros()
    return symmetric


def _validated_curvature(
    curvature: np.ndarray,
    n_vars: int,
    curvature_tol: float,
    *,
    batch: bool,
) -> np.ndarray:
    if not np.isfinite(curvature_tol) or curvature_tol < 0.0:
        raise ValueError("curvature_tol must be finite and nonnegative")
    d = np.asarray(curvature, dtype=np.float64)
    if (batch and d.ndim < 1) or (not batch and d.ndim != 1):
        expected = "(..., n_vars)" if batch else "(n_vars,)"
        raise ValueError(f"curvature must have shape {expected}")
    if d.shape[-1] != n_vars:
        raise ValueError("the last curvature dimension must match the Hessian size")
    if d.size == 0:
        raise ValueError("curvature must not be empty")
    if not np.isfinite(d).all():
        raise ValueError("curvature entries must all be finite")
    minimum = float(d.min())
    if minimum <= curvature_tol:
        raise ValueError(
            f"curvature entries must all be greater than curvature_tol={curvature_tol:g}; "
            f"minimum was {minimum:g}"
        )
    return d


def _validated_relative_level(relative_level: float) -> float:
    value = np.asarray(relative_level)
    if value.ndim != 0 or np.iscomplexobj(value):
        raise ValueError("relative_level must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result < -1.0:
        raise ValueError("relative_level must be finite and at least -1")
    return result


def _validate_n_vars(n_vars: int) -> int:
    if not isinstance(n_vars, (int, np.integer)) or isinstance(n_vars, (bool, np.bool_)):
        raise TypeError("n_vars must be a positive integer")
    result = int(n_vars)
    if result <= 0:
        raise ValueError("n_vars must be a positive integer")
    return result


def _validated_indices(indices: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(indices) or not np.issubdtype(indices.dtype, np.number):
        raise TypeError(f"COO {name} indices must be integers")
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all():
        raise ValueError(f"COO {name} indices must be finite integers")
    result = indices.astype(np.int64, copy=False)
    if result.size and result.min() < 0:
        raise ValueError(f"COO {name} indices must be nonnegative")
    return result


__all__ = [
    "CurvatureInitDiagnostics",
    "matched_dual_from_curvature",
    "quadratic_hessian",
    "scalar_boundary_from_curvature",
    "smallest_eigenvalue",
]
