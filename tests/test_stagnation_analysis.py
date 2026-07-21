import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_stagnation.py"
SPEC = importlib.util.spec_from_file_location("analyze_stagnation", SCRIPT_PATH)
stagnation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stagnation
SPEC.loader.exec_module(stagnation)


def test_projected_min_curvature_uses_only_free_coordinates():
    objective_hessian = sparse.diags([-4.0, -2.0, 3.0], format="csr")
    primal = np.array([0.01, 0.5, 0.6])
    dual = np.array([0.0, 0.25, 0.0])

    result = stagnation.projected_min_curvature(
        objective_hessian,
        primal,
        dual,
        free_margin=0.05,
        eig_tol=1e-10,
        eig_maxiter=100,
        seed=0,
    )

    assert result.free_variables == 2
    assert result.lambda_min == pytest.approx(-1.5)
    assert result.residual < 1e-10


def test_projected_min_curvature_reports_binary_boundary():
    result = stagnation.projected_min_curvature(
        sparse.eye(3, format="csr"),
        np.array([0.0, 1.0, 0.01]),
        np.zeros(3),
        free_margin=0.05,
        eig_tol=1e-10,
        eig_maxiter=100,
        seed=0,
    )

    assert result.free_variables == 0
    assert np.isnan(result.lambda_min)
    assert result.status == "no_free_variables"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"fractionality": 1e-4}, "near_binary_discrete_basin"),
        ({"movement_rms_per_step": 2e-4}, "continuous_state_still_moving"),
        ({"gradient_rms": 2e-2}, "optimizer_or_projection_limited"),
        ({"lambda_min": -0.5}, "negative_curvature_stall"),
        ({"lambda_min": 5e-4}, "flat_fractional_stall"),
        ({"lambda_min": 0.5}, "positive_curvature_fractional_basin"),
    ],
)
def test_stagnation_classification(overrides, expected):
    values = {
        "fractionality": 0.2,
        "free_fraction": 0.8,
        "movement_rms_per_step": 1e-7,
        "gradient_rms": 1e-5,
        "lambda_min": 0.5,
        "binary_tol": 1e-3,
        "free_fraction_tol": 0.01,
        "movement_tol": 1e-5,
        "gradient_tol": 1e-3,
        "curvature_tol": 1e-3,
    }
    values.update(overrides)
    assert stagnation.classify_stagnation(**values) == expected
