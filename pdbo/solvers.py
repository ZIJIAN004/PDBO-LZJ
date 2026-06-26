"""Solver classes exposed by the PDBO package."""

from solver_jax import LABS_PUBO_JAX, MAXSAT_JAX, MAX_K_CUT_JAX, PDBO_JAX, PDBOResult, PDQUBO_JAX

PDBOSolver = PDBO_JAX
PDQuboSolver = PDQUBO_JAX
LABSPuboSolver = LABS_PUBO_JAX
MaxKCutSolver = MAX_K_CUT_JAX
MaxSatSolver = MAXSAT_JAX

__all__ = ["PDBOSolver", "PDBOResult", "PDQuboSolver", "LABSPuboSolver", "MaxKCutSolver", "MaxSatSolver"]
