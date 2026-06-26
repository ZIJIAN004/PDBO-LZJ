"""Local-search refinements for PDBO incumbents."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefinementResult:
    bits: np.ndarray
    objective: float
    steps: int
    seconds: float


def qubo_objective(bits, data):
    x = np.asarray(bits, dtype=np.float32)
    qx = data["Q_sparse"].dot(x)
    return float(x.dot(qx) + np.dot(data["c"], x) + data.get("objective_offset", 0.0))


def maxsat_unsatisfied(bits, cnf):
    x = np.asarray(bits, dtype=np.int8)
    unsatisfied = 0
    for clause in cnf:
        satisfied = False
        for literal in clause:
            var = abs(int(literal)) - 1
            value = bool(x[var])
            if (literal > 0 and value) or (literal < 0 and not value):
                satisfied = True
                break
        if not satisfied:
            unsatisfied += 1
    return float(unsatisfied)


def one_flip_search(bits, objective_fn, max_passes=None):
    """Greedy one-flip local search for minimization objectives."""
    import time

    start = time.perf_counter()
    current = np.asarray(bits, dtype=np.int8).copy()
    best_obj = float(objective_fn(current))
    steps = 0
    passes = 0

    while max_passes is None or passes < max_passes:
        improved = False
        passes += 1
        for idx in range(current.size):
            current[idx] = 1 - current[idx]
            candidate_obj = float(objective_fn(current))
            if candidate_obj < best_obj:
                best_obj = candidate_obj
                steps += 1
                improved = True
            else:
                current[idx] = 1 - current[idx]
        if not improved:
            break

    return RefinementResult(
        bits=current.astype(np.int32),
        objective=best_obj,
        steps=steps,
        seconds=time.perf_counter() - start,
    )


def refine_binary_incumbent(task, bits, data, max_passes=None):
    if task == "labs":
        from .problems import evaluate_labs_bits

        n = data["num_x_vars"]
        initial = np.asarray(bits[:n], dtype=np.int8)
        return one_flip_search(initial, evaluate_labs_bits, max_passes=max_passes)

    if task == "maxsat":
        initial = np.asarray(bits, dtype=np.int8)
        return one_flip_search(
            initial,
            lambda candidate: maxsat_unsatisfied(candidate, data["CNF"]),
            max_passes=max_passes,
        )

    initial = np.asarray(bits, dtype=np.int8)
    return one_flip_search(
        initial,
        lambda candidate: qubo_objective(candidate, data),
        max_passes=max_passes,
    )
