# PDBO

PDBO is a JAX implementation of primal-dual optimization for binary optimization,
released with the paper **Smoothing Binary Optimization: A Primal-Dual Perspective**.


## Installation

```bash
pip install -e .
```

For GPU runs, install the JAX build that matches your CUDA version before installing
this package. See the official JAX installation guide for platform-specific wheels.

To run smoke tests:

```bash
pip install -e ".[test]"
pytest
```

## Quick Start

Run all bundled small examples:

```bash
python examples/run_examples.py
```

Run PDBO on a random regular MIS instance:

```bash
python main.py --task mis --graph reg --n 1000 --d 3 --batch 10
```

Run PDBO on a Gset Max-Cut instance:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --batch 100 --dual_init 6
```

Run PDBO on a LABS instance:

```bash
python main.py --task labs --labs_n 47 --labs_penalty 10000 --batch 100
```

The default iteration budget is `max_iters=5000`.

## Python API

For quadratic objectives, PDBO solves

```text
minimize x^T Q x + c^T x,  x in {0, 1}^n.
```

```python
from pdbo import PDBOSolver, generate_mis, random_graph

graph = random_graph(n=1000, d=3, seed=0)
data = generate_mis(graph, penalty=4)

solver = PDBOSolver(
    n_vars=data["num_vars"],
    objective_type="quadratic",
    Q_indices=data["Q_indices"],
    Q_values=data["Q_values"],
    c=data["c"],
    batch_size=10,
    primal_lr=0.02,
    dual_lr=0.02,
    dual_init=5,
    rounding_samples=8,
    seed=0,
)
result = solver.optimize()
print(solver.ObjVal, solver.X)
print(result.objective, result.incumbent)
```

For custom objectives, provide a differentiable JAX function defining a continuous
extension on `[0, 1]^n`:

```python
import jax.numpy as jnp
from pdbo import PDBOSolver

def extension(x):
    return jnp.sum((x[:-1] - x[1:]) ** 2) + 0.1 * jnp.sum(x)

solver = PDBOSolver(
    n_vars=100,
    objective_type="custom",
    objective_fn=extension,
    batch_size=32,
)
result = solver.optimize()
```

`PDQuboSolver` remains available as a compatibility wrapper for the quadratic path.

## Options

Initialize every parallel trajectory at the local PSD boundary of the
Lagrangian Hessian (quadratic objectives and positive finite `g''(x0)` only):

```bash
python main.py --task mc --graph Gset --Gset_id 1 \
    --dual_init_mode curvature --hessian_init_level 0
```

Use a negative relative level to start with controlled nonconvexity. For
example, `-0.1` sets the initial smallest Hessian eigenvalue to 10% of the
unpenalized negative curvature:

```bash
python main.py --task mc --graph Gset --Gset_id 1 \
    --dual_init_mode curvature --hessian_init_level=-0.1
```

The solver uses coordinatewise dual values so different random starts and
different constraint functions have the same initial Hessian curvature. See
[`docs/hessian_initialization.md`](docs/hessian_initialization.md) for the
derivation and the cases where no finite boundary initialization exists.

Early stopping based on rounded incumbent stagnation:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --patience 1000 --check_every 10
```

Extra rounded candidates sampled from the final relaxed batch:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --rounding_samples 8
```

Greedy one-flip local-search refinement:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --refine
```

Perturbation is disabled by default. It can help under short budgets or highly
fractional stagnation, but may slightly hurt some long-budget runs:

```bash
python main.py --task mc --graph Gset --Gset_id 1 --perturbation --perturbation_fraction 0.05
```

For quadratic objectives, `--quadratic_backend sparse` is the default. Use
`--quadratic_backend edge` to evaluate the QUBO directly from edge indices.

## Citation

If you use this code, please cite:

```bibtex
@misc{liu2026smoothingbinaryoptimizationprimaldual,
      title={Smoothing Binary Optimization: A Primal-Dual Perspective}, 
      author={Wenbo Liu and Akang Wang and Dun Ma and Hongyi Jiang and Jianghua Wu and Wenguo Yang},
      year={2026},
      eprint={2509.21064},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2509.21064}, 
}
```

## Repository Layout

- `pdbo/`: public package API.
- `solver_jax.py`: PDBO solver implementations.
- `problem_parser.py`: problem parsers and QUBO builders.
- `main.py`: command-line entry point.
- `examples/`: small runnable examples for each supported problem type.
- `instance/`: minimal bundled example instances (`Gset/G1.txt` and one small 3-SAT CNF).
- `scripts/pdqubo/`: optional PDBO-only run scripts.

Baseline solvers, large benchmark collections, and paper-reproduction artifacts
are intentionally not included in this clean public package.
