import math
import time
from dataclasses import dataclass
from jax.experimental import sparse
import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Optional, Callable
import numpy as np


# Constraint functions g(x) selectable via g_type.
#   G_TYPES_CONVEX  : satisfy the paper's C1 convexity conditions, but need not be
#                     C2 or strongly convex. Some therefore cannot convexify a
#                     coupled objective at every point.
#   G_TYPES_PARTIAL : convex only near x=1/2, concave near the edges (controls that
#                     do not satisfy the paper's global convexity condition).
#   G_TYPES_LINEAR  : a convex V-shape that is nondifferentiable at x=1/2 and has
#                     zero curvature almost everywhere (exact-penalty control).
G_TYPES_CONVEX = (
    "quad", "poly4", "poly6", "poly8", "abs_pow3", "abs_pow1p5",
    "entropy", "sin", "semicircle", "cosh", "huber",
)
G_TYPES_PARTIAL = ("quartic_well", "sin2")
G_TYPES_LINEAR = ("vshape",)
G_TYPES = G_TYPES_CONVEX + G_TYPES_PARTIAL + G_TYPES_LINEAR


@dataclass(frozen=True)
class PDBOResult:
    objective: float
    incumbent: np.ndarray
    objective_history: list
    timing_history: list
    runtime: float
    stop_reason: Optional[str]


class LABS_PUBO_JAX:
    def __init__(
            self,
            n_vars: int,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            timelimit: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            primal_init: str = 'uniform',
            step_callback: Optional[Callable] = None,
    ):
        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"
        assert primal_init in {'uniform', 'half', 'binary'}, "Invalid primal init"

        self.key = jax.random.PRNGKey(seed)
        self.n = n_vars
        self.batch_size = batch_size
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.timelimit = timelimit
        self.verbose = verbose
        self.step_callback = step_callback

        self.primal, self.dual = self._init_variables(dual_init, primal_init)
        self.incumbent = jnp.full(self.n, 0, dtype=jnp.float32)
        self.objVal = self._labs_energy(self.incumbent)
        self.objVal_record = [self.objVal.item()]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None

        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t, step):
        print(self.objVal, f'time:{t}')

    def _init_variables(self, dual_init: float, primal_init: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key, subkey = jax.random.split(self.key)
        shape = (self.batch_size, self.n)
        if primal_init == 'half':
            primal = jnp.full(shape, 0.5, dtype=jnp.float32)
        elif primal_init == 'binary':
            primal = jax.random.bernoulli(subkey, 0.5, shape).astype(jnp.float32)
        else:
            primal = jax.random.uniform(subkey, shape)
        dual = jnp.full(shape, dual_init, dtype=jnp.float32)
        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:
        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)

    def _labs_energy(self, x):
        spins = 2 * x - 1
        energy = 0.0
        for k in range(1, self.n):
            autocorrelation = jnp.dot(spins[:self.n - k], spins[k:])
            energy = energy + autocorrelation ** 2
        return energy

    def optimize(self):
        self.start_time = time.perf_counter()
        batch_obj = jax.vmap(self._labs_energy, in_axes=0)

        def base_term(x):
            return batch_obj(x).sum()

        def penalty_term(x, y):
            return (y * (x ** 2 - x)).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(x) + penalty_term(x, y), argnums=0)

        @jax.jit
        def primal_dual_update(x, y, opt_state_x, opt_state_y, objVal, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)
            x = jnp.clip(x, 0.0, 1.0)
            y += self.dual_lr * (x ** 2 - x)

            batch_int_x = jax.lax.stop_gradient(jnp.round(x))
            int_x = jnp.concatenate((batch_int_x, incumbent[jnp.newaxis, :]), axis=0)
            objs = batch_obj(int_x)
            idx = jnp.argmin(objs)
            objVal = objs[idx]
            incumbent = int_x[idx]
            return x, y, opt_state_x, opt_state_y, objVal, incumbent

        for _step in range(self.max_iters):
            t = time.perf_counter()
            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal,
             self.incumbent) = primal_dual_update(
                self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal, self.incumbent)

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()
            self.objVal.block_until_ready()

            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            step_time = time.perf_counter() - t
            self.log(step_time, _step)
            if self.step_callback is not None:
                self.step_callback(_step, step_time, self.objVal, self.incumbent)
            if self.timelimit is not None and time.perf_counter() - self.start_time >= self.timelimit:
                break

        self.solving_time = time.perf_counter() - self.start_time


class PDBO_JAX:
    def __init__(
            self,
            n_vars: int,
            objective_type: str = 'custom',
            objective_fn: Optional[Callable] = None,
            Q_indices: Optional[np.ndarray] = None,
            Q_values: Optional[np.ndarray] = None,
            c: Optional[np.ndarray] = None,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            tolerance: float = 1e-8,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            timelimit: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            primal_init: str = 'uniform',
            step_callback: Optional[Callable] = None,
            state_callback: Optional[Callable] = None,
            state_callback_every: int = 100,
            incumbent_score_fn: Optional[Callable] = None,
            patience: Optional[int] = None,
            min_delta: float = 0.0,
            check_every: int = 1,
            quadratic_backend: str = 'sparse',
            g_type: str = 'quad',
            g_normalize: bool = False,
            rounding_samples: int = 0,
            perturbation: bool = False,
            perturbation_strength: float = 0.4,
            perturbation_fraction: float = 0.05,
            perturbation_patience: int = 200,
            perturbation_integrality_tol: float = 0.2,
            perturbation_reset_optimizer: bool = True,
            dual_init_mode: str = 'constant',
            hessian_init_level: float = 0.0,
            curvature_tol: float = 1e-12,
            eig_tol: float = 1e-8,
            trusted_objective_hessian_lambda_min: Optional[float] = None,
            primal_initial_values: Optional[np.ndarray] = None,
    ):
        assert objective_type in {'quadratic', 'custom'}, "Invalid objective type"
        assert optimizer_type in {'sgd', 'rmsprop', 'adam'}, "Invalid optimizer type"
        assert primal_init in {'uniform', 'half', 'binary'}, "Invalid primal init"
        assert quadratic_backend in {'edge', 'sparse'}, "Invalid quadratic backend"
        assert g_type in G_TYPES, f"Invalid g_type: {g_type}"
        if dual_init_mode not in {'constant', 'curvature'}:
            raise ValueError("dual_init_mode must be 'constant' or 'curvature'")
        if hessian_init_level < -1.0:
            raise ValueError("hessian_init_level must be at least -1")
        if curvature_tol < 0.0:
            raise ValueError("curvature_tol must be non-negative")
        if eig_tol <= 0.0:
            raise ValueError("eig_tol must be positive")
        if (
                trusted_objective_hessian_lambda_min is not None
                and not np.isfinite(trusted_objective_hessian_lambda_min)
        ):
            raise ValueError("trusted_objective_hessian_lambda_min must be finite or None")
        if dual_init_mode == 'curvature' and objective_type != 'quadratic':
            raise ValueError("curvature-matched dual initialization currently requires a quadratic objective")
        if patience is not None and patience < 1:
            raise ValueError("patience must be positive or None")
        if check_every < 1:
            raise ValueError("check_every must be positive")
        if state_callback_every < 1:
            raise ValueError("state_callback_every must be positive")
        if rounding_samples < 0:
            raise ValueError("rounding_samples must be non-negative")
        if perturbation_fraction < 0.0 or perturbation_fraction > 1.0:
            raise ValueError("perturbation_fraction must be in [0, 1]")
        if perturbation_patience < 1:
            raise ValueError("perturbation_patience must be positive")

        self.key = jax.random.PRNGKey(seed)
        self.n = n_vars
        self.objective_type = objective_type
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.dual_init_mode = dual_init_mode
        self.hessian_init_level = hessian_init_level
        self.curvature_tol = curvature_tol
        self.eig_tol = eig_tol
        self.trusted_objective_hessian_lambda_min = trusted_objective_hessian_lambda_min
        self.max_iters = max_iters
        self.timelimit = timelimit
        self.verbose = verbose
        self.step_callback = step_callback
        self.state_callback = state_callback
        self.state_callback_every = state_callback_every
        self.incumbent_score_fn = incumbent_score_fn
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.quadratic_backend = quadratic_backend
        self.g_type = g_type
        self.g_normalize = g_normalize
        self.rounding_samples = rounding_samples
        self.perturbation = perturbation
        self.perturbation_strength = perturbation_strength
        self.perturbation_fraction = perturbation_fraction
        self.perturbation_patience = perturbation_patience
        self.perturbation_integrality_tol = perturbation_integrality_tol
        self.perturbation_reset_optimizer = perturbation_reset_optimizer
        self.stop_reason = None
        self.perturbation_count = 0

        if objective_type == 'quadratic':
            if Q_indices is None or Q_values is None:
                raise ValueError("Q_indices and Q_values are required for objective_type='quadratic'")
            if c is None:
                c = np.zeros(n_vars, dtype=np.float32)
            self.m = Q_indices.shape[1]
            self.Q_indices = jnp.asarray(Q_indices)
            self.Q_values = jnp.asarray(Q_values)
            self.c = jnp.asarray(c)
            if quadratic_backend == 'sparse':
                self.Q = sparse.BCOO((self.Q_values, jnp.column_stack(self.Q_indices)), shape=(n_vars, n_vars))

                def quadratic_objective(x):
                    return (((x @ self.Q) + self.c) * x).sum()
            else:
                self.Q = None

                def quadratic_objective(x):
                    return (self.Q_values * x[self.Q_indices[0]] * x[self.Q_indices[1]]).sum() + jnp.dot(self.c, x)

            self.objective_fn = quadratic_objective
        else:
            if objective_fn is None:
                raise ValueError("objective_fn is required for objective_type='custom'")
            self.m = None
            self.Q_indices = None
            self.Q_values = None
            self.Q = None
            self.c = None
            self.objective_fn = objective_fn

        self.primal, self.dual = self._init_variables(dual_init, primal_init)
        if primal_initial_values is not None:
            self.primal = self._validate_primal_initial_values(primal_initial_values)
        self.initial_curvature_diagnostics = None
        self.initial_g_curvature_min = None
        self.initial_g_curvature_max = None
        if self.dual_init_mode == 'curvature':
            self._initialize_curvature_matched_dual()
        self.initial_dual_min = float(self.dual.min())
        self.initial_dual_max = float(self.dual.max())
        self.initial_dual_mean = float(self.dual.mean())
        self.incumbent = jnp.full(self.n, 0, dtype=jnp.float32)
        self.objVal = self._score_incumbent(self.incumbent)
        self.objVal_record = [self.objVal.item()]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None

        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t, step):
        print(self.objVal, f'time:{t}')

    @property
    def ObjVal(self):
        return self.objVal

    @property
    def X(self):
        return self.incumbent

    @property
    def Runtime(self):
        return self.solving_time

    @property
    def obj_val(self):
        return self.objVal

    @property
    def objective_history(self):
        return self.objVal_record

    @property
    def timing_history(self):
        return self.timing_record

    def _init_variables(self, dual_init: float, primal_init: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key, subkey = jax.random.split(self.key)
        self.key = key
        shape = (self.batch_size, self.n)
        if primal_init == 'half':
            primal = jnp.full(shape, 0.5, dtype=jnp.float32)
        elif primal_init == 'binary':
            primal = jax.random.bernoulli(subkey, 0.5, shape).astype(jnp.float32)
        else:
            primal = jax.random.uniform(subkey, shape)
        dual = jnp.full((self.batch_size, self.n), dual_init, dtype=jnp.float32)
        return primal, dual

    def _validate_primal_initial_values(self, values: np.ndarray) -> jnp.ndarray:
        """Validate and batch a caller-supplied primal initialization."""
        supplied = np.asarray(values)
        if np.iscomplexobj(supplied):
            raise TypeError("primal_initial_values must be real-valued")
        if supplied.shape == (self.n,):
            supplied = np.broadcast_to(supplied, (self.batch_size, self.n)).copy()
        elif supplied.shape != (self.batch_size, self.n):
            raise ValueError(
                "primal_initial_values must have shape "
                f"({self.n},) or ({self.batch_size}, {self.n})"
            )
        supplied = np.asarray(supplied, dtype=np.float32)
        if not np.isfinite(supplied).all():
            raise ValueError("primal_initial_values must all be finite")
        if np.any(supplied < 0.0) or np.any(supplied > 1.0):
            raise ValueError("primal_initial_values must lie in [0, 1]")
        return jnp.asarray(supplied)

    def _initialize_curvature_matched_dual(self):
        """Match every initial Hessian to a common relative curvature level.

        For A = grad^2 f and d_i = g''(x_i), this sets
        y_i = shift / d_i. Hence diag(y * d) = shift * I for every batch,
        despite different random primal initializations.
        """
        from pdbo.curvature import matched_dual_from_curvature, quadratic_hessian

        g_fn = self._make_g()
        g_second = jax.grad(jax.grad(g_fn))
        curvature_fn = jax.vmap(jax.vmap(g_second))
        curvature = np.asarray(curvature_fn(self.primal), dtype=np.float64)
        hessian = quadratic_hessian(
            np.asarray(self.Q_indices),
            np.asarray(self.Q_values),
            self.n,
        )
        dual, diagnostics = matched_dual_from_curvature(
            hessian,
            curvature,
            relative_level=self.hessian_init_level,
            trusted_objective_lambda_min=self.trusted_objective_hessian_lambda_min,
            curvature_tol=self.curvature_tol,
            eig_tol=self.eig_tol,
        )
        self.dual = jnp.asarray(dual, dtype=self.primal.dtype)
        self.initial_curvature_diagnostics = diagnostics
        self.initial_g_curvature_min = float(np.min(curvature))
        self.initial_g_curvature_max = float(np.max(curvature))

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:
        if optimizer_type == 'sgd':
            return optax.sgd(lr)
        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)

    def _make_g(self) -> Callable:
        """Return the binarity constraint function g(x) selected by ``self.g_type``.

        Every choice is symmetric, vanishes at the endpoints, and is nonpositive on
        [0, 1], so the dual variables decrease monotonically. Only
        ``G_TYPES_CONVEX`` satisfies all of the paper's differentiable convexity
        conditions; see the group definitions for the experimental controls.

        When ``self.g_normalize`` is set, g is rescaled so its minimum depth |g(1/2)| is
        1, i.e. the value range becomes exactly [-1, 0]. This isolates the shape of g
        from its scale (the dual update y += beta*g(x) is otherwise sensitive to depth).
        """
        a = 2.0        # cosh sharpness
        cosh_a = math.cosh(a)
        eps = 1e-7     # guard log(0) / division-by-zero near the binary points

        def quad(x):        return x ** 2 - x
        def poly4(x):       return (2.0 * x - 1.0) ** 4 - 1.0
        def poly6(x):       return (2.0 * x - 1.0) ** 6 - 1.0
        def poly8(x):       return (2.0 * x - 1.0) ** 8 - 1.0
        def abs_pow3(x):    return jnp.abs(2.0 * x - 1.0) ** 3 - 1.0
        def abs_pow1p5(x):  return jnp.abs(2.0 * x - 1.0) ** 1.5 - 1.0

        def entropy(x):
            xc = jnp.clip(x, eps, 1.0 - eps)
            return xc * jnp.log(xc) + (1.0 - xc) * jnp.log(1.0 - xc)

        def sin_g(x):       return -jnp.sin(jnp.pi * x)

        def semicircle(x):
            xc = jnp.clip(x, 1e-6, 1.0 - 1e-6)  # keep sqrt argument strictly positive
            return -jnp.sqrt(xc - xc ** 2)

        # cosh, pre-scaled so its native range is already [-1, 0] (raw depth would be
        # cosh(a)-1 ~= 2.76, an outlier that would dominate the dual update otherwise)
        def cosh_g(x):      return (jnp.cosh(a * (2.0 * x - 1.0)) - cosh_a) / (cosh_a - 1.0)

        # partially convex: convex near x=1/2, concave near the edges
        def quartic_well(x): return -(x - x ** 2) ** 2
        def sin2(x):         return -jnp.sin(jnp.pi * x) ** 2

        # piecewise-linear V-shape (curvature 0 a.e.): an exact-penalty control
        def vshape(x):      return jnp.abs(2.0 * x - 1.0) - 1.0

        # Huber-smoothed V-shape: globally convex, but not strictly convex because
        # the outer regions remain linear. Only the central band has positive curvature.
        def huber(x, eps_h=0.05):
            c = 2.0 / (1.0 - eps_h)  # normalize so g(1/2) = -1
            t = x - 0.5
            quad = (c / (2.0 * eps_h)) * t ** 2 - c * (1.0 - eps_h) / 2.0
            lin = -c * (0.5 - jnp.abs(t))
            return jnp.where(jnp.abs(t) <= eps_h, quad, lin)

        # (function, minimum depth |g(1/2)|) used for optional [-1, 0] normalization
        table = {
            "quad": (quad, 0.25),
            "poly4": (poly4, 1.0),
            "poly6": (poly6, 1.0),
            "poly8": (poly8, 1.0),
            "abs_pow3": (abs_pow3, 1.0),
            "abs_pow1p5": (abs_pow1p5, 1.0),
            "entropy": (entropy, math.log(2.0)),
            "sin": (sin_g, 1.0),
            "semicircle": (semicircle, 0.5),
            "cosh": (cosh_g, 1.0),
            "huber": (huber, 1.0),
            "quartic_well": (quartic_well, 0.0625),
            "sin2": (sin2, 1.0),
            "vshape": (vshape, 1.0),
        }
        fn, depth = table[self.g_type]
        if self.g_normalize:
            scale = 1.0 / depth  # rescale minimum depth to 1 -> value range [-1, 0]
            return lambda x: fn(x) * scale
        return fn

    def _score_incumbent(self, x):
        if self.incumbent_score_fn is not None:
            return self.incumbent_score_fn(x[jnp.newaxis, :])[0]
        return self.objective_fn(x)

    def optimize(self):
        self.start_time = time.perf_counter()
        batch_obj = jax.vmap(self.objective_fn, in_axes=0)
        g_fn = self._make_g()

        def base_term(x):
            return batch_obj(x).sum()

        def penalty_term(x, y):
            return (y * g_fn(x)).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(x) + penalty_term(x, y), argnums=0)
        diagnostic_grad_fn = jax.jit(grad_x_fn) if self.state_callback is not None else None

        @jax.jit
        def primal_dual_update(x: jnp.ndarray, y: jnp.ndarray, opt_state_x, opt_state_y, objVal, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)
            x = jnp.clip(x, 0.0, 1.0)
            y += self.dual_lr * g_fn(x)

            int_x = jax.lax.stop_gradient(jnp.round(x))
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :]), axis=0)
            if self.incumbent_score_fn is not None:
                objs = self.incumbent_score_fn(int_x)
            else:
                objs = batch_obj(int_x)
            idx = jnp.argmin(objs)
            objVal = objs[idx]
            incumbent = int_x[idx]
            return x, y, opt_state_x, opt_state_y, objVal, incumbent

        @jax.jit
        def perturb_primal(x: jnp.ndarray, key):
            key_mask, key_sign = jax.random.split(key)
            fractional = jnp.abs(x - 0.5) < 0.45
            mask = fractional & (jax.random.uniform(key_mask, x.shape) < self.perturbation_fraction)
            signs = jax.random.bernoulli(key_sign, 0.5, x.shape)
            kicked = jnp.where(signs, 0.5 + self.perturbation_strength, 0.5 - self.perturbation_strength)
            return jnp.clip(jnp.where(mask, kicked, x), 0.0, 1.0)

        last_improvement_step = 0
        for _step in range(self.max_iters):
            t = time.perf_counter()
            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal,
             self.incumbent) = primal_dual_update(
                self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal, self.incumbent)

            state_callback_due = (
                    self.state_callback is not None
                    and (
                            (_step + 1) % self.state_callback_every == 0
                            or _step == self.max_iters - 1
                    )
            )
            should_check = (
                    self.verbose
                    or self.step_callback is not None
                    or state_callback_due
                    or (_step + 1) % self.check_every == 0
                    or _step == self.max_iters - 1
            )
            if should_check:
                self.objVal.block_until_ready()
                current_obj = self.objVal.item()
                if current_obj < self.objVal_record[-1] - self.min_delta:
                    self.objVal_record.append(current_obj)
                    self.timing_record.append(time.perf_counter() - self.start_time)
                    last_improvement_step = _step
                step_time = time.perf_counter() - t
                self.log(step_time, _step)
                if self.step_callback is not None:
                    self.incumbent.block_until_ready()
                    self.step_callback(_step, step_time, self.objVal, self.incumbent)
                if state_callback_due:
                    diagnostic_gradient = diagnostic_grad_fn(self.primal, self.dual)
                    diagnostic_gradient.block_until_ready()
                    self.state_callback(
                        _step,
                        self.primal,
                        self.dual,
                        diagnostic_gradient,
                        self.objVal,
                        self.incumbent,
                        last_improvement_step,
                    )
                if self.timelimit is not None and time.perf_counter() - self.start_time >= self.timelimit:
                    self.stop_reason = "timelimit"
                    break
                if (
                        self.perturbation
                        and _step - last_improvement_step >= self.perturbation_patience
                ):
                    integrality = (self.primal * (1.0 - self.primal)).mean()
                    integrality.block_until_ready()
                    if float(integrality) > self.perturbation_integrality_tol:
                        self.key, subkey = jax.random.split(self.key)
                        self.primal = perturb_primal(self.primal, subkey)
                        if self.perturbation_reset_optimizer:
                            self.opt_state_primal = self.optimizer_primal.init(self.primal)
                        last_improvement_step = _step
                        self.perturbation_count += 1
                if self.patience is not None and _step - last_improvement_step >= self.patience:
                    self.stop_reason = "patience"
                    break
        else:
            self.stop_reason = "max_iters"

        self.solving_time = time.perf_counter() - self.start_time

        # Convergence diagnostics (for deciding whether a run is comparable):
        #   - fractionality x*(1-x) in [0, 0.25], g-independent so fair across g's;
        #     0 means the relaxed primal has fully settled onto a binary point.
        #   - last_improvement_step: when the incumbent objective last improved; if it
        #     is near max_iters the run was probably still climbing at the budget.
        frac = self.primal * (1.0 - self.primal)
        self.final_integrality = float(frac.mean())              # over the whole batch
        self.final_integrality_min = float(frac.mean(axis=1).min())  # most-binary init
        self.last_improvement_step = int(last_improvement_step)

        if self.rounding_samples > 0:
            self.key, subkey = jax.random.split(self.key)
            sampled = jax.random.bernoulli(
                subkey,
                p=self.primal,
                shape=(self.rounding_samples,) + self.primal.shape,
            ).astype(jnp.float32)
            sampled = sampled.reshape((-1, self.n))
            candidates = jnp.concatenate((sampled, self.incumbent[jnp.newaxis, :]), axis=0)
            if self.incumbent_score_fn is not None:
                objs = self.incumbent_score_fn(candidates)
            else:
                objs = batch_obj(candidates)
            idx = jnp.argmin(objs)
            candidate_obj = objs[idx]
            candidate_obj.block_until_ready()
            if float(candidate_obj) < float(self.objVal) - self.min_delta:
                self.objVal = candidate_obj
                self.incumbent = candidates[idx]
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
                self.solving_time = time.perf_counter() - self.start_time
        return PDBOResult(
            objective=float(self.objVal),
            incumbent=np.asarray(self.incumbent, dtype=np.int32),
            objective_history=list(self.objVal_record),
            timing_history=list(self.timing_record),
            runtime=self.solving_time,
            stop_reason=self.stop_reason,
        )


class PDQUBO_JAX(PDBO_JAX):
    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: jnp.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            tolerance: float = 1e-8,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            timelimit: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            primal_init: str = 'uniform',
            step_callback: Optional[Callable] = None,
            state_callback: Optional[Callable] = None,
            state_callback_every: int = 100,
            incumbent_score_fn: Optional[Callable] = None,
            patience: Optional[int] = None,
            min_delta: float = 0.0,
            check_every: int = 1,
            quadratic_backend: str = 'sparse',
            rounding_samples: int = 0,
            perturbation: bool = False,
            perturbation_strength: float = 0.4,
            perturbation_fraction: float = 0.05,
            perturbation_patience: int = 200,
            perturbation_integrality_tol: float = 0.2,
            perturbation_reset_optimizer: bool = True,
            dual_init_mode: str = 'constant',
            hessian_init_level: float = 0.0,
            curvature_tol: float = 1e-12,
            eig_tol: float = 1e-8,
            trusted_objective_hessian_lambda_min: Optional[float] = None,
    ):
        super().__init__(
            n_vars=n_vars,
            objective_type='quadratic',
            Q_indices=Q_indices,
            Q_values=Q_values,
            c=c,
            optimizer_type=optimizer_type,
            batch_size=batch_size,
            primal_lr=primal_lr,
            tolerance=tolerance,
            dual_lr=dual_lr,
            dual_init=dual_init,
            dual_init_mode=dual_init_mode,
            hessian_init_level=hessian_init_level,
            curvature_tol=curvature_tol,
            eig_tol=eig_tol,
            trusted_objective_hessian_lambda_min=trusted_objective_hessian_lambda_min,
            max_iters=max_iters,
            timelimit=timelimit,
            seed=seed,
            verbose=verbose,
            primal_init=primal_init,
            step_callback=step_callback,
            state_callback=state_callback,
            state_callback_every=state_callback_every,
            incumbent_score_fn=incumbent_score_fn,
            patience=patience,
            min_delta=min_delta,
            check_every=check_every,
            quadratic_backend=quadratic_backend,
            rounding_samples=rounding_samples,
            perturbation=perturbation,
            perturbation_strength=perturbation_strength,
            perturbation_fraction=perturbation_fraction,
            perturbation_patience=perturbation_patience,
            perturbation_integrality_tol=perturbation_integrality_tol,
            perturbation_reset_optimizer=perturbation_reset_optimizer,
        )


class MAX_K_CUT_JAX:

    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: jnp.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            seed: int = 0,
            k: int = 3,
            verbose: bool = True
    ):

        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"

        self.key = jax.random.PRNGKey(seed)
        self.n = n_vars
        self.m = Q_indices.shape[1]
        self.batch_size = batch_size
        self.Q_indices = Q_indices
        self.Q_values = Q_values
        self.Q = sparse.BCOO((Q_values, jnp.column_stack(Q_indices)), shape=(n_vars, n_vars))
        self.Q_sum = self.Q.sum()
        self.k = k
        self.c = c
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.verbose = verbose

        self.primal, self.dual = self._init_variables(dual_init)
        incumbent = jnp.full((self.k - 1, self.n), 0, dtype=jnp.float32)
        incumbet_last_row = jnp.full((1, self.n), 1, dtype=jnp.float32)
        self.incumbent = jnp.concatenate((incumbent, incumbet_last_row), axis=0)

        self.objVal = jnp.array(0.)
        self.objVal_record = [0.]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None


        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.cur_lag = jnp.array(0.)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t):
        print(self.objVal, f'time:{t}')

    def _init_variables(self, dual_init: float) -> Tuple[
        jnp.ndarray, jnp.ndarray]:
        key, subkey = jax.random.split(self.key)
        shape = (self.batch_size, self.k, self.n)
        primal = jax.random.uniform(subkey, shape)
        dual = jnp.full((self.batch_size, self.n), dual_init, dtype=jnp.float32)

        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:

        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)



    def optimize(self):
        self.start_time = time.perf_counter()


        def obj(x):
            # x [k, n]
            return (jnp.trace(x @ self.Q @ x.T) - self.Q_sum) / 2


        batch_obj = jax.vmap(obj, in_axes=0)

        def base_term(x):
            return batch_obj(x).sum()
        def penalty_term(x, y):
            #  p[b, k, n] //  y [b, n]
            return (y * ((x ** 2).sum(1) - 1)).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(norm_vmap(x)) + penalty_term(norm_vmap(x), y), argnums=0)
        norm_vmap = jax.vmap(jax.vmap(lambda x: jnp.abs(x) / jnp.sum(jnp.abs(x)), (-1), -1), (0,), 0)


        @jax.jit
        def primal_dual_update(x, y, opt_state_x, opt_state_y, best_obj, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)
            x = jnp.clip(x, 0.0)
            p = norm_vmap(x)
            y += self.dual_lr * ((p ** 2).sum(1) - 1)

            one_hot_indices = jnp.argmax(x, axis=1)
            int_x = jax.lax.stop_gradient(jax.nn.one_hot(one_hot_indices, x.shape[1], axis=1))  # int_x [b, k, n]
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :, :]), axis=0)

            objs = batch_obj(int_x)
            idx = jnp.argmin(objs)
            best_obj = objs[idx]
            incumbent = int_x[idx]

            return x, y, opt_state_x, opt_state_y, incumbent, best_obj

        for _step in range(self.max_iters):
            t = time.perf_counter()

            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.incumbent, self.objVal) = primal_dual_update(
                x=self.primal, y=self.dual, opt_state_x=self.opt_state_primal,
                opt_state_y=self.opt_state_dual, best_obj=self.objVal, incumbent=self.incumbent
            )
            # print(base_term(self.incumbent[jnp.newaxis, :, :]))

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()
            self.cur_lag.block_until_ready()
            self.objVal.block_until_ready()
            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            self.log(time.perf_counter() - t)
        self.solving_time = time.perf_counter() - self.start_time



class MAXSAT_JAX:
    def __init__(
            self,
            n_vars: int,
            CNF: np.ndarray,
            optimizer_type: str = 'rmsprop',
            batch_size: int = 1,
            primal_lr: float = 0.001,
            dual_lr: float = 0.001,
            dual_init: float = 4,
            max_iters: int = 5000,
            seed: int = 0,
            verbose: bool = True,
    ):

        assert optimizer_type in {'rmsprop', 'adam'}, "Invalid optimizer type"


        self.key = jax.random.PRNGKey(seed)

        self.num_vars = n_vars
        self.num_clause = CNF.shape[0]
        self.batch_size = batch_size
        self.indices = np.abs(CNF) - 1
        self.sign = np.sign(CNF)
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.max_iters = max_iters
        self.verbose = verbose

        self.primal, self.dual = self._init_variables(dual_init)
        self.incumbent = jnp.full(self.num_vars, 0, dtype=jnp.float32)

        self.objVal = jnp.prod(jnp.where(self.sign < 0,
                                         self.incumbent[self.indices],
                                         1 - self.incumbent[self.indices]), axis=1).sum()


        self.objVal_record = [self.objVal.item()]
        self.timing_record = [0.]
        self.start_time = None
        self.solving_time = None

        self.optimizer_primal = self._configure_optimizer(optimizer_type, primal_lr)
        self.optimizer_dual = self._configure_optimizer(optimizer_type, dual_lr)
        self.opt_state_primal = self.optimizer_primal.init(self.primal)
        self.opt_state_dual = self.optimizer_dual.init(self.dual)
        self.log = self._log_verbose if self.verbose else lambda *args, **kwargs: None

    def _log_verbose(self, t):
        # print(self.objVal, f'time:{t}, integrality:{(self.primal - self.primal ** 2).mean()}, dual:{self.dual.mean()}')
        print(self.objVal)
    def _init_variables(self, dual_init: float) -> Tuple[
        jnp.ndarray, jnp.ndarray]:

        key, subkey = jax.random.split(self.key)

        shape = (self.batch_size, self.num_vars)

        primal = jax.random.uniform(subkey, shape)

        dual = jnp.full((self.batch_size, self.num_vars), dual_init, dtype=jnp.float32)

        return primal, dual

    def _configure_optimizer(self, optimizer_type: str, lr: float) -> optax.GradientTransformation:

        if optimizer_type == 'rmsprop':
            return optax.rmsprop(lr, decay=0.98, eps=1e-8, momentum=0.91)
        return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)

    def optimize(self):
        self.start_time = time.perf_counter()

        def obj(x):
            # The number of UNsatisfied clauses
            x_literal = jnp.where(self.sign < 0,
                                  x[self.indices],
                                  1 - x[self.indices])  # shape [m, k]
            clause_values = jnp.prod(x_literal, axis=1)  # shape [m]
            return jnp.sum(clause_values)

        batch_obj = jax.vmap(obj, in_axes=0)

        def base_term(x):
            # x [b, n]
            return batch_obj(x).sum()

        def penalty_term(x, y):
            return (y * (x**2-x)).sum()

        grad_x_fn = jax.grad(lambda x, y: base_term(x) + penalty_term(x, y), argnums=0)

        @jax.jit
        def primal_dual_update(x: jnp.ndarray, y: jnp.ndarray, opt_state_x, opt_state_y, objVal, incumbent):
            grad_x = grad_x_fn(x, y)
            updates_x, opt_state_x = self.optimizer_primal.update(grad_x, opt_state_x, x)
            x = optax.apply_updates(x, updates_x)

            x = jnp.clip(x, 0, 1)
            y += self.dual_lr * (x **2 - x)

            ### Update incumbent
            int_x = jax.lax.stop_gradient(jnp.round(x))
            int_x = jnp.concatenate((int_x, incumbent[jnp.newaxis, :]), axis=0)
            objs = batch_obj(int_x)
            # print(objs.shape)
            # print(objs)
            idx = jnp.argmin(objs)
            objVal = objs[idx]
            incumbent = int_x[idx]

            return x, y, opt_state_x, opt_state_y, objVal, incumbent

        for _step in range(self.max_iters):
            t = time.perf_counter()
            (self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal,
             self.incumbent) = primal_dual_update(
                self.primal, self.dual, self.opt_state_primal, self.opt_state_dual, self.objVal, self.incumbent)

            self.primal.block_until_ready()
            self.dual.block_until_ready()
            self.incumbent.block_until_ready()
            self.objVal.block_until_ready()
            if self.objVal < self.objVal_record[-1]:
                self.objVal_record.append(self.objVal.item())
                self.timing_record.append(time.perf_counter() - self.start_time)
            self.log(time.perf_counter() - t)

        self.solving_time = time.perf_counter() - self.start_time
