# Hessian-boundary initialization for general binarity functions

This note generalizes the Max-Cut calculation in Section 4.4 (page 8) of
*Smoothing Binary Optimization: A Primal-Dual Perspective*
(arXiv:2509.21064v2). The purpose is to initialize PDBO at, or at a controlled
distance from, the convex/nonconvex boundary. All statements below concern the
Hessian with the dual variable held fixed.

## 1. The Max-Cut special case

The paper writes the minimization form of Max-Cut as

$$
f(x)=x^T W x-\mathbf 1^T W x,
$$

where $W=W^T$ and $\operatorname{diag}(W)=0$, and uses
$g(s)=s^2-s$. Thus

$$
L(x,y)=f(x)+\sum_{i=1}^n y_i g(x_i), \qquad
\nabla_{xx}^2 L(x,y)=2\bigl(W+\operatorname{diag}(y)\bigr).
$$

For the scalar initialization $y=\bar y\mathbf 1$,

$$
\nabla_{xx}^2 L=2(W+\bar y I).
$$

Consequently, $\bar y\geq-\lambda_{\min}(W)$ makes the Lagrangian
convex in $x$. The exact boundary value is

$$
\boxed{\bar y_*=-\lambda_{\min}(W)},
$$

for which the Hessian is positive semidefinite and singular. If $v_*$ is a
minimum-eigenvalue eigenvector of $W$, then

$$
\nabla_{xx}^2 L\succeq0,
\qquad
v_*^T\nabla_{xx}^2 L\,v_*=0.
$$

The inequality in the paper describes the entire convex side. Equality, not a
strict inequality, selects the boundary.

## 2. General local Hessian

Let $f:[0,1]^n\to\mathbb R$ and $g:[0,1]\to\mathbb R$ be twice
differentiable at the point under consideration. Define

$$
A(x):=\nabla^2 f(x),\qquad
d_i(x):=g''(x_i),\qquad
D(x):=\operatorname{diag}(d_1(x),\ldots,d_n(x)).
$$

Then

$$
\boxed{H(x,y):=\nabla_{xx}^2L(x,y)
=A(x)+\operatorname{diag}\bigl(y_i d_i(x)\bigr).}
$$

This identity is local in $x$. For a quadratic objective, $A$ is constant. For
a higher-order multilinear extension, $A(x)$ generally changes with $x$.

A symmetric matrix is on the convex/nonconvex boundary precisely when

$$
H\succeq0 \quad\text{and}\quad \lambda_{\min}(H)=0.
$$

Equivalently, $H\succeq0$ and there is a nonzero $v$ with $v^THv=0$.
The PSD condition must not be omitted: an indefinite matrix can also have a
nonzero isotropic vector. For example,
$\operatorname{diag}(-1,1)$ has $v^THv=0$ for $v=(1,1)^T$, but it is not a
boundary PSD matrix. When $H\succeq0$, $v^THv=0$ additionally implies $Hv=0$.

## 3. One scalar dual value at a fixed point

Assume first that

$$
D(x)\succ0,
$$

which is stronger than the conditions imposed on $g$ in the paper. With
$y=\bar y\mathbf 1$, define

$$
B(x):=D(x)^{-1/2}A(x)D(x)^{-1/2}.
$$

Congruence gives

$$
H(x,\bar y\mathbf 1)
=D^{1/2}\bigl(B+\bar y I\bigr)D^{1/2}.
$$

Therefore

$$
\boxed{
\bar y_*(x)
=-\lambda_{\min}\!\left(D(x)^{-1/2}A(x)D(x)^{-1/2}\right)
=-\min_{v\ne0}\frac{v^TA(x)v}{v^TD(x)v}.}
$$

It follows that

$$
H(x,\bar y\mathbf 1)\succeq0
\quad\Longleftrightarrow\quad
\bar y\geq\bar y_*(x),
$$

and equality makes the Hessian singular. This is the smallest generalized
eigenvalue of the pencil $(A,D)$, not in general
$-\lambda_{\min}(A)/\min_i d_i$; the latter is only a conservative bound.

If $A(x)$ is indefinite, then $\bar y_*(x)>0$, as required by the paper's
positive initialization. If $A(x)\succeq0$, the nonnegative ray is already
convex. In particular, if $A(x)\succ0$, its singular boundary lies at a
negative scalar dual value and cannot be reached while requiring $\bar y\geq0$.

The same calculation covers a nonuniform positive ray. Given $q\in\mathbb
R_{++}^n$, set $y=tq$ and

$$
C_q(x):=\operatorname{diag}\bigl(q_i d_i(x)\bigr)\succ0.
$$

Then

$$
t_*(x;q)=-\lambda_{\min}
\left(C_q(x)^{-1/2}A(x)C_q(x)^{-1/2}\right),
\qquad y_*(x;q)=t_*(x;q)q.
$$

Thus a vector initialization becomes well defined once its ray $q$ is fixed;
without such a choice there are generally infinitely many boundary vectors.

## 4. Explicit coordinatewise boundary and relative curvature

When every $d_i(x)>0$, a particularly simple vector construction equalizes the
added curvature. Suppose $a(x):=-\lambda_{\min}(A(x))>0$ and set

$$
\boxed{y_{*,i}^{\rm eq}(x)=\frac{a(x)}{d_i(x)}.}
$$

It gives

$$
\operatorname{diag}\bigl(y_{*,i}^{\rm eq}d_i\bigr)=aI,
\qquad
H(x,y_*^{\rm eq})=A(x)+aI\succeq0,
$$

with minimum eigenvalue zero. This is the preceding ray formula with
$q_i=1/d_i(x)$. It is exact, but a small $d_i$ produces a large $y_i$ and can
make the dual dynamics badly scaled.

For controlled departures from the boundary, define the dimensionless
relative curvature level $r$ by

$$
y_i(r;x)=(1+r)y_{*,i}^{\rm eq}(x).
$$

Then

$$
H(x,y(r;x))=A(x)+(1+r)a(x)I,
\qquad
\lambda_{\min}(H)=r\,a(x).
$$

Hence $r=0$ is the boundary, $r>0$ is the strictly convex side,
$-1<r<0$ is a controlled nonconvex initialization with positive dual
variables, and $r=-1$ is the unpenalized Hessian $A(x)$. This definition is
invariant to a positive rescaling of $g$: scaling $g$ rescales $y$ inversely
but leaves $y_i g''(x_i)$ unchanged.

For a scalar initialization, use the analogous generalized-curvature level.
If $\bar y_*(x)>0$, set

$$
\bar y(r;x)=(1+r)\bar y_*(x).
$$

The minimum eigenvalue of
$D^{-1/2}H D^{-1/2}=B+\bar y I$ is then
$r\bar y_*(x)$. Congruence preserves PSD and inertia, although this generalized
minimum curvature is not the ordinary $\lambda_{\min}(H)$ when $D$ is not a
multiple of the identity.

## 5. A threshold valid over the whole cube

A local boundary at $x^0$ does not imply that $L(\cdot,y^0)$ is globally
convex. Under the stronger assumptions

$$
f\in C^2([0,1]^n),\qquad g\in C^2([0,1]),\qquad
g''(s)\geq m>0\quad\text{for every }s\in[0,1],
$$

the exact scalar threshold for global convexity is

$$
\boxed{
\bar y_{\rm global}
=\sup_{x\in[0,1]^n}
\left[-\lambda_{\min}
\left(D(x)^{-1/2}A(x)D(x)^{-1/2}\right)\right].}
$$

Every $\bar y\geq\bar y_{\rm global}$ makes
$L(\cdot,\bar y\mathbf1)$ convex on the cube. With continuity and compactness,
the supremum is attained, so at $\bar y_{\rm global}$ at least one Hessian is
singular. Other points can be strictly convex; one scalar cannot generally put
every $H(x,\bar y\mathbf1)$ on its boundary simultaneously.

A simpler sufficient, but generally non-tight, bound is

$$
M:=\sup_{x\in[0,1]^n}\max\{0,-\lambda_{\min}(A(x))\},
\qquad
\bar y\geq M/m.
$$

For Max-Cut, $A=2W$ is constant. If
$m_g=\min_{s\in(0,1)}g''(s)>0$ and the minimum is attained, this bound is
exact because all coordinates can be set to a common minimizer of $g''$:

$$
\boxed{\bar y_{\rm global}=-\frac{2\lambda_{\min}(W)}{m_g}.}
$$

The quadratic choice has $m_g=2$ and recovers
$-\lambda_{\min}(W)$. Raw entropy has $m_g=4$ on the open interval and gives
$-\lambda_{\min}(W)/2$ for all interior Hessians, with equality at
$x=\tfrac12\mathbf1$. A positive rescaling of $g$ rescales this dual threshold
inversely.

For the paper's quadratic Max-Cut model, $A=2W$ and $D=2I$, so the local and
global formulas both reduce to $-\lambda_{\min}(W)$.

For a fixed coordinatewise vector $y$, the exact global feasible set is the
semi-infinite spectrahedron

$$
\mathcal Y_{\rm global}
=\{y:A(x)+\operatorname{diag}(y_i g''(x_i))\succeq0
\text{ for every }x\in[0,1]^n\}.
$$

Its boundary is in general a semi-infinite semidefinite problem, not a single
closed-form vector. The pointwise formula in Section 4 should therefore be
described as a local Hessian initialization, not a global convex extension.

## 6. What changes when $D$ is singular

Convexity of $g$ only gives $D\succeq0$ at points where $g''$ exists; it does
not give $D\succ0$. The following fixed-$x$ condition is exact. Split the space
orthogonally as $\operatorname{range}(D)\oplus\ker(D)$ and write

$$
D=\begin{bmatrix}D_R&0\\0&0\end{bmatrix},\qquad
A=\begin{bmatrix}A_{RR}&A_{RN}\\A_{NR}&A_{NN}\end{bmatrix},
\qquad D_R\succ0.
$$

There exists a finite scalar $\bar y$ such that $A+\bar yD\succeq0$ if and
only if

$$
A_{NN}\succeq0,
\qquad
\operatorname{range}(A_{NR})\subseteq\operatorname{range}(A_{NN}).
$$

The first condition says that an unpenalized direction cannot have negative
curvature. The second prevents coupling to a zero-curvature direction that
would make every block matrix indefinite. When they hold, define the generalized
Schur complement

$$
S=A_{RR}-A_{RN}A_{NN}^{\dagger}A_{NR}.
$$

The least scalar loading is

$$
\bar y_*=-\lambda_{\min}(D_R^{-1/2}SD_R^{-1/2}).
$$

This is also a necessary-and-sufficient description of the exceptional cases
where a finite threshold survives despite zero entries of $g''$. If $D=0$,
the dual term cannot change the Hessian at all. If $D$ has negative entries,
increasing a positive scalar dual can itself add negative curvature, and the
one-sided threshold formulas above do not apply.

## 7. Why the paper's conditions on $g$ are insufficient

Section 3.1 requires $g$ to be convex and continuous on $[0,1]$, differentiable
only on $(0,1)$, symmetric, zero at the endpoints, and to have its unique
interior stationary point at $1/2$. These conditions correctly encode binarity,
but they do not guarantee any of the following properties needed by a Hessian
boundary construction:

1. existence of $g''$ on the closed interval;
2. strict positivity of $g''$ at the initialization;
3. a uniform lower bound $\inf_{s\in[0,1]}g''(s)>0$.

Two examples used by the paper make the distinction concrete.

For

$$
g_{\rm poly4}(s)=(2s-1)^4-1,
\qquad g_{\rm poly4}''(s)=48(2s-1)^2,
$$

all of the paper's conditions hold, but $g''(1/2)=0$. At the common half
initialization $x=\tfrac12\mathbf1$, $D=0$, so no finite scalar or vector dual
value changes the Hessian. If $A(x)$ is indefinite, a PSD boundary
initialization is impossible there. Near $1/2$, the explicit coordinatewise
dual values also diverge like $(2x_i-1)^{-2}$.

For

$$
g_{\rm ent}(s)=s\log s+(1-s)\log(1-s),
\qquad g_{\rm ent}''(s)=\frac{1}{s(1-s)}\quad(0<s<1),
$$

the interior curvature is positive and at least $4$, but the derivatives are
not finite at $0$ and $1$. Thus a classical Hessian initialization is valid on
a restricted interior box $[\epsilon,1-\epsilon]^n$, not on the closed cube.
The implementation in `solver_jax.py` clips the entropy argument. That clipped
function is not the same smooth function at the clipping breakpoints and has
flat clipped regions, so its autodifferentiated curvature must be analyzed as
implemented rather than substituted from the raw entropy formula.

Accordingly, an "arbitrary admissible $g$" theorem must add a curvature
assumption. Pointwise initialization needs $g''(x_i)>0$ for every coordinate
(or the singular-$D$ compatibility conditions above). A finite, readily
computable global threshold is guaranteed by $C^2$ regularity and strong
convexity, $g''\geq m>0$.

## 8. Batch semantics in PDBO

With $B$ independent primal starts, the solver stores $x,y\in\mathbb R^{B\times
n}$. The summed batch objective has a block-diagonal primal Hessian,

$$
H_{\rm batch}=\operatorname{blkdiag}(H_1,\ldots,H_B).
$$

There are three distinct initialization claims:

- A single scalar $\bar y$ places the full batch matrix on the PSD boundary by
  taking $\bar y=\max_b\bar y_*(x^{(b)})$. Every block is then PSD, but normally
  only a worst-case block is singular.
- A per-start scalar $\bar y_b=\bar y_*(x^{(b)})$ places every block on its own
  boundary. This requires batch-dependent dual initialization.
- A per-start, per-coordinate vector from Section 4 places every block on its
  ordinary-curvature boundary, provided all required $g''(x_i^{(b)})$ are
  positive.

It is generally impossible for one shared scalar to make every independently
sampled block singular. Moreover, any of these pointwise certificates applies
only at iteration zero; after both $x$ and $y$ update, the Hessian must be
recomputed if its inertia is to be tracked.

## 9. Recommended experiment

To test whether PDBO benefits from beginning in a globally convex landscape,
compare matched curvature levels rather than raw dual values. A useful initial
grid is

$$
r\in\{+0.1,\ 0,\ -0.1,\ -0.25,\ -0.5,\ -1\}.
$$

This includes a convex-side control, the exact boundary, three progressively
nonconvex starts with positive dual variables, and the unpenalized objective.
Use the same primal starts for every level and every $g$. Normalizing only the
depth $|g(1/2)|$ does not normalize $g''$; the relative-curvature construction
does.

For each instance, seed, batch start, and $g$, record at least
$\lambda_{\min}(A)$, the generalized threshold, the chosen dual value,
$\lambda_{\min}(H)$, an estimate of $\|H\|_2$, the number of negative
eigenvalues when practical, and the boundary eigenvector residual
$\|Hv\|_2/\|v\|_2$. Also record objective quality, integrality, time to best
incumbent, convergence/plateau status, and perturbation count. On large sparse
instances, use a Lanczos extremal-eigenvalue calculation rather than forming a
dense Hessian.

Numerical labels should be scale aware. For example, declare PSD when

$$
\lambda_{\min}(H)\geq-\tau\max\{1,\|H\|_2\},
$$

and declare the boundary only when PSD holds and
$|\lambda_{\min}(H)|\leq\tau\max\{1,\|H\|_2\}$. Checking only a small Rayleigh
quotient $v^THv$ is not a valid convexity certificate.
