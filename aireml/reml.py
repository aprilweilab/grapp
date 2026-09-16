"""
Average-information REML (AI-REML) for variance components of a linear mixed
model whose covariance components are given as linear operators.

The model is

    y | X, b, theta ~ Normal(X b, V),   V = sum_i theta_i V_i,

with the last component conventionally the identity (residual variance).
Writing ``P`` for the REML projection matrix, the objective (minus twice the
restricted log likelihood) has

    gradient_i  = trace(P V_i) - y^T P V_i P y
    AI_ij       = y^T P V_i P V_j P y

and the update is ``theta <- theta - AI^-1 gradient`` (Gilmour et al. 1995;
Lee et al. 2026).  The traces are the only terms that are not a handful of
matrix-vector products, and they are estimated stochastically; the average
information itself is exact, because the trace terms in the Hessian and in the
Fisher information cancel when they are averaged.
"""

import time
from typing import Callable, List, Optional, Sequence, Union

import numpy
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from .operators import (
    Coefficients,
    as_symmetric_operator,
    identity_operator,
    materialize,
)
from .model import RemlState
from .solvers import ConjugateGradientSolver, CovarianceSolver, DenseSolver
from .trace import estimate_trace, rademacher

__all__ = ["REMLResult", "fit_reml", "haseman_elston"]


OperatorLike = Union[numpy.ndarray, LinearOperator]


def _as_operator_list(relatedness) -> List[LinearOperator]:
    if isinstance(relatedness, (list, tuple)):
        return [as_symmetric_operator(op) for op in relatedness]
    return [as_symmetric_operator(relatedness)]


def _residual_maker(
    covariates: numpy.ndarray,
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """Return ``block -> (I - X (X^T X)^-1 X^T) block``."""
    basis, _ = numpy.linalg.qr(covariates)

    def _apply(block: numpy.ndarray) -> numpy.ndarray:
        return block - basis @ (basis.T @ block)

    return _apply


def haseman_elston(
    y: numpy.ndarray,
    operators: Sequence[LinearOperator],
    covariates: numpy.ndarray,
    num_vectors: int = 50,
    rng: Optional[numpy.random.Generator] = None,
    exact: bool = False,
) -> numpy.ndarray:
    """
    Randomized Haseman-Elston moment estimates of the variance components.

    Solves the moment equations ``S theta = c`` with

        S_ij = trace(M V_i M V_j),   c_i = (M y)^T V_i (M y),

    where ``M`` projects out the covariates (Wu and Sankararaman 2018; Lee
    et al. 2026, equation 17).  ``S`` is estimated with Hutchinson's estimator
    using ``num_vectors`` shared test vectors, which costs ``num_vectors``
    products per component -- far less than one AI-REML iteration, which is
    why this is used as the starting point for AI-REML.

    Negative solutions are clipped to a small positive value.
    """
    y = numpy.asarray(y, dtype=numpy.float64).ravel()
    size = y.shape[0]
    num_components = len(operators)
    residualize = _residual_maker(covariates)

    if exact:
        dense = [residualize(materialize(op).T).T for op in operators]
        dense = [residualize(matrix) for matrix in dense]  # M V_i M
        gram = numpy.array(
            [[float(numpy.sum(left * right.T)) for right in dense] for left in dense]
        )
    else:
        rng = numpy.random.default_rng() if rng is None else rng
        num_vectors = min(num_vectors, size)
        test = residualize(rademacher(size, num_vectors, rng))
        products = [numpy.asarray(op.matmat(test)) for op in operators]
        projected = [residualize(product) for product in products]
        gram = numpy.array(
            [
                [float(numpy.sum(left * right)) / num_vectors for right in projected]
                for left in products
            ]
        )
        gram = 0.5 * (gram + gram.T)

    residual_y = residualize(y.reshape(-1, 1)).ravel()
    moments = numpy.array(
        [float(residual_y @ numpy.asarray(op.matvec(residual_y))) for op in operators]
    )
    try:
        estimates = numpy.linalg.solve(gram, moments)
    except numpy.linalg.LinAlgError:  # pragma: no cover - singular moment system
        estimates = numpy.linalg.lstsq(gram, moments, rcond=None)[0]
    total = max(float(numpy.var(residual_y)), numpy.finfo(float).tiny)
    floor = 1e-4 * total
    if not numpy.all(numpy.isfinite(estimates)):
        return numpy.full(num_components, total / num_components)
    return numpy.maximum(estimates, floor)


class REMLResult:
    """
    Fitted AI-REML model.

    Attributes
    ----------
    variance_components:
        Estimated ``theta``, one per covariance component, in the order the
        components were supplied with the residual variance last.
    genetic_variance, residual_variance, heritability:
        Convenience views for the common one-GRM case.
    standard_errors, covariance:
        Approximate sampling covariance ``2 AI^-1`` of ``theta`` and its
        square-rooted diagonal.
    fixed_effects:
        Generalized least squares estimate of ``b`` at the fitted ``theta``.
    projection:
        ``P y`` at the fitted ``theta``; the only ``N``-vector needed for
        BLUPs.
    """

    def __init__(
        self,
        variance_components: numpy.ndarray,
        state: RemlState,
        genetic_operators: Sequence[LinearOperator],
        gradient: numpy.ndarray,
        average_information: numpy.ndarray,
        history: List[numpy.ndarray],
        converged: bool,
        num_iterations: int,
        initial_values: numpy.ndarray,
        elapsed: float,
    ):
        self.variance_components = numpy.asarray(variance_components)
        self.initial_values = numpy.asarray(initial_values)
        self.gradient = numpy.asarray(gradient)
        self.average_information = numpy.asarray(average_information)
        self.history = [numpy.asarray(item) for item in history]
        self.converged = bool(converged)
        self.num_iterations = int(num_iterations)
        self.elapsed = float(elapsed)
        self.fixed_effects = state.fixed_effects
        self.projection = state.projected_y
        self.num_matvecs = state.solver.num_matvecs
        self.num_solves = state.solver.num_solves
        self._state = state
        self._genetic_operators = list(genetic_operators)

        try:
            self.covariance = 2.0 * numpy.linalg.inv(self.average_information)
        except numpy.linalg.LinAlgError:  # pragma: no cover
            self.covariance = numpy.full_like(self.average_information, numpy.nan)
        self.standard_errors = numpy.sqrt(numpy.abs(numpy.diag(self.covariance)))

    # ------------------------------------------------------------------
    @property
    def residual_variance(self) -> float:
        return float(self.variance_components[-1])

    @property
    def genetic_variance(self) -> float:
        """Sum of the non-residual variance components."""
        return float(numpy.sum(self.variance_components[:-1]))

    @property
    def heritability(self) -> float:
        """``sum(theta_genetic) / sum(theta)``."""
        total = float(numpy.sum(self.variance_components))
        return self.genetic_variance / total if total > 0 else float("nan")

    @property
    def heritability_stderr(self) -> float:
        """Delta-method standard error of :attr:`heritability`."""
        total = float(numpy.sum(self.variance_components))
        if total <= 0 or not numpy.all(numpy.isfinite(self.covariance)):
            return float("nan")
        genetic = self.genetic_variance
        gradient = numpy.empty(self.variance_components.shape[0])
        gradient[:-1] = (total - genetic) / total**2
        gradient[-1] = -genetic / total**2
        variance = float(gradient @ self.covariance @ gradient)
        return float(numpy.sqrt(variance)) if variance > 0 else float("nan")

    # ------------------------------------------------------------------
    def blup(self) -> numpy.ndarray:
        """
        Best linear unbiased predictions of the genetic values of the
        individuals that were fitted: ``g_hat = sum_i theta_i V_i P y``.
        """
        weights = self.variance_components[: len(self._genetic_operators)]
        total = numpy.zeros(self._state.size)
        for weight, operator in zip(weights, self._genetic_operators):
            total = total + weight * numpy.asarray(operator.matvec(self.projection))
        return total

    def predict(self, cross_covariance: Union[OperatorLike, Sequence[OperatorLike]]):
        """
        BLUPs for individuals that were not fitted (Lee et al. 2026, eq. 22):

            g_hat_new = sum_i theta_i V_i[new, fitted] P y.

        :param cross_covariance: One ``N_new x N`` operator per genetic
            component (or a single operator if there is only one).
        """
        if not isinstance(cross_covariance, (list, tuple)):
            cross_covariance = [cross_covariance]
        if len(cross_covariance) != len(self._genetic_operators):
            raise ValueError(
                f"Expected {len(self._genetic_operators)} cross-covariance operators, "
                f"got {len(cross_covariance)}"
            )
        weights = self.variance_components[: len(self._genetic_operators)]
        total = None
        for weight, operator in zip(weights, cross_covariance):
            operator = aslinearoperator(operator)
            if operator.shape[1] != self._state.size:
                raise ValueError(
                    f"cross-covariance operator has {operator.shape[1]} columns, "
                    f"expected {self._state.size}"
                )
            contribution = weight * numpy.asarray(operator.matvec(self.projection))
            total = contribution if total is None else total + contribution
        return total

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "AI-REML fit",
            f"  converged            : {self.converged} "
            f"({self.num_iterations} iterations, {self.elapsed:.2f}s)",
            f"  V-matvecs / solves   : {self.num_matvecs} / {self.num_solves}",
        ]
        for index, (value, stderr) in enumerate(
            zip(self.variance_components, self.standard_errors)
        ):
            label = (
                "residual"
                if index == self.variance_components.shape[0] - 1
                else f"component {index}"
            )
            lines.append(f"  {label:<21}: {value:.6g} (se {stderr:.3g})")
        lines.append(
            f"  {'heritability':<21}: {self.heritability:.6g} "
            f"(se {self.heritability_stderr:.3g})"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"REMLResult(variance_components={numpy.array2string(self.variance_components, precision=5)}, "
            f"heritability={self.heritability:.5g}, converged={self.converged})"
        )


def fit_reml(
    y: numpy.ndarray,
    relatedness: Union[OperatorLike, Sequence[OperatorLike]],
    covariates: Optional[numpy.ndarray] = None,
    add_intercept: bool = True,
    initial_values: Optional[Coefficients] = None,
    max_iterations: int = 100,
    tolerance: float = 0.05,
    extra_iterations: int = 15,
    num_trace_vectors: int = 50,
    trace_method: str = "xtrace",
    solver: Union[str, CovarianceSolver] = "cg",
    cg_tol: float = 1e-5,
    cg_maxiter: Optional[int] = None,
    preconditioner_rank: int = 100,
    min_variance: Optional[float] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, numpy.ndarray, numpy.ndarray], None]] = None,
    verbose: bool = False,
) -> REMLResult:
    """
    Estimate variance components of ``y ~ Normal(X b, sum_i theta_i V_i)``.

    :param y: Phenotype vector of length ``N``.
    :param relatedness: One ``N x N`` symmetric positive semi-definite
        :class:`~scipy.sparse.linalg.LinearOperator` per genetic variance
        component (or a list of them).  A residual ``sigma^2 I`` component is
        appended automatically.
    :param covariates: Optional ``N x C`` fixed-effect design matrix.
    :param add_intercept: Prepend a column of ones to ``covariates``.
    :param initial_values: Starting variance components.  Defaults to the
        randomized Haseman-Elston estimates.
    :param max_iterations: Hard cap on AI-REML iterations.
    :param tolerance: Relative change in the estimates that counts as
        converged (0.05 in Lee et al. 2026).
    :param extra_iterations: Iterations to run after the convergence criterion
        is first met; the returned estimate is their average, which averages
        out the noise in the stochastic gradient (15 in Lee et al. 2026).
    :param num_trace_vectors: Random test vectors per trace estimate.
    :param trace_method: ``"xtrace"``, ``"hutchinson"`` or ``"exact"``.
    :param solver: ``"cg"``, ``"dense"``, or a
        :class:`~aireml.solvers.CovarianceSolver` instance.
    :param cg_tol: Relative residual tolerance of the CG solves.
    :param cg_maxiter: Iteration cap per CG solve.
    :param preconditioner_rank: Rank of the randomized Nystrom preconditioner
        (0 disables it).  The sketch costs this many products per component,
        once, and is reused by every solve; it pays off when the relatedness
        spectrum is dominated by a few large eigenvalues (population
        structure), and needs to comfortably exceed the number of such
        eigenvalues to help.  Lee et al. (2026) use 500 at biobank scale.
    :param min_variance: Lower bound on each variance component.  Defaults to
        ``1e-6 * var(y)``.
    :param seed: Seed for the randomized linear algebra.
    :param callback: Called as ``callback(iteration, theta, gradient)``.
    :param verbose: Print per-iteration diagnostics.
    """
    started = time.perf_counter()
    rng = numpy.random.default_rng(seed)

    y = numpy.asarray(y, dtype=numpy.float64).ravel()
    size = y.shape[0]
    if not numpy.all(numpy.isfinite(y)):
        raise ValueError("y contains non-finite values")

    genetic = _as_operator_list(relatedness)
    for operator in genetic:
        if operator.shape != (size, size):
            raise ValueError(
                f"relatedness operator has shape {operator.shape}, expected "
                f"{(size, size)}"
            )
    operators = genetic + [identity_operator(size)]
    num_components = len(operators)

    design: numpy.ndarray
    if covariates is None:
        design = numpy.ones((size, 1)) if add_intercept else numpy.empty((size, 0))
    else:
        design = numpy.asarray(covariates, dtype=numpy.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        if add_intercept:
            design = numpy.column_stack([numpy.ones(size), design])
    if design.shape[1] == 0:
        raise ValueError("Need at least one covariate; set add_intercept=True")

    if isinstance(solver, CovarianceSolver):
        covariance_solver = solver
    elif solver == "dense":
        covariance_solver = DenseSolver(operators)
    elif solver == "cg":
        covariance_solver = ConjugateGradientSolver(
            operators,
            tol=cg_tol,
            maxiter=cg_maxiter,
            preconditioner_rank=preconditioner_rank,
            ridge_index=-1,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    state = RemlState(y, design, covariance_solver)

    if initial_values is None:
        theta = haseman_elston(
            y,
            operators,
            design,
            num_vectors=num_trace_vectors,
            rng=rng,
            exact=(trace_method == "exact"),
        )
    else:
        theta = numpy.asarray(initial_values, dtype=numpy.float64)
        if theta.shape != (num_components,):
            raise ValueError(
                f"initial_values must have length {num_components} "
                "(one per component, residual last)"
            )
    initial = theta.copy()

    if min_variance is None:
        min_variance = 1e-6 * max(float(numpy.var(y)), numpy.finfo(float).tiny)
    theta = numpy.maximum(theta, min_variance)

    history: List[numpy.ndarray] = []
    gradient = numpy.zeros(num_components)
    information = numpy.eye(num_components)
    converged = False
    since_converged = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        state.update(theta)
        traces = numpy.array(
            [
                estimate_trace(
                    state.projected_component_operator(index),
                    num_vectors=num_trace_vectors,
                    method=trace_method,
                    rng=rng,
                ).value
                for index in range(num_components)
            ]
        )
        gradient = traces - state.quadratic_terms()
        information = state.average_information()

        try:
            step = numpy.linalg.solve(information, gradient)
        except numpy.linalg.LinAlgError:  # pragma: no cover - singular AI
            step = numpy.linalg.lstsq(information, gradient, rcond=None)[0]

        scale = 1.0
        candidate = theta - step
        while numpy.any(candidate < min_variance) and scale > 1e-4:
            scale *= 0.5
            candidate = theta - scale * step
        candidate = numpy.maximum(candidate, min_variance)

        relative = numpy.max(
            numpy.abs(candidate - theta) / numpy.maximum(numpy.abs(theta), min_variance)
        )
        theta = candidate
        history.append(theta.copy())
        if callback is not None:
            callback(iteration, theta.copy(), gradient.copy())
        if verbose:
            values = " ".join(f"{value:.6g}" for value in theta)
            print(
                f"[aireml] iter {iteration:3d}  theta = {values}  "
                f"rel.change = {relative:.3g}  |grad| = "
                f"{numpy.linalg.norm(gradient):.3g}"
            )

        if converged:
            since_converged += 1
            if since_converged >= extra_iterations:
                break
        elif relative < tolerance:
            converged = True
            if extra_iterations == 0:
                break

    if converged and extra_iterations > 0:
        window = history[-extra_iterations:]
        theta = numpy.mean(numpy.array(window), axis=0)

    state.update(theta)
    # Recompute the curvature at the returned estimate so the reported
    # standard errors correspond to it (cheap: one solve per component).
    information = state.average_information()
    return REMLResult(
        variance_components=theta,
        state=state,
        genetic_operators=genetic,
        gradient=gradient,
        average_information=information,
        history=history,
        converged=converged,
        num_iterations=iteration,
        initial_values=initial,
        elapsed=time.perf_counter() - started,
    )
