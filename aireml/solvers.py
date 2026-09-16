"""
Linear solves against the phenotypic covariance ``V = sum_i theta_i V_i``.

Everything AI-REML needs from ``V`` is a solve ``V^-1 b``, and the whole point
of working with linear operators is that ``V`` is never formed.  The default
:class:`ConjugateGradientSolver` therefore uses preconditioned conjugate
gradients (Hestenes and Stiefel 1952) with the randomized Nystrom
preconditioner of Frangella et al. (2021).  :class:`DenseSolver` materializes
``V`` and is there for small problems and for testing.
"""

import inspect
from typing import List, Optional, Sequence

import numpy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse.linalg import LinearOperator, cg

from .operators import (
    Coefficients,
    LinearCombinationOperator,
    as_symmetric_operator,
    materialize,
)

__all__ = [
    "ConjugateGradientSolver",
    "CovarianceSolver",
    "DenseSolver",
    "NystromSketch",
]


_CG_USES_RTOL = "rtol" in inspect.signature(cg).parameters


def _cg(operator, rhs, tol, maxiter, preconditioner, x0=None, callback=None):
    kwargs = dict(maxiter=maxiter, M=preconditioner, x0=x0, callback=callback)
    if _CG_USES_RTOL:  # scipy >= 1.12
        kwargs.update(rtol=tol, atol=0.0)
    else:  # pragma: no cover - older scipy
        kwargs.update(tol=tol, atol=0.0)
    return cg(operator, rhs, **kwargs)


class NystromSketch:
    """
    Reusable randomized sketch of a set of covariance components.

    The Nystrom preconditioner for ``V = A + sigma^2 I`` needs a sketch
    ``A @ Omega``.  During AI-REML, ``A = sum_i theta_i V_i`` changes every
    iteration but the operators ``V_i`` do not, so each ``V_i @ Omega`` is
    computed once here and recombined for free as the coefficients move.
    """

    def __init__(
        self,
        operators: Sequence[LinearOperator],
        rank: int,
        rng: numpy.random.Generator,
    ):
        size = operators[0].shape[0]
        self.rank = int(max(1, min(rank, size)))
        self.test_matrix, _ = numpy.linalg.qr(rng.standard_normal((size, self.rank)))
        self.sketches = [numpy.asarray(op.matmat(self.test_matrix)) for op in operators]
        self.num_matvecs = self.rank * len(operators)

    def preconditioner(
        self, coefficients: Coefficients, ridge: float
    ) -> Optional[LinearOperator]:
        """
        Build ``P^-1`` for ``V = sum_i coefficients[i] V_i + ridge * I``.

        Returns None if the sketch is degenerate (for example when every
        coefficient is zero), in which case unpreconditioned CG is fine
        anyway.
        """
        sketch = numpy.zeros_like(self.test_matrix)
        for coefficient, component in zip(coefficients, self.sketches):
            if coefficient != 0.0:
                sketch = sketch + coefficient * component
        norm = numpy.linalg.norm(sketch)
        if not numpy.isfinite(norm) or norm == 0.0:
            return None

        shift = numpy.sqrt(self.test_matrix.shape[0]) * numpy.finfo(float).eps * norm
        shifted = sketch + shift * self.test_matrix
        core = self.test_matrix.T @ shifted
        core = 0.5 * (core + core.T)
        try:
            chol = numpy.linalg.cholesky(core)
        except numpy.linalg.LinAlgError:  # pragma: no cover - degenerate sketch
            return None
        factor = solve_triangular(chol, shifted.T, lower=True).T
        left, singular, _ = numpy.linalg.svd(factor, full_matrices=False)
        eigenvalues = numpy.maximum(singular**2 - shift, 0.0)

        smallest = eigenvalues[-1]
        scale = (smallest + ridge) / (eigenvalues + ridge)

        def _matmat(other: numpy.ndarray) -> numpy.ndarray:
            projected = left.T @ other
            return other - left @ projected + left @ (scale[:, None] * projected)

        return LinearOperator(
            shape=(self.test_matrix.shape[0],) * 2,
            matvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
            matmat=_matmat,
            rmatvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
            rmatmat=_matmat,
            dtype=numpy.float64,
        )


class CovarianceSolver:
    """
    Base class: holds the components of ``V`` and solves against it.

    Subclasses implement :meth:`_solve_columns`.  ``update`` is called once per
    AI-REML iteration with the current variance components.
    """

    def __init__(self, operators: Sequence[LinearOperator]):
        self.operators = [as_symmetric_operator(op) for op in operators]
        self.covariance = LinearCombinationOperator(self.operators)
        self.size = self.covariance.shape[0]
        self.num_components = len(self.operators)
        self.coefficients: numpy.ndarray = numpy.ones(self.num_components)
        #: Number of ``V``-matvecs consumed so far (a hardware-independent
        #: proxy for runtime).
        self.num_matvecs = 0
        #: Number of right-hand sides solved so far.
        self.num_solves = 0

    def update(self, coefficients: Coefficients) -> None:
        self.coefficients = numpy.asarray(coefficients, dtype=numpy.float64)
        self.covariance.set_coefficients(self.coefficients)

    def matvec(self, vector: numpy.ndarray) -> numpy.ndarray:
        """``V @ vector``."""
        return self.covariance.matvec(vector)

    def component_matmat(self, index: int, block: numpy.ndarray) -> numpy.ndarray:
        """``V_index @ block``."""
        block = numpy.asarray(block, dtype=numpy.float64)
        if block.ndim == 1:
            return numpy.asarray(self.operators[index].matvec(block))
        return numpy.asarray(self.operators[index].matmat(block))

    def solve(self, rhs: numpy.ndarray) -> numpy.ndarray:
        """Solve ``V x = rhs`` for one or several right-hand sides."""
        rhs = numpy.asarray(rhs, dtype=numpy.float64)
        vector_input = rhs.ndim == 1
        block = rhs.reshape(-1, 1) if vector_input else rhs
        self.num_solves += block.shape[1]
        result = self._solve_columns(block)
        return result.ravel() if vector_input else result

    def _solve_columns(self, block: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError


class DenseSolver(CovarianceSolver):
    """
    Materializes every component once and solves by Cholesky factorization.

    ``O(N^2)`` memory and ``O(N^3)`` per update, so only for small ``N``; it
    gives exact solves, which makes it the reference implementation in tests.
    """

    def __init__(self, operators: Sequence[LinearOperator]):
        super().__init__(operators)
        self.dense: List[numpy.ndarray] = [materialize(op) for op in self.operators]
        self.dense = [0.5 * (d + d.T) for d in self.dense]
        self._factor: Optional[tuple] = None

    def update(self, coefficients: Coefficients) -> None:
        super().update(coefficients)
        self._factor = cho_factor(self.dense_covariance(), lower=True)

    def dense_covariance(self) -> numpy.ndarray:
        """The materialized ``V`` for the current coefficients."""
        matrix = numpy.zeros((self.size, self.size))
        for coefficient, component in zip(self.coefficients, self.dense):
            matrix += coefficient * component
        return matrix

    def _solve_columns(self, block: numpy.ndarray) -> numpy.ndarray:
        if self._factor is None:
            raise RuntimeError("update() must be called before solve()")
        return cho_solve(self._factor, block)


class ConjugateGradientSolver(CovarianceSolver):
    """
    Matrix-free solves by preconditioned conjugate gradients.

    :param operators: Components ``V_i`` of ``V = sum_i theta_i V_i``.
    :param tol: Relative residual tolerance for CG.
    :param maxiter: Maximum CG iterations per right-hand side.
    :param preconditioner_rank: Rank of the randomized Nystrom preconditioner;
        0 disables preconditioning.
    :param ridge_index: Index of the component that is the identity (the
        residual variance).  Its coefficient is the shift ``sigma^2`` in the
        Nystrom preconditioner; the remaining components are sketched.
    :param rng: Random generator used for the sketch.
    """

    def __init__(
        self,
        operators: Sequence[LinearOperator],
        tol: float = 1e-5,
        maxiter: Optional[int] = None,
        preconditioner_rank: int = 100,
        ridge_index: Optional[int] = -1,
        rng: Optional[numpy.random.Generator] = None,
    ):
        super().__init__(operators)
        self.tol = float(tol)
        self.maxiter = maxiter
        self.ridge_index = (
            None if ridge_index is None else ridge_index % self.num_components
        )
        #: Total CG iterations across all solves.
        self.num_cg_iterations = 0
        #: Right-hand sides for which CG hit ``maxiter`` without converging.
        self.num_failures = 0

        self._sketch: Optional[NystromSketch] = None
        if preconditioner_rank > 0 and self.ridge_index is not None:
            sketched = [
                op for i, op in enumerate(self.operators) if i != self.ridge_index
            ]
            if sketched:
                rng = numpy.random.default_rng() if rng is None else rng
                self._sketch = NystromSketch(sketched, preconditioner_rank, rng)
                self.num_matvecs += self._sketch.num_matvecs
        self._preconditioner: Optional[LinearOperator] = None

    def update(self, coefficients: Coefficients) -> None:
        super().update(coefficients)
        if self._sketch is None:
            self._preconditioner = None
            return
        sketched = [c for i, c in enumerate(self.coefficients) if i != self.ridge_index]
        ridge = float(self.coefficients[self.ridge_index])
        self._preconditioner = self._sketch.preconditioner(sketched, ridge)

    def _solve_columns(self, block: numpy.ndarray) -> numpy.ndarray:
        result = numpy.empty_like(block)
        counter = {"iterations": 0}

        def _callback(_):
            counter["iterations"] += 1

        for column in range(block.shape[1]):
            counter["iterations"] = 0
            solution, info = _cg(
                self.covariance,
                block[:, column],
                tol=self.tol,
                maxiter=self.maxiter,
                preconditioner=self._preconditioner,
                x0=None,
                callback=_callback,
            )
            if info > 0:
                self.num_failures += 1
            result[:, column] = solution
            # One matvec against V per CG iteration.
            self.num_cg_iterations += counter["iterations"]
            self.num_matvecs += counter["iterations"]
        return result
