"""
The quantities AI-REML needs at a given value of the variance components.

Everything here is phrased in terms of the projection matrix

    P = V^-1 - V^-1 X (X^T V^-1 X)^-1 X^T V^-1,

which is never formed.  Multiplying by ``P`` costs one solve against ``V``
plus dense work that is linear in ``N`` (because ``X`` is ``N x C`` with
``C`` small), exactly as in Lee et al. (2026).
"""

from typing import List, Optional

import numpy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

from .operators import Coefficients
from .solvers import CovarianceSolver, DenseSolver

__all__ = ["RemlState"]


class RemlState:
    """
    Cached per-iteration state: ``V^-1 X``, ``b_hat``, ``P y``.

    :param y: Phenotype vector, length ``N``.
    :param covariates: ``N x C`` fixed-effect design matrix.
    :param solver: Solver against ``V``.
    """

    def __init__(
        self,
        y: numpy.ndarray,
        covariates: numpy.ndarray,
        solver: CovarianceSolver,
    ):
        self.y = numpy.asarray(y, dtype=numpy.float64).ravel()
        self.covariates = numpy.asarray(covariates, dtype=numpy.float64)
        if self.covariates.ndim == 1:
            self.covariates = self.covariates.reshape(-1, 1)
        self.solver = solver
        self.size = self.y.shape[0]
        self.num_covariates = self.covariates.shape[1]
        if self.covariates.shape[0] != self.size:
            raise ValueError(
                f"covariates has {self.covariates.shape[0]} rows, y has {self.size}"
            )
        if solver.size != self.size:
            raise ValueError(
                f"operators are {solver.size} x {solver.size}, y has length {self.size}"
            )
        self.num_components = solver.num_components

        self.coefficients: Optional[numpy.ndarray] = None
        self.projected_y: Optional[numpy.ndarray] = None
        self.fixed_effects: Optional[numpy.ndarray] = None
        self._solved_covariates: Optional[numpy.ndarray] = None
        self._information_factor: Optional[tuple] = None

    # ------------------------------------------------------------------
    # state update
    # ------------------------------------------------------------------
    def update(self, coefficients: Coefficients) -> None:
        """Move to a new value of the variance components."""
        self.coefficients = numpy.asarray(coefficients, dtype=numpy.float64)
        self.solver.update(self.coefficients)
        self._solved_covariates = self.solver.solve(self.covariates)
        information = self.covariates.T @ self._solved_covariates
        information = 0.5 * (information + information.T)
        self._information_factor = cho_factor(information, lower=True)
        solved_y = self.solver.solve(self.y)
        self.fixed_effects = cho_solve(
            self._information_factor, self.covariates.T @ solved_y
        )
        self.projected_y = solved_y - self._solved_covariates @ self.fixed_effects

    def _require_update(self) -> None:
        if self.coefficients is None:
            raise RuntimeError("update() must be called first")

    # ------------------------------------------------------------------
    # products with P
    # ------------------------------------------------------------------
    def apply_projection(self, block: numpy.ndarray) -> numpy.ndarray:
        """``P @ block`` for a vector or a block of vectors."""
        self._require_update()
        array = numpy.asarray(block, dtype=numpy.float64)
        vector_input = array.ndim == 1
        matrix = array.reshape(-1, 1) if vector_input else array
        solved = self.solver.solve(matrix)
        assert self._solved_covariates is not None
        correction = self._solved_covariates @ cho_solve(
            self._information_factor, self.covariates.T @ solved
        )
        result = solved - correction
        return result.ravel() if vector_input else result

    def projected_component_operator(self, index: int) -> LinearOperator:
        """``P V_index`` as a (non-symmetric) :class:`LinearOperator`."""
        self._require_update()

        def _matmat(other: numpy.ndarray) -> numpy.ndarray:
            return self.apply_projection(self.solver.component_matmat(index, other))

        return LinearOperator(
            shape=(self.size, self.size),
            matvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
            matmat=_matmat,
            dtype=numpy.float64,
        )

    # ------------------------------------------------------------------
    # gradient and average information
    # ------------------------------------------------------------------
    def quadratic_terms(self) -> numpy.ndarray:
        """``y^T P V_i P y`` for each component ``i``."""
        self._require_update()
        assert self.projected_y is not None
        return numpy.array(
            [
                float(
                    self.projected_y
                    @ self.solver.component_matmat(index, self.projected_y)
                )
                for index in range(self.num_components)
            ]
        )

    def average_information(self) -> numpy.ndarray:
        """
        The average information matrix ``AI_ij = y^T P V_i P V_j P y``.

        This is the average of the observed Hessian and the Fisher information
        of the REML objective; the expensive trace terms present in each of
        them cancel (Gilmour et al. 1995; Lee et al. 2026, equation 14).
        Costs one solve per component.
        """
        self._require_update()
        assert self.projected_y is not None
        weighted: List[numpy.ndarray] = [
            self.solver.component_matmat(index, self.projected_y)
            for index in range(self.num_components)
        ]
        stacked = numpy.column_stack(weighted)
        projected = self.apply_projection(stacked)
        information = stacked.T @ projected
        return 0.5 * (information + information.T)

    # ------------------------------------------------------------------
    # dense-only diagnostics
    # ------------------------------------------------------------------
    def restricted_objective(self) -> float:
        """
        ``-2`` times the restricted log likelihood, up to an additive constant
        (Lee et al. 2026, equation 5).

        Requires a :class:`~aireml.solvers.DenseSolver`, since it needs
        ``log det V``.
        """
        self._require_update()
        if not isinstance(self.solver, DenseSolver):
            raise TypeError("restricted_objective() requires a DenseSolver")
        covariance = self.solver.dense_covariance()
        sign, logdet = numpy.linalg.slogdet(covariance)
        if sign <= 0:
            return float("inf")
        assert self._solved_covariates is not None
        information = self.covariates.T @ self._solved_covariates
        _, logdet_information = numpy.linalg.slogdet(information)
        assert self.projected_y is not None
        return float(
            (self.size - self.num_covariates) * numpy.log(2.0 * numpy.pi)
            + logdet
            + logdet_information
            + self.y @ self.projected_y
        )
