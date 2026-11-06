"""
Tests of the AI-REML derivatives against dense reference implementations.

These pin down the algebra of Lee et al. (2026), equations 5, 11 and 14:
the gradient must match finite differences of the REML objective, and the
average information must match the explicit triple product
``y^T P V_i P V_j P y``.
"""

import unittest

import numpy

from aireml.model import RemlState
from aireml.operators import as_symmetric_operator, identity_operator
from aireml.solvers import DenseSolver
from aireml.trace import estimate_trace


class TestRemlState(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(31337)
        self.size = 100
        genotypes = self.rng.standard_normal((self.size, 250))
        self.grm = genotypes @ genotypes.T / 250.0
        self.covariates = numpy.column_stack(
            [numpy.ones(self.size), self.rng.standard_normal(self.size)]
        )
        self.components = [self.grm, numpy.eye(self.size)]
        truth = numpy.array([0.8, 0.5])
        covariance = truth[0] * self.grm + truth[1] * numpy.eye(self.size)
        self.y = self.covariates @ numpy.array([1.0, -0.5]) + numpy.linalg.cholesky(
            covariance
        ) @ self.rng.standard_normal(self.size)
        self.operators = [
            as_symmetric_operator(self.grm),
            identity_operator(self.size),
        ]
        self.state = RemlState(self.y, self.covariates, DenseSolver(self.operators))
        self.theta = numpy.array([0.55, 0.62])

    def _dense_projection(self, theta):
        covariance = theta[0] * self.grm + theta[1] * numpy.eye(self.size)
        inverse = numpy.linalg.inv(covariance)
        middle = numpy.linalg.inv(self.covariates.T @ inverse @ self.covariates)
        return (
            inverse - inverse @ self.covariates @ middle @ self.covariates.T @ inverse
        )

    def test_projection_of_y(self):
        self.state.update(self.theta)
        expected = self._dense_projection(self.theta) @ self.y
        numpy.testing.assert_allclose(self.state.projected_y, expected, atol=1e-10)

    def test_apply_projection_matches_dense(self):
        self.state.update(self.theta)
        projection = self._dense_projection(self.theta)
        block = self.rng.standard_normal((self.size, 3))
        numpy.testing.assert_allclose(
            self.state.apply_projection(block), projection @ block, atol=1e-10
        )
        vector = block[:, 0]
        numpy.testing.assert_allclose(
            self.state.apply_projection(vector), projection @ vector, atol=1e-10
        )

    def test_fixed_effects_are_generalized_least_squares(self):
        self.state.update(self.theta)
        covariance = self.theta[0] * self.grm + self.theta[1] * numpy.eye(self.size)
        inverse = numpy.linalg.inv(covariance)
        expected = numpy.linalg.solve(
            self.covariates.T @ inverse @ self.covariates,
            self.covariates.T @ inverse @ self.y,
        )
        numpy.testing.assert_allclose(self.state.fixed_effects, expected, atol=1e-10)

    def test_gradient_matches_finite_differences(self):
        self.state.update(self.theta)
        traces = numpy.array(
            [
                estimate_trace(
                    self.state.projected_component_operator(index), method="exact"
                ).value
                for index in range(2)
            ]
        )
        gradient = traces - self.state.quadratic_terms()

        def objective(theta):
            self.state.update(theta)
            return self.state.restricted_objective()

        step = 1e-5
        finite = numpy.empty(2)
        for index in range(2):
            offset = numpy.zeros(2)
            offset[index] = step
            finite[index] = (
                objective(self.theta + offset) - objective(self.theta - offset)
            ) / (2.0 * step)
        numpy.testing.assert_allclose(gradient, finite, rtol=1e-5)

    def test_average_information_matches_dense(self):
        self.state.update(self.theta)
        projection = self._dense_projection(self.theta)
        expected = numpy.array(
            [
                [
                    self.y
                    @ projection
                    @ self.components[i]
                    @ projection
                    @ self.components[j]
                    @ projection
                    @ self.y
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
        numpy.testing.assert_allclose(
            self.state.average_information(), expected, rtol=1e-8
        )

    def test_average_information_is_positive_definite(self):
        self.state.update(self.theta)
        eigenvalues = numpy.linalg.eigvalsh(self.state.average_information())
        self.assertTrue(numpy.all(eigenvalues > 0))

    def test_restricted_objective_matches_formula(self):
        self.state.update(self.theta)
        covariance = self.theta[0] * self.grm + self.theta[1] * numpy.eye(self.size)
        inverse = numpy.linalg.inv(covariance)
        projection = self._dense_projection(self.theta)
        expected = (
            (self.size - self.covariates.shape[1]) * numpy.log(2.0 * numpy.pi)
            + numpy.linalg.slogdet(covariance)[1]
            + numpy.linalg.slogdet(self.covariates.T @ inverse @ self.covariates)[1]
            + self.y @ projection @ self.y
        )
        self.assertAlmostEqual(self.state.restricted_objective(), expected, places=8)

    def test_objective_requires_dense_solver(self):
        from aireml.solvers import ConjugateGradientSolver

        state = RemlState(
            self.y,
            self.covariates,
            ConjugateGradientSolver(self.operators, preconditioner_rank=0),
        )
        state.update(self.theta)
        with self.assertRaises(TypeError):
            state.restricted_objective()

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            RemlState(
                self.y[:-1],
                self.covariates[:-1],
                DenseSolver(self.operators),
            )
        with self.assertRaises(ValueError):
            RemlState(self.y, self.covariates[:-1], DenseSolver(self.operators))

    def test_update_required_before_use(self):
        state = RemlState(self.y, self.covariates, DenseSolver(self.operators))
        with self.assertRaises(RuntimeError):
            state.apply_projection(self.y)


if __name__ == "__main__":
    unittest.main()
