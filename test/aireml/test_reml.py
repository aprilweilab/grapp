"""
Accuracy tests for the AI-REML driver.

The reference point throughout is a direct numerical maximization of the REML
objective on a materialized covariance matrix: AI-REML is just a fast way of
finding that same optimum, so with exact traces and exact solves the two must
agree to many digits.
"""

import unittest

import numpy
from scipy.optimize import minimize

from aireml import fit_reml, haseman_elston
from aireml.model import RemlState
from aireml.operators import (
    as_symmetric_operator,
    grm_from_genotypes,
    identity_operator,
)
from aireml.solvers import DenseSolver


def simulate(rng, size=200, num_variants=500, theta=(0.8, 0.5), effects=(2.0, -1.0)):
    """A GRM, a covariate design and a phenotype drawn from the LMM."""
    genotypes = rng.standard_normal((size, num_variants))
    grm = genotypes @ genotypes.T / num_variants
    covariates = numpy.column_stack([numpy.ones(size), rng.standard_normal(size)])
    covariance = theta[0] * grm + theta[1] * numpy.eye(size)
    phenotype = covariates @ numpy.array(effects) + numpy.linalg.cholesky(
        covariance
    ) @ rng.standard_normal(size)
    return genotypes, grm, covariates, phenotype


def reference_optimum(y, components, covariates):
    """Maximize the REML likelihood by brute force, on log-variances."""
    operators = [as_symmetric_operator(component) for component in components]
    state = RemlState(y, covariates, DenseSolver(operators))

    def objective(log_theta):
        state.update(numpy.exp(log_theta))
        return state.restricted_objective()

    start = numpy.log(numpy.full(len(components), numpy.var(y) / len(components)))
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000, maxfev=20000),
    )
    return numpy.exp(result.x)


class TestAIREML(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(20260327)
        self.genotypes, self.grm, self.covariates, self.y = simulate(self.rng)
        self.size = self.y.shape[0]

    def test_matches_brute_force_reml_optimum(self):
        reference = reference_optimum(
            self.y, [self.grm, numpy.eye(self.size)], self.covariates
        )
        result = fit_reml(
            self.y,
            self.grm,
            covariates=self.covariates[:, 1:],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            max_iterations=100,
            seed=0,
        )
        self.assertTrue(result.converged)
        numpy.testing.assert_allclose(result.variance_components, reference, rtol=1e-5)

    def test_converges_from_a_bad_start(self):
        reference = reference_optimum(
            self.y, [self.grm, numpy.eye(self.size)], self.covariates
        )
        result = fit_reml(
            self.y,
            self.grm,
            covariates=self.covariates[:, 1:],
            initial_values=[50.0, 0.01],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            max_iterations=200,
            seed=0,
        )
        numpy.testing.assert_allclose(result.variance_components, reference, rtol=1e-4)

    def test_gradient_vanishes_at_the_optimum(self):
        result = fit_reml(
            self.y,
            self.grm,
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            max_iterations=100,
            seed=0,
        )
        scale = numpy.abs(result.average_information).max()
        self.assertLess(numpy.linalg.norm(result.gradient) / scale, 1e-4)

    def test_matrix_free_matches_dense(self):
        # The whole point: identical answers without ever forming the GRM.
        dense = fit_reml(
            self.y,
            self.grm,
            covariates=self.covariates[:, 1:],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )
        operator = grm_from_genotypes(self.genotypes)
        matrix_free = fit_reml(
            self.y,
            operator,
            covariates=self.covariates[:, 1:],
            solver="cg",
            trace_method="xtrace",
            num_trace_vectors=40,
            cg_tol=1e-8,
            preconditioner_rank=40,
            seed=0,
        )
        numpy.testing.assert_allclose(
            matrix_free.variance_components,
            dense.variance_components,
            rtol=0.06,
        )
        self.assertTrue(matrix_free.converged)
        self.assertGreater(matrix_free.num_solves, 0)

    def test_fixed_effects_are_recovered(self):
        result = fit_reml(
            self.y,
            self.grm,
            covariates=self.covariates[:, 1:],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )
        numpy.testing.assert_allclose(
            result.fixed_effects, numpy.array([2.0, -1.0]), atol=0.35
        )

    def test_estimates_are_close_to_truth_on_average(self):
        # REML is unbiased for the variance components under the model; over
        # replicates the mean estimate should sit on the simulated value.
        truth = numpy.array([0.8, 0.5])
        estimates = []
        for seed in range(30):
            rng = numpy.random.default_rng(1000 + seed)
            _, grm, covariates, y = simulate(rng, size=150, theta=tuple(truth))
            result = fit_reml(
                y,
                grm,
                covariates=covariates[:, 1:],
                solver="dense",
                trace_method="exact",
                tolerance=1e-8,
                extra_iterations=0,
                seed=seed,
            )
            estimates.append(result.variance_components)
        estimates = numpy.array(estimates)
        standard_error = estimates.std(axis=0, ddof=1) / numpy.sqrt(len(estimates))
        numpy.testing.assert_array_less(
            numpy.abs(estimates.mean(axis=0) - truth), 4.0 * standard_error
        )

    def test_reported_standard_errors_track_the_empirical_spread(self):
        truth = (0.8, 0.5)
        estimates, reported = [], []
        for seed in range(30):
            rng = numpy.random.default_rng(2000 + seed)
            _, grm, covariates, y = simulate(rng, size=150, theta=truth)
            result = fit_reml(
                y,
                grm,
                covariates=covariates[:, 1:],
                solver="dense",
                trace_method="exact",
                tolerance=1e-8,
                extra_iterations=0,
                seed=seed,
            )
            estimates.append(result.variance_components)
            reported.append(result.standard_errors)
        empirical = numpy.array(estimates).std(axis=0, ddof=1)
        average = numpy.array(reported).mean(axis=0)
        # Asymptotic standard errors, so agreement is approximate.
        numpy.testing.assert_allclose(average, empirical, rtol=0.4)

    def test_multiple_variance_components(self):
        rng = numpy.random.default_rng(7)
        size = 150
        first = rng.standard_normal((size, 400))
        first = first @ first.T / 400.0
        second = rng.standard_normal((size, 400))
        second = second @ second.T / 400.0
        truth = numpy.array([0.6, 0.9, 0.5])
        covariance = truth[0] * first + truth[1] * second + truth[2] * numpy.eye(size)
        y = numpy.linalg.cholesky(covariance) @ rng.standard_normal(size)

        reference = reference_optimum(
            y, [first, second, numpy.eye(size)], numpy.ones((size, 1))
        )
        result = fit_reml(
            y,
            [first, second],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            max_iterations=200,
            seed=0,
        )
        self.assertEqual(result.variance_components.shape, (3,))
        numpy.testing.assert_allclose(result.variance_components, reference, rtol=1e-3)
        self.assertAlmostEqual(
            result.genetic_variance,
            float(result.variance_components[:2].sum()),
            places=12,
        )

    def test_zero_heritability_is_driven_to_the_floor(self):
        rng = numpy.random.default_rng(4242)
        _, grm, covariates, _ = simulate(rng, size=150)
        y = rng.standard_normal(150)
        result = fit_reml(
            y,
            grm,
            covariates=covariates[:, 1:],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )
        self.assertLess(result.heritability, 0.2)
        self.assertGreater(result.variance_components[0], 0.0)

    def test_averaging_window_is_applied(self):
        result = fit_reml(
            self.y,
            self.grm,
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=5,
            seed=0,
        )
        expected = numpy.mean(numpy.array(result.history[-5:]), axis=0)
        numpy.testing.assert_allclose(result.variance_components, expected)

    def test_haseman_elston_is_in_the_right_ballpark(self):
        operators = [as_symmetric_operator(self.grm), identity_operator(self.size)]
        exact = haseman_elston(self.y, operators, self.covariates, exact=True)
        randomized = haseman_elston(
            self.y,
            operators,
            self.covariates,
            num_vectors=100,
            rng=numpy.random.default_rng(0),
        )
        numpy.testing.assert_allclose(randomized, exact, rtol=0.35)
        # HE is a moment estimator: a usable starting point, not the answer.
        numpy.testing.assert_allclose(exact, numpy.array([0.8, 0.5]), atol=0.6)

    def test_callback_and_history(self):
        seen = []
        result = fit_reml(
            self.y,
            self.grm,
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            callback=lambda i, theta, grad: seen.append((i, theta.copy())),
            seed=0,
        )
        self.assertEqual(len(seen), result.num_iterations)
        self.assertEqual(len(result.history), result.num_iterations)
        numpy.testing.assert_allclose(result.history[-1], result.variance_components)

    def test_summary_and_repr(self):
        result = fit_reml(
            self.y,
            self.grm,
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )
        self.assertIn("heritability", result.summary())
        self.assertIn("REMLResult", repr(result))
        self.assertTrue(0.0 <= result.heritability <= 1.0)
        self.assertGreater(result.heritability_stderr, 0.0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            fit_reml(self.y[:-1], self.grm, solver="dense")
        with self.assertRaises(ValueError):
            fit_reml(self.y, self.grm, solver="nope")
        with self.assertRaises(ValueError):
            fit_reml(self.y, self.grm, initial_values=[1.0], solver="dense")
        with self.assertRaises(ValueError):
            fit_reml(self.y, self.grm, add_intercept=False, solver="dense")
        bad = self.y.copy()
        bad[0] = numpy.nan
        with self.assertRaises(ValueError):
            fit_reml(bad, self.grm, solver="dense")


if __name__ == "__main__":
    unittest.main()
