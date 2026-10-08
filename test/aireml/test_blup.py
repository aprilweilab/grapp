"""
Tests for best linear unbiased prediction of genetic values.

BLUPs reduce to ``g_hat = tau^2 B P y`` in sample and
``g_hat_new = tau^2 B[new, fitted] P y`` out of sample (Lee et al. 2026,
equation 22), so both are checked against the explicit matrix formulas as
well as against simulated truth.
"""

import unittest

import numpy
from scipy.sparse.linalg import aslinearoperator

from aireml import fit_reml


def simulate_population(
    rng, size=400, num_variants=800, num_groups=20, theta=(1.0, 0.5)
):
    # Individuals cluster into related groups so the GRM has real off-diagonal
    # structure; with unrelated individuals there is nothing to predict from.
    shared = rng.standard_normal((num_groups, num_variants))
    membership = rng.integers(0, num_groups, size=size)
    genotypes = shared[membership] + rng.standard_normal((size, num_variants))
    genotypes -= genotypes.mean(axis=0, keepdims=True)
    grm = genotypes @ genotypes.T / num_variants
    # Genetic values drawn from the LMM prior, so the BLUP is the exact
    # posterior mean under the fitted model.
    genetic = numpy.linalg.cholesky(
        theta[0] * grm + 1e-8 * numpy.eye(size)
    ) @ rng.standard_normal(size)
    phenotype = genetic + numpy.sqrt(theta[1]) * rng.standard_normal(size)
    return grm, genetic, phenotype


class TestBLUP(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(555)
        self.grm, self.genetic, self.phenotype = simulate_population(self.rng)
        self.size = self.phenotype.shape[0]
        self.fitted = numpy.arange(self.size // 2)
        self.held_out = numpy.arange(self.size // 2, self.size)

    def _fit_on_subset(self):
        subset = numpy.ix_(self.fitted, self.fitted)
        return fit_reml(
            self.phenotype[self.fitted],
            self.grm[subset],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )

    def test_in_sample_blup_matches_dense_formula(self):
        result = self._fit_on_subset()
        subset = numpy.ix_(self.fitted, self.fitted)
        grm = self.grm[subset]
        tau, sigma = result.variance_components
        covariance = tau * grm + sigma * numpy.eye(len(self.fitted))
        design = numpy.ones((len(self.fitted), 1))
        residual = self.phenotype[self.fitted] - design @ result.fixed_effects
        expected = tau * grm @ numpy.linalg.solve(covariance, residual)
        numpy.testing.assert_allclose(result.blup(), expected, atol=1e-8)

    def test_out_of_sample_blup_matches_dense_formula(self):
        result = self._fit_on_subset()
        cross = self.grm[numpy.ix_(self.held_out, self.fitted)]
        tau, sigma = result.variance_components
        grm = self.grm[numpy.ix_(self.fitted, self.fitted)]
        covariance = tau * grm + sigma * numpy.eye(len(self.fitted))
        design = numpy.ones((len(self.fitted), 1))
        residual = self.phenotype[self.fitted] - design @ result.fixed_effects
        expected = tau * cross @ numpy.linalg.solve(covariance, residual)
        numpy.testing.assert_allclose(result.predict(cross), expected, atol=1e-8)

    def test_prediction_accepts_a_linear_operator(self):
        result = self._fit_on_subset()
        cross = self.grm[numpy.ix_(self.held_out, self.fitted)]
        numpy.testing.assert_allclose(
            result.predict(aslinearoperator(cross)), result.predict(cross), atol=1e-12
        )

    def test_predictions_correlate_with_true_genetic_values(self):
        result = self._fit_on_subset()
        cross = self.grm[numpy.ix_(self.held_out, self.fitted)]
        predicted = result.predict(cross)
        correlation = numpy.corrcoef(predicted, self.genetic[self.held_out])[0, 1]
        self.assertGreater(correlation, 0.4)

    def test_in_sample_blup_correlates_with_truth(self):
        result = self._fit_on_subset()
        correlation = numpy.corrcoef(result.blup(), self.genetic[self.fitted])[0, 1]
        self.assertGreater(correlation, 0.75)

    def test_predict_validates_shapes(self):
        result = self._fit_on_subset()
        with self.assertRaises(ValueError):
            result.predict([self.grm, self.grm])
        with self.assertRaises(ValueError):
            result.predict(self.grm[numpy.ix_(self.held_out, self.held_out[:3])])

    def test_multi_component_blup_sums_contributions(self):
        rng = numpy.random.default_rng(88)
        size = 120
        first = rng.standard_normal((size, 300))
        first = first @ first.T / 300.0
        second = rng.standard_normal((size, 300))
        second = second @ second.T / 300.0
        covariance = 0.5 * first + 0.9 * second + 0.4 * numpy.eye(size)
        y = numpy.linalg.cholesky(covariance) @ rng.standard_normal(size)
        result = fit_reml(
            y,
            [first, second],
            solver="dense",
            trace_method="exact",
            tolerance=1e-8,
            extra_iterations=0,
            seed=0,
        )
        expected = (
            result.variance_components[0] * first @ result.projection
            + result.variance_components[1] * second @ result.projection
        )
        numpy.testing.assert_allclose(result.blup(), expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
