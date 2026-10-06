"""Tests for the stochastic trace estimators."""

import unittest

import numpy
from scipy.sparse.linalg import aslinearoperator

from aireml.trace import estimate_trace, rademacher


def _spectrum_matrix(size, exponent, rng):
    """Symmetric matrix with eigenvalues ``k ** -exponent``."""
    basis, _ = numpy.linalg.qr(rng.standard_normal((size, size)))
    eigenvalues = numpy.arange(1, size + 1, dtype=numpy.float64) ** -exponent
    return basis @ numpy.diag(eigenvalues) @ basis.T


class TestTraceEstimators(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(99)
        self.size = 200
        self.matrix = _spectrum_matrix(self.size, 1.5, self.rng)
        self.exact = numpy.trace(self.matrix)

    def test_rademacher_is_plus_minus_one(self):
        sample = rademacher(50, 20, self.rng)
        self.assertEqual(sample.shape, (50, 20))
        self.assertTrue(numpy.all(numpy.abs(sample) == 1.0))

    def test_exact_method(self):
        estimate = estimate_trace(aslinearoperator(self.matrix), method="exact")
        self.assertAlmostEqual(estimate.value, self.exact, places=10)
        self.assertEqual(estimate.stderr, 0.0)

    def test_estimators_are_unbiased(self):
        operator = aslinearoperator(self.matrix)
        for method in ("hutchinson", "hutchpp", "xtrace"):
            values = numpy.array(
                [
                    estimate_trace(
                        operator,
                        num_vectors=20,
                        method=method,
                        rng=numpy.random.default_rng(seed),
                    ).value
                    for seed in range(200)
                ]
            )
            # Bias should be well inside the Monte-Carlo error of the mean.
            standard_error = values.std(ddof=1) / numpy.sqrt(values.size)
            self.assertLess(
                abs(values.mean() - self.exact),
                4.0 * standard_error,
                msg=f"{method} appears biased",
            )

    def test_variance_ranking_at_equal_matvec_budget(self):
        """
        Every product against ``P V_i`` is a CG solve, so the estimators have
        to be compared at equal matrix-vector count, not at equal
        ``num_vectors``: XTrace spends two products per test vector while
        Hutchinson and Hutch++ spend one.
        """
        operator = aslinearoperator(self.matrix)
        budget = 60
        spread = {}
        for method, num_vectors in (
            ("hutchinson", budget),
            ("hutchpp", budget),
            ("xtrace", budget // 2),
        ):
            values = numpy.array(
                [
                    estimate_trace(
                        operator,
                        num_vectors=num_vectors,
                        method=method,
                        rng=numpy.random.default_rng(seed),
                    ).value
                    for seed in range(120)
                ]
            )
            spread[method] = values.std(ddof=1)
        # Both sketch-and-correct estimators crush plain Hutchinson, and
        # XTrace edges out Hutch++ for the same number of products.
        self.assertLess(spread["hutchpp"], 0.2 * spread["hutchinson"])
        self.assertLess(spread["xtrace"], 0.2 * spread["hutchinson"])
        self.assertLess(spread["xtrace"], spread["hutchpp"])

    def test_hutchpp_spends_its_budget_in_three_equal_parts(self):
        operator = aslinearoperator(self.matrix)
        for budget in (9, 30, 61):
            estimate = estimate_trace(operator, budget, "hutchpp", self.rng)
            self.assertEqual(estimate.num_matvecs, 3 * (budget // 3))
            self.assertLessEqual(estimate.num_matvecs, budget)

    def test_hutchpp_requires_a_minimum_budget(self):
        with self.assertRaises(ValueError):
            estimate_trace(aslinearoperator(self.matrix), 2, "hutchpp", self.rng)

    def test_hutchpp_matches_an_independent_implementation(self):
        """
        Cross-check against pylops' trace_hutchpp, which implements the same
        Algorithm 1.  Skipped when pylops is not installed: it is a
        verification aid, not a dependency of this package.
        """
        try:
            from pylops import MatrixMult
            from pylops.utils.estimators import trace_hutchpp
        except ImportError:  # pragma: no cover - optional cross-check
            self.skipTest("pylops not installed")
        operator = aslinearoperator(self.matrix)
        mine = numpy.array(
            [
                estimate_trace(
                    operator, 60, "hutchpp", numpy.random.default_rng(seed)
                ).value
                for seed in range(120)
            ]
        )
        theirs = numpy.array(
            [
                trace_hutchpp(MatrixMult(self.matrix), neval=60, sampler="rademacher")
                for _ in range(120)
            ]
        )
        # Same estimator, so the sampling distributions must agree: both
        # unbiased, and comparable spread.
        pooled = numpy.sqrt(
            mine.var(ddof=1) / mine.size + theirs.var(ddof=1) / theirs.size
        )
        self.assertLess(abs(mine.mean() - theirs.mean()), 4.0 * pooled)
        self.assertLess(mine.std(ddof=1) / theirs.std(ddof=1), 1.5)
        self.assertGreater(mine.std(ddof=1) / theirs.std(ddof=1), 0.67)

    def test_xtrace_has_lower_variance_than_hutchinson(self):
        operator = aslinearoperator(self.matrix)
        spread = {}
        for method in ("hutchinson", "xtrace"):
            values = numpy.array(
                [
                    estimate_trace(
                        operator,
                        num_vectors=20,
                        method=method,
                        rng=numpy.random.default_rng(seed),
                    ).value
                    for seed in range(100)
                ]
            )
            spread[method] = values.std(ddof=1)
        # For a decaying spectrum XTrace is dramatically better per vector.
        self.assertLess(spread["xtrace"], 0.25 * spread["hutchinson"])

    def test_works_for_nonsymmetric_operators(self):
        # trace(P V) in AI-REML is a product of two symmetric matrices, and so
        # is not itself symmetric.
        other = _spectrum_matrix(self.size, 0.5, self.rng)
        product = self.matrix @ other
        exact = numpy.trace(product)
        for method in ("hutchpp", "xtrace"):
            values = numpy.array(
                [
                    estimate_trace(
                        aslinearoperator(product),
                        num_vectors=40 if method == "hutchpp" else 20,
                        method=method,
                        rng=numpy.random.default_rng(seed),
                    ).value
                    for seed in range(100)
                ]
            )
            standard_error = values.std(ddof=1) / numpy.sqrt(values.size)
            self.assertLess(
                abs(values.mean() - exact),
                4.0 * standard_error,
                msg=f"{method} appears biased on a non-symmetric operator",
            )

    def test_num_matvecs_reported(self):
        operator = aslinearoperator(self.matrix)
        self.assertEqual(
            estimate_trace(operator, 10, "hutchinson", self.rng).num_matvecs, 10
        )
        self.assertEqual(
            estimate_trace(operator, 10, "xtrace", self.rng).num_matvecs, 20
        )
        self.assertEqual(
            estimate_trace(operator, 9, "hutchpp", self.rng).num_matvecs, 9
        )

    def test_low_rank_operator_does_not_blow_up(self):
        # A rank-deficient sketch makes the leave-one-out downdate singular;
        # the estimator must fall back rather than divide by zero.
        low_rank = numpy.outer(numpy.ones(self.size), numpy.ones(self.size))
        estimate = estimate_trace(
            aslinearoperator(low_rank),
            num_vectors=10,
            method="xtrace",
            rng=self.rng,
        )
        self.assertTrue(numpy.isfinite(estimate.value))

    def test_rejects_unknown_method_and_shape(self):
        with self.assertRaises(ValueError):
            estimate_trace(aslinearoperator(self.matrix), method="nope")
        with self.assertRaises(ValueError):
            estimate_trace(aslinearoperator(numpy.ones((3, 4))))


if __name__ == "__main__":
    unittest.main()
