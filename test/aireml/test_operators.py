"""Tests for the LinearOperator helpers used by aireml."""

import unittest

import numpy
from scipy.sparse.linalg import aslinearoperator

from aireml.operators import (
    as_symmetric_operator,
    centered_operator,
    grm_from_genotypes,
    identity_operator,
    LinearCombinationOperator,
    materialize,
)

TOLERANCE = 1e-10


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(1234)
        self.num_samples = 40
        self.num_variants = 90
        self.genotypes = self.rng.integers(
            0, 3, size=(self.num_samples, self.num_variants)
        ).astype(numpy.float64)

    def test_identity(self):
        operator = identity_operator(7)
        block = self.rng.standard_normal((7, 3))
        numpy.testing.assert_allclose(operator.matmat(block), block)
        numpy.testing.assert_allclose(materialize(operator), numpy.eye(7))

    def test_grm_from_genotypes_matches_dense(self):
        operator = grm_from_genotypes(self.genotypes)
        expected = self.genotypes @ self.genotypes.T / self.num_variants
        numpy.testing.assert_allclose(materialize(operator), expected, atol=TOLERANCE)

    def test_grm_from_genotypes_custom_scale(self):
        operator = grm_from_genotypes(self.genotypes, scale=2.0)
        expected = self.genotypes @ self.genotypes.T / 2.0
        numpy.testing.assert_allclose(materialize(operator), expected, atol=TOLERANCE)

    def test_grm_from_genotypes_centered(self):
        operator = grm_from_genotypes(self.genotypes, center=True)
        centered = self.genotypes - self.genotypes.mean(axis=0, keepdims=True)
        expected = centered @ centered.T / self.num_variants
        numpy.testing.assert_allclose(materialize(operator), expected, atol=TOLERANCE)

    def test_grm_from_genotypes_accepts_operator(self):
        # The point of the abstraction: only matvecs are required.
        wrapped = aslinearoperator(self.genotypes)
        from_operator = materialize(grm_from_genotypes(wrapped))
        from_array = materialize(grm_from_genotypes(self.genotypes))
        numpy.testing.assert_allclose(from_operator, from_array, atol=TOLERANCE)

    def test_grm_rejects_bad_scale(self):
        with self.assertRaises(ValueError):
            grm_from_genotypes(self.genotypes, scale=0.0)

    def test_linear_combination(self):
        first = self.rng.standard_normal((6, 6))
        first = first @ first.T
        second = numpy.eye(6)
        combination = LinearCombinationOperator([first, second], [2.0, 3.0])
        numpy.testing.assert_allclose(
            materialize(combination), 2.0 * first + 3.0 * second, atol=TOLERANCE
        )
        combination.set_coefficients([0.0, 1.5])
        numpy.testing.assert_allclose(
            materialize(combination), 1.5 * second, atol=TOLERANCE
        )

    def test_linear_combination_validates(self):
        with self.assertRaises(ValueError):
            LinearCombinationOperator([])
        with self.assertRaises(ValueError):
            LinearCombinationOperator([numpy.eye(3), numpy.eye(4)])
        with self.assertRaises(ValueError):
            LinearCombinationOperator([numpy.eye(3)], [1.0, 2.0])

    def test_centered_operator(self):
        matrix = self.rng.standard_normal((9, 9))
        matrix = matrix @ matrix.T
        size = matrix.shape[0]
        projector = numpy.eye(size) - numpy.ones((size, size)) / size
        numpy.testing.assert_allclose(
            materialize(centered_operator(matrix)),
            projector @ matrix @ projector,
            atol=1e-9,
        )

    def test_as_symmetric_operator_casts_dtype(self):
        matrix = numpy.eye(5, dtype=numpy.float32)
        operator = as_symmetric_operator(matrix)
        self.assertEqual(operator.dtype, numpy.float64)
        numpy.testing.assert_allclose(materialize(operator), numpy.eye(5))

    def test_as_symmetric_operator_rejects_rectangular(self):
        with self.assertRaises(ValueError):
            as_symmetric_operator(numpy.ones((3, 4)))


if __name__ == "__main__":
    unittest.main()
