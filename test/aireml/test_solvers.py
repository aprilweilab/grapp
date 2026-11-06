"""Tests for the covariance solvers and the Nystrom preconditioner."""

import unittest

import numpy

from aireml.operators import as_symmetric_operator, identity_operator, materialize
from aireml.solvers import ConjugateGradientSolver, DenseSolver, NystromSketch


def _spiked_matrix(size, num_spikes, rng):
    """A GRM-like matrix: a few large eigenvalues over a flat bulk."""
    basis, _ = numpy.linalg.qr(rng.standard_normal((size, size)))
    eigenvalues = numpy.concatenate(
        [numpy.linspace(200.0, 20.0, num_spikes), 0.5 * numpy.ones(size - num_spikes)]
    )
    matrix = basis @ numpy.diag(eigenvalues) @ basis.T
    return 0.5 * (matrix + matrix.T)


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.default_rng(2024)
        self.size = 120
        genotypes = self.rng.standard_normal((self.size, 300))
        self.grm = genotypes @ genotypes.T / 300.0
        self.operators = [
            as_symmetric_operator(self.grm),
            identity_operator(self.size),
        ]
        self.coefficients = [0.7, 0.3]
        self.covariance = self.coefficients[0] * self.grm + self.coefficients[
            1
        ] * numpy.eye(self.size)

    def test_dense_solver_matches_numpy(self):
        solver = DenseSolver(self.operators)
        solver.update(self.coefficients)
        rhs = self.rng.standard_normal((self.size, 4))
        numpy.testing.assert_allclose(
            solver.solve(rhs), numpy.linalg.solve(self.covariance, rhs), atol=1e-9
        )
        numpy.testing.assert_allclose(
            solver.dense_covariance(), self.covariance, atol=1e-9
        )

    def test_dense_solver_requires_update(self):
        solver = DenseSolver(self.operators)
        with self.assertRaises(RuntimeError):
            solver.solve(numpy.ones(self.size))

    def test_solver_handles_vector_and_block(self):
        solver = DenseSolver(self.operators)
        solver.update(self.coefficients)
        vector = self.rng.standard_normal(self.size)
        self.assertEqual(solver.solve(vector).shape, (self.size,))
        self.assertEqual(solver.solve(vector.reshape(-1, 1)).shape, (self.size, 1))

    def test_cg_matches_dense(self):
        solver = ConjugateGradientSolver(
            self.operators,
            tol=1e-10,
            preconditioner_rank=30,
            rng=numpy.random.default_rng(0),
        )
        solver.update(self.coefficients)
        rhs = self.rng.standard_normal((self.size, 3))
        numpy.testing.assert_allclose(
            solver.solve(rhs), numpy.linalg.solve(self.covariance, rhs), atol=1e-6
        )
        self.assertEqual(solver.num_failures, 0)
        self.assertGreater(solver.num_cg_iterations, 0)

    def test_cg_matvec_matches_covariance(self):
        solver = ConjugateGradientSolver(
            self.operators, preconditioner_rank=0, rng=numpy.random.default_rng(0)
        )
        solver.update(self.coefficients)
        vector = self.rng.standard_normal(self.size)
        numpy.testing.assert_allclose(
            solver.matvec(vector), self.covariance @ vector, atol=1e-9
        )

    def test_component_matmat(self):
        solver = DenseSolver(self.operators)
        solver.update(self.coefficients)
        block = self.rng.standard_normal((self.size, 2))
        numpy.testing.assert_allclose(
            solver.component_matmat(0, block), self.grm @ block, atol=1e-9
        )
        numpy.testing.assert_allclose(
            solver.component_matmat(1, block), block, atol=1e-12
        )

    def test_nystrom_preconditioner_reduces_condition_number(self):
        size = 400
        num_spikes = 20
        matrix = _spiked_matrix(size, num_spikes, self.rng)
        ridge = 0.2
        covariance = matrix + ridge * numpy.eye(size)
        unpreconditioned = numpy.linalg.cond(covariance)

        # Randomized Nystrom needs to oversample past the number of spikes.
        sketch = NystromSketch(
            [as_symmetric_operator(matrix)], 3 * num_spikes, numpy.random.default_rng(0)
        )
        preconditioner = sketch.preconditioner([1.0], ridge)
        self.assertIsNotNone(preconditioner)
        preconditioned = numpy.linalg.eigvals(
            materialize(preconditioner) @ covariance
        ).real
        self.assertLess(
            preconditioned.max() / preconditioned.min(), 0.1 * unpreconditioned
        )

    def test_nystrom_sketch_is_reused_across_coefficients(self):
        # The sketch of each component is taken once; changing the variance
        # components must not need new products.
        sketch = NystromSketch(
            [as_symmetric_operator(self.grm)], 20, numpy.random.default_rng(0)
        )
        self.assertEqual(sketch.num_matvecs, 20)
        for ridge in (0.1, 1.0, 10.0):
            self.assertIsNotNone(sketch.preconditioner([0.5], ridge))
        self.assertEqual(sketch.num_matvecs, 20)

    def test_nystrom_returns_none_for_zero_operator(self):
        sketch = NystromSketch(
            [as_symmetric_operator(numpy.zeros((10, 10)))],
            5,
            numpy.random.default_rng(0),
        )
        self.assertIsNone(sketch.preconditioner([1.0], 1.0))

    def test_preconditioned_and_unpreconditioned_agree(self):
        rhs = self.rng.standard_normal(self.size)
        reference = numpy.linalg.solve(self.covariance, rhs)
        for rank in (0, 10, 60):
            solver = ConjugateGradientSolver(
                self.operators,
                tol=1e-10,
                preconditioner_rank=rank,
                rng=numpy.random.default_rng(3),
            )
            solver.update(self.coefficients)
            numpy.testing.assert_allclose(solver.solve(rhs), reference, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
