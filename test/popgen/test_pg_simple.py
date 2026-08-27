import os
import sys
import unittest
import numpy
import pygrgl
from grapp.popgen import (
    allele_counts,
    pop_allele_counts,
    population_pairs,
    sample_pop_matrix,
)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import make_grg_sparse_mat

INPUT_DIR = os.path.join(THIS_DIR, "..", "input")


class TestSimple(unittest.TestCase):
    def setUp(cls):
        # 4 samples, 6 variants.
        cls.gt = numpy.array(
            [
                [0, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [0, 1, 0, 1, 1, 1],
            ]
        )
        # pop_sample_matrix - 3 populations, 4 samples
        cls.psm = numpy.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        cls.grg = make_grg_sparse_mat(
            list(range(6)), ["A"] * 6, ["T"] * 6, cls.gt, pop_sample_matrix=cls.psm
        )

    def test_pop_matrix(self):
        result = sample_pop_matrix(self.grg)
        numpy.testing.assert_equal(self.psm, result)

    def test_pop_pairs(self):
        self.assertEqual([(2, 1), (2, 0), (1, 0)], population_pairs(self.grg))

    def test_pop_ac(self):
        pop_ac = pop_allele_counts(self.grg, return_ref=False)
        self.assertEqual(pop_ac.shape, (3, 6))
        numpy.testing.assert_equal(
            [
                [1, 0, 1, 1, 0, 1],  # pop0
                [1, 1, 2, 1, 0, 2],  # pop1
                [0, 1, 0, 1, 1, 1],  # pop2
            ],
            pop_ac,
        )

        # return_ref=True adds another dimension, showing the REF count
        pop_ac = pop_allele_counts(self.grg, return_ref=True)
        self.assertEqual(pop_ac.shape, (3, 6, 2))
        numpy.testing.assert_equal(
            [
                [(0, 1), (1, 0), (0, 1), (0, 1), (1, 0), (0, 1)],  # pop0
                [(1, 1), (1, 1), (0, 2), (1, 1), (2, 0), (0, 2)],  # pop1
                [(1, 0), (0, 1), (1, 0), (0, 1), (0, 1), (0, 1)],  # pop2
            ],
            pop_ac,
        )

    def test_ac(self):
        ac = allele_counts(self.grg, return_ref=False)
        self.assertEqual(ac.shape, (6,))
        numpy.testing.assert_equal(
            [2, 2, 3, 3, 1, 4],
            ac,
        )

        ac = allele_counts(self.grg, return_ref=True)
        self.assertEqual(ac.shape, (6, 2))
        numpy.testing.assert_equal(
            [(2, 2), (2, 2), (1, 3), (1, 3), (3, 1), (0, 4)],
            ac,
        )


if __name__ == "__main__":
    unittest.main()
