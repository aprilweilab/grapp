import os
import sys
import unittest
import numpy
import pygrgl
from grapp.popgen import (
    sfs,
    sfs_scaled,
    sfs_folded,
    sfs_folded_scaled,
    joint_sfs,
    joint_sfs_scaled,
    joint_sfs_folded,
    joint_sfs_folded_scaled,
)

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import make_grg_sparse_mat

INPUT_DIR = os.path.join(THIS_DIR, "..", "input")


class TestSFS(unittest.TestCase):
    def setUp(cls):
        # 5 samples, 8 variants.
        cls.gt = numpy.array(
            [
                [0, 1, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 1, 1, 1],
            ]
        )
        # pop_sample_matrix - 3 populations, 5 samples
        cls.psm = numpy.array(
            [
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        )
        cls.grg = make_grg_sparse_mat(
            list(range(8)), ["A"] * 8, ["T"] * 8, cls.gt, pop_sample_matrix=cls.psm
        )

    def test_sfs(self):
        # The simple SFS over all populations
        s = sfs(self.grg)
        numpy.testing.assert_equal([0, 0, 4, 2, 1, 1], s)

        # Just make sure the scaled versions don't barf
        s = sfs_scaled(self.grg)
        self.assertEqual((self.grg.num_samples + 1,), s.shape)
        s = sfs_folded_scaled(self.grg)
        self.assertEqual(((self.grg.num_samples // 2) + 1,), s.shape)

        # Folded
        s = sfs_folded(self.grg)
        numpy.testing.assert_equal([1, 1, 6], s)

    def test_joint_sfs(self):
        def _get_pair(a, b, list_of_results):
            for pop1, pop2, sfs in list_of_results:
                if (pop1, pop2) == (a, b):
                    return sfs
            return None

        # Pop0:
        #                [1, 0, 1, 1, 0, 1, 0, 1],
        #                [0, 1, 0, 1, 1, 1, 1, 1],
        # Pop1:
        #                [0, 1, 1, 0, 0, 1, 0, 0],
        #                [1, 0, 1, 1, 0, 1, 0, 0],
        # Pop2:
        #                [0, 1, 0, 1, 1, 1, 1, 0],

        # The joint SFS for each unique population pair
        jsfs_list = joint_sfs(self.grg)

        # pop1, pop0  (the order is always larger index on y-axis)
        numpy.testing.assert_equal(
            [[0, 2, 1], [0, 2, 1], [0, 1, 1]],  # pop0 --->  # pop1  #  |  #  V
            _get_pair(1, 0, jsfs_list),
        )

        # pop2, pop1  (the order is always larger index on y-axis)
        numpy.testing.assert_equal(
            [  # pop1 --->
                [1, 1, 1],  # pop2
                [2, 2, 1],  #  |
                #  V
            ],
            _get_pair(2, 1, jsfs_list),
        )


if __name__ == "__main__":
    unittest.main()
