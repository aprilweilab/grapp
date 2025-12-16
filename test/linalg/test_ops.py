from grapp.linalg.ops_scipy import (
    MultiSciPyStdXOperator,
    MultiSciPyStdXTXOperator,
    MultiSciPyXOperator,
    MultiSciPyXTXOperator,
    SciPyStdXOperator,
    SciPyStdXTXOperator,
    SciPyXOperator,
    SciPyXTXOperator,
)
from grapp.util import allele_frequencies
import glob
import numpy
import os
import pygrgl
import shutil
import subprocess
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, standardize_X, grg2X, split_and_load

CLEANUP = True
JOBS = 4

# Absolute error tolerated between numpy and GRG methods.
ABSOLUTE_TOLERANCE = 1e-10


class TestLinearOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.linop.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

        numpy.random.seed(42)

    def test_simple_op(self):
        """
        The simple operator works on the genotype matrix without any modification. In this case,
        the diploid and haploid formulations should product the same result.
        """
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X_hap = grg2X(self.grg, diploid=False)
        numpy_hap_result = numpy.matmul(X_hap, random_input)

        X_dip = grg2X(self.grg, diploid=True)
        self.assertNotEqual(X_hap.shape, X_dip.shape)
        self.assertEqual(numpy.max(X_hap), 1)
        self.assertEqual(numpy.max(X_dip), 2)
        numpy_dip_result = numpy.matmul(X_dip, random_input)
        self.assertAlmostEqual(numpy.sum(numpy_hap_result), numpy.sum(numpy_dip_result))

        grg_hap_op = SciPyXOperator(
            self.grg, pygrgl.TraversalDirection.UP, haploid=True
        )
        grg_hap_result = grg_hap_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_hap_result, numpy_hap_result)

    def test_standardized_op_X(self):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        numpy_result = numpy.matmul(X_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(self.grg, pygrgl.TraversalDirection.UP, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_standardized_op_XT(self):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T

        X = grg2X(self.grg, diploid=True)
        XT_stand = standardize_X(X).T
        numpy_result = numpy.matmul(XT_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(self.grg, pygrgl.TraversalDirection.DOWN, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_standardized_op_XtX(self):
        K = 20  # Number of random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        XtX = X_stand.T @ X_stand
        # (NxM)x(MxK) == NxK
        numpy_result = numpy.matmul(XtX, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXTXOperator(self.grg, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

        # Reversed result should be identical, because XtX.T == XtX
        numpy_result = numpy.matmul(XtX.T, random_input)
        grg_result = grg_op._rmatmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    def test_multi_ops(self):
        """
        Test that the operators that work with multiple GRGs produce the same result
        as ones that work with a single GRG.
        """

        # Split the graph and get the multiple GRGs, for testing all of the below.
        test_dir = "test.multi_ops.split"
        grgs = split_and_load(self.grg_filename, test_dir, 1_000_000, JOBS, CLEANUP)

        #### Direction == UP
        K = 10
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        # Result on the full graph.
        grg_op = SciPyXOperator(self.grg, pygrgl.TraversalDirection.UP, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, pygrgl.TraversalDirection.UP, haploid=False, threads=JOBS
        )
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        #### Direction == DOWN
        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        # Reverse from above
        full_dip_result = grg_op._rmatmat(random_input)
        split_dip_result = multi_op._rmatmat(random_input)
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Result on the full graph.
        grg_op = SciPyXOperator(self.grg, pygrgl.TraversalDirection.DOWN, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, pygrgl.TraversalDirection.DOWN, haploid=False, threads=JOBS
        )
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### XTX non-standardized
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        grg_op = SciPyXTXOperator(self.grg, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXTXOperator(grgs, haploid=False, threads=JOBS)
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### X standardized (UP)
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(
            self.grg, pygrgl.TraversalDirection.UP, freqs, haploid=False
        )
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        freq_list = list(map(allele_frequencies, grgs))
        multi_op = MultiSciPyStdXOperator(
            grgs, pygrgl.TraversalDirection.UP, freq_list, haploid=False, threads=JOBS
        )
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        #### XTX standardized
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T
        # Result on the full graph.
        grg_op = SciPyStdXTXOperator(self.grg, freqs, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyStdXTXOperator(
            grgs, freq_list, haploid=False, threads=JOBS
        )
        split_dip_result = multi_op._matmat(random_input)
        # Test equality
        numpy.testing.assert_allclose(
            full_dip_result, split_dip_result, atol=ABSOLUTE_TOLERANCE
        )

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        ### Test with a contiguous mutation filter
        total_muts = sum([g.num_mutations for g in grgs])
        keep_mutations = list(range(total_muts // 2))
        random_input = numpy.random.standard_normal((K, len(keep_mutations))).T
        grg_op = SciPyXOperator(
            self.grg,
            pygrgl.TraversalDirection.UP,
            haploid=False,
            mutation_filter=keep_mutations,
        )
        full_dip_result = grg_op._matmat(random_input)
        multi_op = MultiSciPyXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            haploid=False,
            mutation_filter=keep_mutations,
            threads=JOBS,
        )
        split_dip_result = multi_op._matmat(random_input)
        numpy.testing.assert_allclose(full_dip_result, split_dip_result)

        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        ### Test with a scattered mutation filter
        total_muts = sum([g.num_mutations for g in grgs])
        keep_mutations = [i * 2 for i in range(total_muts // 2)]
        random_input = numpy.random.standard_normal((K, len(keep_mutations))).T
        freqs = allele_frequencies(self.grg)
        freq_list = list(map(allele_frequencies, grgs))

        grg_op = SciPyStdXTXOperator(
            self.grg,
            freqs,
            haploid=False,
            mutation_filter=keep_mutations,
        )
        full_dip_result = grg_op._matmat(random_input)
        multi_op = MultiSciPyStdXTXOperator(
            grgs,
            freq_list,
            haploid=False,
            mutation_filter=keep_mutations,
            threads=JOBS,
        )
        split_dip_result = multi_op._matmat(random_input)
        numpy.testing.assert_allclose(
            full_dip_result, split_dip_result, atol=ABSOLUTE_TOLERANCE
        )
        # Vector version
        vec_result = grg_op._matvec(random_input[:, 1])
        split_vec_result = multi_op._matvec(random_input[:, 1])
        numpy.testing.assert_allclose(
            vec_result, split_vec_result, atol=ABSOLUTE_TOLERANCE
        )

        # Reverse direction
        # random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        rev_result = grg_op._rmatmat(random_input)
        split_rev_result = multi_op._rmatmat(random_input)
        numpy.testing.assert_allclose(
            rev_result, split_rev_result, atol=ABSOLUTE_TOLERANCE
        )

    def test_filtering(self):
        """
        Test the operators with filters enabled.
        """
        keep_mutations = list(range(self.grg.num_mutations // 2))

        K = 20  # Use 20 random vectors for test.
        random_mutvals = numpy.random.standard_normal((K, len(keep_mutations))).T
        random_mutvec = numpy.random.standard_normal(len(keep_mutations))
        random_sampvals = numpy.random.standard_normal((K, self.grg.num_individuals)).T

        X = grg2X(self.grg, diploid=True)
        X_std = standardize_X(X)
        X_dip = X[:, keep_mutations]
        X_dip_std = X_std[:, keep_mutations]
        freqs = allele_frequencies(self.grg)

        ### Non-standardized X operator
        # UP
        grg_dip_op = SciPyXOperator(
            self.grg, pygrgl.TraversalDirection.UP, mutation_filter=keep_mutations
        )
        numpy_dip_result = numpy.matmul(X_dip, random_mutvec)
        grg_dip_result = grg_dip_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(X_dip, random_mutvals)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        grg_dip_multi_op = MultiSciPyXOperator(
            [self.grg], pygrgl.TraversalDirection.UP, mutation_filter=keep_mutations
        )
        grg_dip_multi_result = grg_dip_multi_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_multi_result, numpy_dip_result)

        # DOWN
        grg_dip_op = SciPyXOperator(
            self.grg, pygrgl.TraversalDirection.DOWN, mutation_filter=keep_mutations
        )
        numpy_dip_result = numpy.matmul(X_dip.T, random_sampvals)
        grg_dip_result = grg_dip_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Non-standardized XTX operator
        grg_dip_op = SciPyXTXOperator(self.grg, mutation_filter=keep_mutations)
        numpy_dip_result = numpy.matmul(numpy.matmul(X_dip.T, X_dip), random_mutvec)
        grg_dip_result = grg_dip_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(numpy.matmul(X_dip.T, X_dip), random_mutvals)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Standardized X operator
        # UP
        grg_op = SciPyStdXOperator(
            self.grg,
            pygrgl.TraversalDirection.UP,
            freqs,
            mutation_filter=keep_mutations,
        )
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvec)
        grg_dip_result = grg_op._matvec(random_mutvec).squeeze()
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvals)
        grg_dip_result = grg_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        # DOWN
        numpy_dip_result = numpy.matmul(X_dip_std.T, random_sampvals)
        grg_op = SciPyStdXOperator(
            self.grg,
            pygrgl.TraversalDirection.DOWN,
            freqs,
            mutation_filter=keep_mutations,
        )
        grg_dip_result = grg_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
