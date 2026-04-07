from grapp.linalg.ops_scipy import (
    MultiSciPyStdXOperator,
    MultiSciPyStdXTXOperator,
    MultiSciPyXOperator,
    MultiSciPyXTXOperator,
    SciPyStdXOperator,
    SciPyXXTOperator,
    SciPyStdXTXOperator,
    SciPyXOperator,
    SciPyXTXOperator,
)
from grapp.util import allele_frequencies
from grapp.util.filter import grg_save_samples
from parameterized import parameterized
import numpy
import os
import pygrgl
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import (
    construct_grg,
    standardize_X,
    grg2X,
    split_and_load,
    complete_sample_sets,
    WRAP_GRG_PARAMS,
)

CLEANUP = True
JOBS = 4

# Absolute error tolerated between numpy and GRG methods.
ABSOLUTE_TOLERANCE = 1e-10


class TestLinearOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.linop.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=True)

        numpy.random.seed(42)

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_simple_op(self, wrap_grg):
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
            wrap_grg(self.grg), pygrgl.TraversalDirection.UP, haploid=True
        )
        grg_hap_result = grg_hap_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_hap_result, numpy_hap_result)

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_XXT(self, wrap_grg):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_samples)).T

        X_hap = grg2X(self.grg, diploid=False)
        numpy_hap_result = numpy.matmul(X_hap @ X_hap.T, random_input)

        grg_hap_op = SciPyXXTOperator(wrap_grg(self.grg), haploid=True)
        grg_hap_result = grg_hap_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_hap_result, numpy_hap_result)

        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T
        X_dip = grg2X(self.grg, diploid=True)
        numpy_dip_result = numpy.matmul(X_dip @ X_dip.T, random_input)
        grg_dip_op = SciPyXXTOperator(wrap_grg(self.grg), haploid=False)
        grg_dip_result = grg_dip_op._matmat(random_input)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_standardized_op_X(self, wrap_grg):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        numpy_result = numpy.matmul(X_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg_op = SciPyStdXOperator(
            wrap_grg(self.grg), pygrgl.TraversalDirection.UP, freqs
        )
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_standardized_op_XT(self, wrap_grg):
        K = 20  # Use 20 random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_individuals)).T

        X = grg2X(self.grg, diploid=True)
        XT_stand = standardize_X(X).T
        numpy_result = numpy.matmul(XT_stand, random_input)

        freqs = allele_frequencies(self.grg)
        grg = wrap_grg(self.grg)
        grg_op = SciPyStdXOperator(grg, pygrgl.TraversalDirection.DOWN, freqs)
        grg_result = grg_op._matmat(random_input)

        self.assertFalse(numpy.any(numpy.isinf(grg_result)))
        self.assertFalse(numpy.any(numpy.isinf(numpy_result)))
        self.assertFalse(numpy.any(numpy.isnan(grg_result)))
        self.assertFalse(numpy.any(numpy.isnan(numpy_result)))
        numpy.testing.assert_allclose(grg_result, numpy_result, atol=ABSOLUTE_TOLERANCE)

        # Use non-default alpha value for variance, and verify it differs from default
        grg_alpha2_op = SciPyStdXOperator(
            grg, pygrgl.TraversalDirection.DOWN, freqs, alpha=-2
        )
        alpha2_result = grg_alpha2_op._matmat(random_input)
        self.assertFalse(numpy.allclose(grg_result, alpha2_result))

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_standardized_op_XtX(self, wrap_grg):
        K = 20  # Number of random vectors for test.
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        X = grg2X(self.grg, diploid=True)
        X_stand = standardize_X(X)
        XtX = X_stand.T @ X_stand
        # (NxM)x(MxK) == NxK
        numpy_result = numpy.matmul(XtX, random_input)

        freqs = allele_frequencies(self.grg)
        grg = wrap_grg(self.grg)
        grg_op = SciPyStdXTXOperator(grg, freqs)
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

        # Use non-default alpha value for variance, and verify it differs from default
        grg_alpha2_op = SciPyStdXTXOperator(grg, freqs, alpha=-2)
        alpha2_result = grg_alpha2_op._rmatmat(random_input)
        self.assertFalse(numpy.allclose(grg_result, alpha2_result))

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_multi_ops(self, wrap_grg):
        """
        Test that the operators that work with multiple GRGs produce the same result
        as ones that work with a single GRG.
        """

        # Split the graph and get the multiple GRGs, for testing all of the below.
        test_dir = "test.multi_ops.split"
        grgs = [wrap_grg(g) for g in split_and_load(self.grg_filename, test_dir, 1_000_000, JOBS, CLEANUP)]
        grg = wrap_grg(self.grg)

        #### Direction == UP
        K = 10
        random_input = numpy.random.standard_normal((K, self.grg.num_mutations)).T

        # Result on the full graph.
        grg_op = SciPyXOperator(grg, pygrgl.TraversalDirection.UP, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, pygrgl.TraversalDirection.UP, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
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
        grg_op = SciPyXOperator(grg, pygrgl.TraversalDirection.DOWN, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXOperator(
            grgs, pygrgl.TraversalDirection.DOWN, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
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
        grg_op = SciPyXTXOperator(grg, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyXTXOperator(grgs, haploid=False, threads=JOBS)
        self.assertEqual(multi_op.shape, grg_op.shape)
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
            grg, pygrgl.TraversalDirection.UP, freqs, haploid=False
        )
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        freq_list = list(map(allele_frequencies, grgs))
        multi_op = MultiSciPyStdXOperator(
            grgs, pygrgl.TraversalDirection.UP, freq_list, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
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
        grg_op = SciPyStdXTXOperator(grg, freqs, haploid=False)
        full_dip_result = grg_op._matmat(random_input)
        # Result on the split graph
        multi_op = MultiSciPyStdXTXOperator(
            grgs, freq_list, haploid=False, threads=JOBS
        )
        self.assertEqual(multi_op.shape, grg_op.shape)
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
            grg,
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
        self.assertEqual(multi_op.shape, grg_op.shape)
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
            grg,
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
        self.assertEqual(multi_op.shape, grg_op.shape)
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

    @parameterized.expand(WRAP_GRG_PARAMS)
    def test_missing(self, wrap_grg):
        # Properties of the input data.
        MISSING_INDIVS = 21
        grg_filename = construct_grg("test-200-samples.miss.igd", "test.linop.miss.grg")
        raw_grg = pygrgl.load_immutable_grg(grg_filename, load_up_edges=False)

        # X is the explicit genotype matrix, with allele frequency used for missing items. So the
        # only non-0,1,2 values should be missing items.
        X = grg2X(raw_grg, diploid=True)
        self.assertEqual(
            len(numpy.where((X > 0) & (X != 1) & (X != 2))[0]), MISSING_INDIVS
        )

        # Create the operator, using the allele frequencies as the mean value for each Mutation
        freqs = allele_frequencies(raw_grg, adjust_missing=True)
        grg = wrap_grg(raw_grg)
        X_op = SciPyXOperator(grg, pygrgl.TraversalDirection.UP, miss_values=freqs)

        #### UP direction (AX) ####
        K = 7
        rv = numpy.random.standard_normal((K, self.grg.num_individuals))
        # Using the explicit genotype matrix vs. GRG operator should produce identical results.
        numpy_result = rv @ X
        grg_result = rv @ X_op
        numpy.testing.assert_allclose(numpy_result, grg_result)

        #### DOWN direction (AX^T) ####
        rv = numpy.random.standard_normal((K, self.grg.num_mutations))
        # Using the explicit genotype matrix vs. GRG operator should produce identical results.
        numpy_result = rv @ X.T
        grg_result = rv @ X_op.T
        numpy.testing.assert_allclose(numpy_result, grg_result)

        # Just a sanity check: using the non-missingness-adjusted operator should cause failure.
        X_nomiss_op = SciPyXOperator(grg, pygrgl.TraversalDirection.UP)
        grg_result = rv @ X_nomiss_op.T
        self.assertFalse(numpy.allclose(numpy_result, grg_result))

    def test_filter_samples(self):
        """
        We downsample a GRG explicitly, and then use an operator's filter to ignore the same individuals,
        and expect the matrix multiplication should produce the same result.
        """
        keep_indivs, ignore_indivs, keep_samples, ignore_samples = complete_sample_sets(
            self.grg, [4, 9, 101, 177]
        )

        filt_name = "test.ignore_samples.grg"
        grg_save_samples(self.grg, filt_name, keep_samples)
        filt_grg = pygrgl.load_immutable_grg(filt_name, load_up_edges=False)

        K = 17
        Y = numpy.random.standard_normal((K, len(keep_indivs)))

        # Non-standardized operator
        truth_op = SciPyXOperator(filt_grg, pygrgl.TraversalDirection.UP)
        truth_matrix = Y @ truth_op
        mask_op = SciPyXOperator(
            self.grg,
            pygrgl.TraversalDirection.UP,
            sample_filter=keep_indivs,
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(truth_matrix, mask_matrix)

        # Standardized operator
        truth_freqs = allele_frequencies(filt_grg)
        mask_freqs = allele_frequencies(self.grg, sample_filter=keep_samples)
        numpy.testing.assert_allclose(truth_freqs, mask_freqs, atol=ABSOLUTE_TOLERANCE)

        truth_op = SciPyStdXOperator(
            filt_grg, pygrgl.TraversalDirection.UP, truth_freqs
        )
        truth_matrix = Y @ truth_op
        mask_op = SciPyStdXOperator(
            self.grg,
            pygrgl.TraversalDirection.UP,
            mask_freqs,
            sample_filter=keep_indivs,
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(
            truth_matrix, mask_matrix, atol=ABSOLUTE_TOLERANCE
        )

        # Now do some testing with the DOWN direction
        Y = numpy.random.standard_normal((K, self.grg.num_mutations))

        # Non-standardized operator
        truth_op = SciPyXOperator(filt_grg, pygrgl.TraversalDirection.DOWN)
        truth_matrix = Y @ truth_op
        mask_op = SciPyXOperator(
            self.grg,
            pygrgl.TraversalDirection.DOWN,
            sample_filter=keep_indivs,
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(truth_matrix, mask_matrix)

        # Standardized operator
        truth_freqs = allele_frequencies(filt_grg)
        mask_freqs = allele_frequencies(self.grg, sample_filter=keep_samples)
        numpy.testing.assert_allclose(truth_freqs, mask_freqs, atol=ABSOLUTE_TOLERANCE)

        truth_op = SciPyStdXOperator(
            filt_grg, pygrgl.TraversalDirection.DOWN, truth_freqs
        )
        truth_matrix = Y @ truth_op
        mask_op = SciPyStdXOperator(
            self.grg,
            pygrgl.TraversalDirection.DOWN,
            mask_freqs,
            sample_filter=keep_indivs,
        )
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(
            truth_matrix, mask_matrix, atol=ABSOLUTE_TOLERANCE
        )

        # XTX
        truth_op = SciPyXTXOperator(filt_grg)
        truth_matrix = Y @ truth_op
        mask_op = SciPyXTXOperator(self.grg, sample_filter=keep_indivs)
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(
            truth_matrix, mask_matrix, atol=ABSOLUTE_TOLERANCE
        )

        # XTX standardized
        truth_op = SciPyStdXTXOperator(filt_grg, truth_freqs)
        truth_matrix = Y @ truth_op
        mask_op = SciPyStdXTXOperator(self.grg, mask_freqs, sample_filter=keep_indivs)
        mask_matrix = Y @ mask_op
        numpy.testing.assert_allclose(
            truth_matrix, mask_matrix, atol=ABSOLUTE_TOLERANCE
        )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
