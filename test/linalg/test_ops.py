from grapp.linalg.ops_scipy import (
    SciPyXOperator,
    SciPyXTXOperator,
    SciPyStdXOperator,
    SciPyStdXTXOperator,
    GRGOperatorFilter,
)
from grapp.util import allele_frequencies
import pygrgl
import numpy
import os
import unittest
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, standardize_X, grg2X

CLEANUP = True

# Absolute error tolerated between numpy and GRG methods.
ABSOLUTE_TOLERANCE = 1e-10


class TestLinearOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.linop.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename)

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

    def test_filtering(self):
        """
        Test the operators with filters enabled.
        """
        keep_individuals = [1, 11, 13, 23, 45, 99]
        keep_mutations = list(range(self.grg.num_mutations // 2))
        test_filter = GRGOperatorFilter(
            sample_filter=keep_individuals, mutation_filter=keep_mutations
        )

        K = 20  # Use 20 random vectors for test.
        random_mutvals = numpy.random.standard_normal((K, len(keep_mutations))).T
        random_sampvals = numpy.random.standard_normal((K, len(keep_individuals))).T

        X = grg2X(self.grg, diploid=True)
        X_std = standardize_X(X)
        X_dip = X[keep_individuals, :][:, keep_mutations]
        X_dip_std = X_std[keep_individuals, :][:, keep_mutations]
        # Here we use the full dataset frequencies, hence we standardize X _first_ above, before filtering
        # it. If we filtered first, we would need to compute the frequencies only on the subset of GRG.
        freqs = allele_frequencies(self.grg)

        ### Non-standardized X operator
        # UP
        numpy_dip_result = numpy.matmul(X_dip, random_mutvals)
        grg_dip_op = SciPyXOperator(
            self.grg, pygrgl.TraversalDirection.UP, filter=test_filter
        )
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        # DOWN
        numpy_dip_result = numpy.matmul(X_dip.T, random_sampvals)
        grg_dip_op = SciPyXOperator(
            self.grg, pygrgl.TraversalDirection.DOWN, filter=test_filter
        )
        grg_dip_result = grg_dip_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Non-standardized XTX operator
        numpy_dip_result = numpy.matmul(numpy.matmul(X_dip.T, X_dip), random_mutvals)
        grg_dip_op = SciPyXTXOperator(self.grg, filter=test_filter)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ### Standardized X operator
        # UP
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvals)
        grg_op = SciPyStdXOperator(
            self.grg, pygrgl.TraversalDirection.UP, freqs, filter=test_filter
        )
        grg_dip_result = grg_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        # DOWN
        numpy_dip_result = numpy.matmul(X_dip_std.T, random_sampvals)
        grg_op = SciPyStdXOperator(
            self.grg, pygrgl.TraversalDirection.DOWN, freqs, filter=test_filter
        )
        grg_dip_result = grg_op._matmat(random_sampvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)

        ###### Filter only mutations, keep all samples
        test_filter = GRGOperatorFilter(mutation_filter=keep_mutations)
        X_dip_std = X_std[:, keep_mutations]
        numpy_dip_result = numpy.matmul(X_dip_std, random_mutvals)
        grg_op = SciPyStdXOperator(
            self.grg, pygrgl.TraversalDirection.UP, freqs, filter=test_filter
        )
        grg_dip_result = grg_op._matmat(random_mutvals)
        numpy.testing.assert_allclose(grg_dip_result, numpy_dip_result)
        self.assertEqual(grg_dip_result.shape, (self.grg.num_individuals, K))

    def test_mut_filter(self):
        """
        Test the operators with filters enabled.
        """
        keep_mutations = list(range(self.grg.num_mutations))
        drop_mut = self.grg.num_mutations // 2
        del keep_mutations[drop_mut]
        test_filter = GRGOperatorFilter(mutation_filter=keep_mutations)

        K = 20  # Use 20 random vectors for test.
        random_mutvals = numpy.random.standard_normal((K, len(keep_mutations))).T

        X = grg2X(self.grg, diploid=True)

        ### Non-standardized XTX operator
        grg_dip_op = SciPyXTXOperator(self.grg, filter=test_filter)
        grg_dip_result = grg_dip_op._matmat(random_mutvals)
        self.assertEqual(grg_dip_op.shape[0], grg_dip_result.shape[0])

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
