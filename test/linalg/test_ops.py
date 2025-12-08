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
from testing_utils import construct_grg, standardize_X, grg2X

CLEANUP = True
JOBS = 4

# Absolute error tolerated between numpy and GRG methods.
ABSOLUTE_TOLERANCE = 1e-10


def split_and_load(grg_filename: str, out_dir: str, cleanup: bool = True):
    subprocess.check_output(
        [
            "grg",
            "split",
            "-j",
            str(JOBS),
            grg_filename,
            "-s",
            str(1_000_000),
            "-o",
            out_dir,
        ]
    )
    grgs = []
    for fn in glob.glob(os.path.join(out_dir, "*.grg")):
        grgs.append(pygrgl.load_immutable_grg(fn))
    grgs.sort(key=lambda g: g.bp_range[0])
    if cleanup:
        shutil.rmtree(out_dir)
    return grgs


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
        grgs = split_and_load(self.grg_filename, test_dir, CLEANUP)

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

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
