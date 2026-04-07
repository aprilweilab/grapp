from grapp.linalg import (
    MatrixSelection,
    PCs,
    eigs as grg_eigs,
    get_pcs_propca,
    sort_by_eigvalues,
)
from grapp.grg_calculator import GRGCalculator, _wrap_grg_spmv
from parameterized import parameterized
import pygrgl
import numpy
import os
import unittest
from scipy.sparse.linalg import eigs as scipy_eigs
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, grg2X, standardize_X

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.pca.grg")
        # Up edges needed for grg2X
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

    @parameterized.expand(
        [
            (lambda g: g,),
            (GRGCalculator,),
            (_wrap_grg_spmv,),
        ]
    )
    def test_eigvals(self, wrap_grg):
        X_stand = standardize_X(grg2X(self.grg, diploid=True))

        D = X_stand.T @ X_stand
        evals, evects = scipy_eigs(D, k=15, which="LR")

        # grg_eigs does this for us, because scipy _does not guarantee_ that these are in the
        # correct order.
        sort_by_eigvalues(evals, evects)

        grg_evals, grg_evects = grg_eigs(MatrixSelection.XTX, wrap_grg(self.grg), 15)
        numpy.testing.assert_array_almost_equal(
            numpy.flip(numpy.sort(grg_evals)), grg_evals, 10
        )
        numpy.testing.assert_array_almost_equal(evals, grg_evals, 3)
        for ev, gev in zip(evects.T, grg_evects.T):
            # Vectors may differ by sign.
            self.assertTrue(numpy.allclose(ev, gev) or numpy.allclose(-ev, gev))

    # Just make sure PCA succeeds
    @parameterized.expand(
        [
            (lambda g: g,),
            (GRGCalculator,),
            (_wrap_grg_spmv,),
        ]
    )
    def test_pca_smoketest(self, wrap_grg):
        # Not all recent versions of scipy support passing in a random number generator,
        # so we need to do this instead for testing purposes
        numpy.random.seed(42)
        scores = PCs(wrap_grg(self.grg), k=20).to_numpy()
        pca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "pca.expected.txt"))
        numpy.testing.assert_allclose(numpy.abs(scores), numpy.abs(pca_expect))

    # Just make sure PCA succeeds
    @parameterized.expand(
        [
            (lambda g: g,),
            (GRGCalculator,),
            (_wrap_grg_spmv,),
        ]
    )
    def test_propca_smoketest(self, wrap_grg):
        scores, _, _ = get_pcs_propca(wrap_grg(self.grg), k=20, convergence_lim=1e-5)
        propca_expect = numpy.loadtxt(os.path.join(INPUT_DIR, "propca.expected.txt"))
        numpy.testing.assert_allclose(
            numpy.abs(scores), numpy.abs(propca_expect), atol=0.005
        )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
