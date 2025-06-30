from grgapp.linalg import eigs as grg_eigs
from grgapp.linalg import PCs
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
    def setUp(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.pca.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename)

    def test_eigvals(self):
        X_stand = standardize_X(grg2X(self.grg, diploid=True))
        D = X_stand.T @ X_stand
        evals, evects = scipy_eigs(D, k=15)
        grg_evals, grg_evects = grg_eigs(self.grg, 15)
        numpy.testing.assert_array_almost_equal(evals, grg_evals, 3)
        for ev, gev in zip(evects.T, grg_evects.T):
            # Vectors may differ by sign.
            self.assertTrue(numpy.allclose(ev, gev) or numpy.allclose(-ev, gev))

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
