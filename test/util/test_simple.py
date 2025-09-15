from grapp.util.simple import allele_frequencies
import numpy
import os
import pygrgl
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg

CLEANUP = True
JOBS = 4
VERBOSE = True


class TestSimpleUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.nomiss.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

        cls.miss_filename = construct_grg("test-200-missing.vcf.gz", "test.miss.grg")
        cls.miss_grg = pygrgl.load_immutable_grg(cls.miss_filename, load_up_edges=False)

    def test_adjust_nothing(self):
        """
        Allele frequencies are identical whether adjusted or not if there is no missing data.
        """
        basic_af = allele_frequencies(self.grg)
        miss_af = allele_frequencies(self.grg, adjust_missing=True)
        numpy.testing.assert_array_equal(basic_af, miss_af)

    def test_adjust_correct(self):
        """
        Allele frequencies differ only in mutations with missing alleles when adjusted.
        """
        muts_with_missing = set()
        for node, mut, missing_node in self.miss_grg.get_node_mutation_miss():
            if missing_node != pygrgl.INVALID_NODE:
                muts_with_missing.add(mut)
        assert len(muts_with_missing) == 18

        basic_af = allele_frequencies(self.miss_grg)
        miss_af = allele_frequencies(self.miss_grg, adjust_missing=True)
        self.assertEqual(len(basic_af), len(miss_af))
        # Only allele frequencies for muts with missing data should change, and those should
        # always have _larger_ frequencies because there are fewer datapoints in the denominator.
        for i in range(len(basic_af)):
            self.assertTrue((basic_af[i] == miss_af[i]) or (i in muts_with_missing))
            if i in muts_with_missing and basic_af[i] != 0.0:
                self.assertGreater(miss_af[i], basic_af[i])

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
