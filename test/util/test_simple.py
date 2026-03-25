from grapp.util.simple import (
    hwe_df,
    site_alleles,
    variants_of_types,
    VariantType,
)
import numpy
import os
import pandas
import pygrgl
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, split_and_load

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.filt.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

    def test_site_alleles(self):
        counts = site_alleles(self.grg)
        self.assertTrue(numpy.all(counts >= 2))

        # These have been verified to be multi-allelic in the freq output
        expected = [
            481,
            482,
            1038,
            1039,
            1645,
            1646,
            1664,
            1665,
            4280,
            4281,
            7227,
            7228,
            8025,
            8026,
            8925,
            8926,
        ]
        self.assertEqual(expected, list(numpy.where(counts >= 3)[0]))

        # Only count ALTs
        counts = site_alleles(self.grg, alt_only=True)
        self.assertTrue(numpy.all(counts >= 1))
        self.assertEqual(expected, list(numpy.where(counts >= 2)[0]))

        # Only count for a few mutations
        counts = site_alleles(self.grg, mut_ids=[1, 5, 9, 481, 482])
        self.assertTrue(numpy.all(counts >= 2))
        self.assertEqual([3, 4], list(numpy.where(counts >= 3)[0]))

    def test_hwe_single(self):
        # Just a smoke test for now
        hom_A = 71758
        het_A = 205484
        other = 222758
        expected = 1.57897e-984
        result = pygrgl.hwe_exact_pv(het_A, hom_A, other)
        self.assertAlmostEqual(expected, result, places=5)

    def test_hwe_df(self):
        # Baseline test
        result = hwe_df(self.grg, jobs=JOBS)
        expect = pandas.read_csv(
            os.path.join(INPUT_DIR, "hwe.baseline.tsv"), delimiter="\t"
        )
        numpy.testing.assert_allclose(result["P"], expect["P"])
        numpy.testing.assert_allclose(result["COUNT"], expect["COUNT"])

    def test_var_types(self):
        # All variants in our test data should be SNPs (this is not a very good test)
        all_muts = variants_of_types(self.grg, set([VariantType.SNPS]))
        self.assertEqual(all_muts, list(range(self.grg.num_mutations)))

        no_muts = variants_of_types(self.grg, set([VariantType.MNPS]))
        self.assertEqual(len(no_muts), 0)

        no_muts = variants_of_types(self.grg, set([VariantType.INDELS]))
        self.assertEqual(len(no_muts), 0)

        no_muts = variants_of_types(self.grg, set([VariantType.OTHER]))
        self.assertEqual(len(no_muts), 0)

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
