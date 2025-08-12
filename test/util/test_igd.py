from grapp.util.igd import (
    export_igd,
    igd_to_vcf,
)
import gzip
import os
import pygrgl
import pyigd
import unittest
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg

CLEANUP = True
JOBS = 4
VERBOSE = True


class TestIGDAndVCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.igd.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename)

    def test_round_trip_grg_to_igd_to_vcf(self):
        """
        The simple operator works on the genotype matrix without any modification. In this case,
        the diploid and haploid formulations should product the same result.
        """
        test_igd_file = "test.export.igd"
        export_igd(
            self.grg, test_igd_file, jobs=JOBS, split_threshold=100_000, verbose=VERBOSE
        )

        with open(test_igd_file, "rb") as figd:
            igd_reader = pyigd.IGDReader(figd)
            self.assertEqual(self.grg.num_mutations, igd_reader.num_variants)

        test_vcf_file = "test.export.vcf.gz"
        with gzip.open(test_vcf_file, "wt") as fgz:
            igd_to_vcf(test_igd_file, fgz, "test")

        new_grg_filename = construct_grg(
            test_vcf_file, "test.igd.roundtrip.grg", is_test_input=False
        )
        new_grg = pygrgl.load_immutable_grg(new_grg_filename)

        self.assertEqual(self.grg.num_samples, new_grg.num_samples)
        self.assertEqual(self.grg.ploidy, new_grg.ploidy)
        self.assertEqual(self.grg.num_mutations, new_grg.num_mutations)
        for mut_id in range(self.grg.num_mutations):
            mut = self.grg.get_mutation_by_id(mut_id)
            mut_new = new_grg.get_mutation_by_id(mut_id)
            self.assertEqual(mut.position, mut_new.position)
            self.assertEqual(mut.ref_allele, mut_new.ref_allele)
            self.assertEqual(mut.allele, mut_new.allele)

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
