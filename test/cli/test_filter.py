import os
import pygrgl
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, grapp_run

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestFilterCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.filtcli.grg")

    @unittest.skip("Skip until pygrgl v2.10 is released")
    def test_filter_indivs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            indivs_file = os.path.join(tmpdir, "indivs.txt")
            grg_out = os.path.join(tmpdir, "out.grg")
            indivs = ["tsk_10", "tsk_99", "tsk_13", "tsk_100", "tsk_58"]

            with open(indivs_file, "w") as fout:
                fout.write("\n".join(indivs))

            grapp_run("filter", "-S", indivs_file, self.grg_filename, grg_out)
            self.assertTrue(os.path.isfile(grg_out))
            grg = pygrgl.load_immutable_grg(grg_out)
            self.assertEqual(grg.ploidy, 2)
            self.assertEqual(grg.num_samples, len(indivs) * 2)
            self.assertEqual(grg.num_individuals, len(indivs))
            for i, ident in enumerate(indivs):
                self.assertEqual(grg.get_individual_id(i), ident)

    @unittest.skip("Skip until pygrgl v2.10 is released")
    def test_filter_haps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hap_file = os.path.join(tmpdir, "haps.txt")
            grg_out = os.path.join(tmpdir, "out.grg")

            with open(hap_file, "w") as fout:
                fout.write("""8
9
32
33
0
1
50
51""")

            grapp_run("filter", "--hap-samples", hap_file, self.grg_filename, grg_out)
            self.assertTrue(os.path.isfile(grg_out))
            grg = pygrgl.load_immutable_grg(grg_out)
            self.assertEqual(grg.num_samples, 8)
            self.assertEqual(grg.num_individuals, 4)
            self.assertEqual(grg.get_individual_id(0), "tsk_4")
            self.assertEqual(grg.get_individual_id(1), "tsk_16")
            self.assertEqual(grg.get_individual_id(2), "tsk_0")
            self.assertEqual(grg.get_individual_id(3), "tsk_25")

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
