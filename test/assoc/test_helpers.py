from grapp.assoc import read_pheno, read_plink_covariates
import numpy
import os
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestAssocHelpers(unittest.TestCase):
    def test_read_covars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cv_file = os.path.join(tmpdir, "covars.txt")
            with self.assertRaises(FileNotFoundError):
                read_plink_covariates(cv_file)

            # Header
            with open(cv_file, "w") as fout:
                fout.write("""FID IID SITE AGE DOB BMI SMOKE
0 tsk_0 11.1 2.1 3.0 100 99.999999
0 tsk_1 12.1 22.1 23.0 120 19.191919
""")
            C, indivs = read_plink_covariates(cv_file, return_indivs=True)
            self.assertEqual(indivs, ["tsk_0", "tsk_1"])
            numpy.testing.assert_allclose(C[0, :], [11.1, 2.1, 3.0, 100, 99.999999])
            numpy.testing.assert_allclose(C[1, :], [12.1, 22.1, 23.0, 120, 19.191919])

            # No header
            with open(cv_file, "w") as fout:
                fout.write("""0 tsk_9 3.14
IGNORED tsk_2 54.123
""")
            C, indivs = read_plink_covariates(cv_file, return_indivs=True)
            self.assertEqual(indivs, ["tsk_9", "tsk_2"])
            numpy.testing.assert_allclose(C[0, :], [3.14])
            numpy.testing.assert_allclose(C[1, :], [54.123])

    def test_read_pheno(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            phen_file = os.path.join(tmpdir, "pheno.txt")
            with self.assertRaises(FileNotFoundError):
                read_pheno(phen_file)

            # Plink-style Header
            with open(phen_file, "w") as fout:
                fout.write("""FID IID PHENOTYPE1
0 tsk_0 1.109
0 tsk_1 2.72
""")
            y, indivs = read_pheno(phen_file, return_indivs=True)
            self.assertEqual(indivs, ["tsk_0", "tsk_1"])
            numpy.testing.assert_allclose(y, [1.109, 2.72])

            # GRG-style Header
            with open(phen_file, "w") as fout:
                fout.write("""person_id      phenotypes
tsk_0     1.109
tsk_1 2.72
""")
            y2, indivs2 = read_pheno(phen_file, return_indivs=True)
            self.assertEqual(indivs, indivs2)
            numpy.testing.assert_allclose(y, y2)

            # GRG-style Header 2
            with open(phen_file, "w") as fout:
                fout.write("""individual_id      phenotypes
tsk_0     1.109
tsk_1 2.72
""")
            y3, indivs3 = read_pheno(phen_file, return_indivs=True)
            self.assertEqual(indivs, indivs3)
            numpy.testing.assert_allclose(y, y3)

            # No header
            with open(phen_file, "w") as fout:
                fout.write("""0 tsk_9 3.14
IGNORED tsk_2 54.123
""")
            y, indivs = read_pheno(phen_file, return_indivs=True)
            self.assertEqual(indivs, ["tsk_9", "tsk_2"])
            numpy.testing.assert_allclose(y, [3.14, 54.123])

            # No header NA indivs
            with open(phen_file, "w") as fout:
                fout.write("""0 NA 3.14
IGNORED NA 54.123
""")
            y, indivs = read_pheno(phen_file, return_indivs=True)
            self.assertTrue(all(map(numpy.isnan, indivs)))
            numpy.testing.assert_allclose(y, [3.14, 54.123])

            # Single column
            with open(phen_file, "w") as fout:
                fout.write("""3.14
54.123
""")
            y, indivs = read_pheno(phen_file, return_indivs=True)
            self.assertEqual(indivs, ["NA", "NA"])
            numpy.testing.assert_allclose(y, [3.14, 54.123])
