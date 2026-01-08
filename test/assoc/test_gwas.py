import sys
import unittest
import os
import pandas as pd
import pygrgl
import numpy
from grapp.assoc import linear_assoc_no_covar, linear_assoc_covar, read_pheno
from grapp.linalg import PCs

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestGWAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.gwas.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)
        cls.pheno_path = os.path.join(INPUT_DIR, "phenotypes.txt")
        assert os.path.isfile(cls.pheno_path)

        baseline_path = os.path.join(INPUT_DIR, "gwas.baseline.txt")
        cls.gwas = pd.read_csv(baseline_path, delimiter="\t")

        covar_baseline_path = os.path.join(INPUT_DIR, "covar.baseline.txt")
        cls.covar = pd.read_csv(covar_baseline_path, dtype=numpy.float64)

        cls.Y = read_pheno(cls.pheno_path)

    def test_gwas_no_covar_vs_grg(self):
        df_py = linear_assoc_no_covar(self.grg, self.Y)
        df_grg = self.gwas

        # Check same number of rows
        self.assertEqual(
            len(df_py),
            len(df_grg),
            f"Mismatch in row count: Python has {len(df_py)}, GRG has {len(df_grg)}",
        )

        # Columns to compare
        columns = ["BETA", "SE", "T", "P"]
        atol = 1e-4
        rtol = 1e-3

        for i in range(len(df_py)):
            for col in columns:
                val_py = df_py.iloc[i][col]
                val_grg = df_grg.iloc[i][col]
                diff = abs(val_py - val_grg)
                rel_err = diff / (abs(val_grg) + 1e-8)
                if diff > atol and rel_err > rtol:
                    self.fail(
                        f"Row {i}, column '{col}' mismatch:\n"
                        f"Python: {val_py}, GRG: {val_grg}, "
                        f"abs_diff={diff}, rel_err={rel_err}"
                    )

    def test_gwas_covar(self):
        C = PCs(self.grg, 10, unitvar=False)
        df_nonstd = linear_assoc_covar(self.grg, self.Y, C)
        # Compare against the baseline of known "good" values.
        numpy.testing.assert_allclose(df_nonstd, self.covar)

        df_std = linear_assoc_covar(self.grg, self.Y, C, standardize=True)
        self.assertEqual(len(df_nonstd), len(df_std))
        self.assertEqual(self.grg.num_mutations, len(df_std))

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
