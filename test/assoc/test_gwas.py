import sys
import unittest
import os
import pandas as pd
import subprocess
import pygrgl
from grapp.assoc import linear_assoc_no_covar, read_pheno
from io import StringIO

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestGWAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.gwas.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename)
        cls.pheno_path = os.path.join(INPUT_DIR, "phenotypes.txt")
        assert os.path.isfile(cls.pheno_path)
        output = subprocess.check_output(
            [
                "grg",
                "process",
                "gwas",
                cls.grg_filename,
                "--phenotype",
                cls.pheno_path,
            ],
            text=True,
        )
        cls.gwas = pd.read_csv(StringIO(output), delim_whitespace=True)
        # Load GRG and data
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename)

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

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
