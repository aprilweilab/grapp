import sys
from pathlib import Path
import unittest
import os
import pandas as pd
import numpy as np
import subprocess
import pygrgl
from grgapp.assoc import linear_assoc_no_covar, read_pheno, compute_XtX
from typing import List, Dict, Tuple, Optional
from io import StringIO

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

INPUT_DIR = os.path.join(THIS_DIR, "input")


def construct_grg(input_file: str, output_file: Optional[str] = None) -> str:
    cmd = [
        "grg",
        "construct",
        "-p",
        "10",
        "-t",
        "2",
        "-j",
        str(JOBS),
        os.path.join(INPUT_DIR, input_file),
    ]
    if output_file is not None:
        cmd.extend(["-o", output_file])
    else:
        output_file = os.path.basename(input_file) + ".final.grg"
    subprocess.check_call(cmd)
    return output_file


class TestGWAS(unittest.TestCase):
    def setUp(self):
        # Paths to test data
        self.grg_filename = construct_grg("test-200-samples.vcf.gz")
        self.pheno_path = os.path.join(INPUT_DIR, "phenotypes.txt")
        output = subprocess.check_output(
            [
                "grg",
                "process",
                "gwas",
                self.grg_filename,
                "--phenotype",
                self.pheno_path,
            ],
            text=True,
        )
        self.gwas = pd.read_csv(StringIO(output), delim_whitespace=True)
        # Load GRG and data
        self.grg = pygrgl.load_immutable_grg(self.grg_filename)

        self.Y = read_pheno(self.pheno_path)

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
        if CLEANUP:
            os.remove(self.grg_filename)
