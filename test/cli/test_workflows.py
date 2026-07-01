"""
Test some end-to-end workflows, e.g. generating phenotypes and covariates from grapp and then using
them for grapp GWAS.
"""

import os
import pandas
import pygrgl
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, grapp_run

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestCLIWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg(
            "test-200-samples.vcf.gz", "test.cli_workflows.grg"
        )

    def test_pheno_to_gwas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pheno_file = os.path.join(tmpdir, "test.phen")
            print(grapp_run("pheno", "-o", pheno_file, self.grg_filename))
            self.assertTrue(os.path.isfile(pheno_file))

            gwas_results = os.path.join(tmpdir, "gwas.out.tsv")
            print(
                grapp_run(
                    "assoc", "-p", pheno_file, "-o", gwas_results, self.grg_filename
                )
            )
            self.assertTrue(os.path.isfile(gwas_results))
            gwas_df = pandas.read_csv(gwas_results, delimiter="\t")
            self.assertEqual(
                gwas_df.columns.to_list(),
                ["POS", "ALT", "REF", "COUNT", "BETA", "B0", "SE", "R2", "T", "P"],
            )

    def test_pheno_and_covar_to_gwas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pheno_file = os.path.join(tmpdir, "test.phen")
            print(grapp_run("pheno", "-o", pheno_file, self.grg_filename))
            self.assertTrue(os.path.isfile(pheno_file))

            cov_file = os.path.join(tmpdir, "test.cov.tsv")
            print(grapp_run("pca", "-o", cov_file, self.grg_filename))
            self.assertTrue(os.path.isfile(cov_file))

            gwas_results = os.path.join(tmpdir, "gwas.out.tsv")
            print(
                grapp_run(
                    "assoc",
                    "-p",
                    pheno_file,
                    "-c",
                    cov_file,
                    "-o",
                    gwas_results,
                    self.grg_filename,
                )
            )
            self.assertTrue(os.path.isfile(gwas_results))
            gwas_df = pandas.read_csv(gwas_results, delimiter="\t")
            # Covariates analysis does not have the intercept (intercept is always 1)
            self.assertEqual(
                gwas_df.columns.to_list(),
                ["POS", "ALT", "REF", "COUNT", "BETA", "SE", "T", "P"],
            )

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
