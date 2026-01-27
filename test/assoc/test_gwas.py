import math
import numpy
import os
import pandas as pd
import pygrgl
import sys
import unittest
from collections import defaultdict
from grapp.assoc import linear_assoc_no_covar, linear_assoc_covar, read_pheno
from grapp.linalg import PCs
from grapp.util.filter import grg_save_samples

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, complete_sample_sets

CLEANUP = True
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestGWAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.gwas.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=True)
        cls.pheno_path = os.path.join(INPUT_DIR, "phenotypes.txt")
        assert os.path.isfile(cls.pheno_path)

        baseline_path = os.path.join(INPUT_DIR, "gwas.baseline.txt")
        cls.gwas = pd.read_csv(baseline_path, delimiter="\t")

        covar_baseline_path = os.path.join(INPUT_DIR, "covar.baseline.txt")
        cls.covar = pd.read_csv(covar_baseline_path, delimiter="\t")

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
        columns = ["COUNT", "BETA", "SE", "T", "P"]
        atol = 1e-4
        rtol = 1e-3

        for i in range(len(df_py)):
            acount = df_py.iloc[i]["COUNT"]
            should_be_nan = acount == 0 or acount == self.grg.num_samples
            for col in columns:
                val_py = df_py.iloc[i][col]
                val_grg = df_grg.iloc[i][col]
                diff = abs(val_py - val_grg)
                rel_err = diff / (abs(val_grg) + 1e-8)
                fail_msg = (
                    f"Row {i}, column '{col}' mismatch:\n"
                    f"Python: {val_py}, GRG: {val_grg}, "
                    f"abs_diff={diff}, rel_err={rel_err}"
                )
                if should_be_nan and col != "COUNT":
                    assert math.isnan(val_py), fail_msg
                    continue
                assert not math.isnan(val_py), fail_msg
                assert diff <= atol or rel_err <= rtol, fail_msg

    def test_gwas_covar(self):
        C = PCs(self.grg, 10, unitvar=False).to_numpy()
        df_nonstd = linear_assoc_covar(self.grg, self.Y, C)
        self.assertFalse(numpy.any(numpy.isinf(df_nonstd["BETA"].to_numpy())))
        # Compare against the baseline of known values. This is just a test to make sure
        # nothing changes.
        numpy.testing.assert_allclose(df_nonstd["BETA"], self.covar["BETA"])
        df_nonstd_regress = linear_assoc_covar(self.grg, self.Y, C, method="regress")

        qr_nans = df_nonstd["BETA"].isna().sum()
        reg_nans = df_nonstd_regress["BETA"].isna().sum()
        self.assertLess(abs(qr_nans - reg_nans), 10)

        # These methods are not identical. But we expect them to be "pretty close", which
        # we measure as 90% of SNPs having a relative difference of less than 0.2 between
        # the two methods.
        total = self.grg.num_mutations - max(qr_nans, reg_nans)
        pretty_close = 0
        reldiff = (df_nonstd["BETA"] - df_nonstd_regress["BETA"]).abs() / df_nonstd[
            "BETA"
        ]
        for r in reldiff:
            if not math.isnan(r) and not math.isinf(r) and r <= 0.2:
                pretty_close += 1
        self.assertGreater(pretty_close / total, 0.9)

        df_std = linear_assoc_covar(self.grg, self.Y, C, standardize=True)
        self.assertEqual(len(df_nonstd), len(df_std))
        self.assertEqual(self.grg.num_mutations, len(df_std))

    def test_gwas_no_covar_dists(self):
        """
        The binomial method and the sample method should be very similar on neutral simulated data
        of large sample size. For small sample sizes (like this test), the deviation can be quite
        large.
        """
        df_sample = linear_assoc_no_covar(self.grg, self.Y)
        df_binomial = linear_assoc_no_covar(self.grg, self.Y, dist="binomial")

        sample_nans = df_sample["BETA"].isna().sum()
        binomial_nans = df_binomial["BETA"].isna().sum()
        self.assertEqual(sample_nans, binomial_nans)

        relerr = (df_binomial["BETA"] - df_sample["BETA"]) / df_sample["BETA"]
        small_errors = numpy.where((relerr < 0.20) & (relerr > -0.20))
        total = self.grg.num_mutations - sample_nans
        small_err_proportion = len(small_errors[0]) / total
        self.assertGreater(
            small_err_proportion, 0.99
        )  # 99% of errors are less than 20% relative error

    def test_gwas_no_covar_missing_Y(self):
        """
        When there are missing phenotypes (Y), they are represented as NaN. In this case, we need to scale
        the variance term in the GWAS appropriately. The calculation should be numerically close to removing
        those individuals from the GRG first, and then performing a GWAS.
        """
        # The individual list needs to be ordered, because GRG will retain the original order when down sampling.
        keep_indivs, ignore_indivs, keep_samples, ignore_samples = complete_sample_sets(
            self.grg, [22, 33, 54, 166, 167]
        )

        # Setup the missing data in the phenotype.
        Y_miss = self.Y.copy()
        Y_miss[ignore_indivs] = math.nan
        Y_kept = self.Y[keep_indivs]

        # Create the filtered GRG by just removing the individuals with missing Y
        filt_name = "test.gwas.missingY.grg"
        grg_save_samples(self.grg, filt_name, keep_samples)
        filt_grg = pygrgl.load_immutable_grg(filt_name, load_up_edges=False)
        self.assertEqual(filt_grg.num_individuals, Y_kept.shape[0])

        # Use binomial estimates
        true_sample_df = linear_assoc_no_covar(filt_grg, Y_kept, dist="binomial")
        mask_sample_df = linear_assoc_no_covar(self.grg, Y_miss, dist="binomial")
        true_nans = true_sample_df["BETA"].isna().sum()
        mask_nans = mask_sample_df["BETA"].isna().sum()
        self.assertEqual(true_nans, mask_nans)
        numpy.testing.assert_allclose(true_sample_df["BETA"], mask_sample_df["BETA"])

        # Use sample estimates (default dist)
        true_sample_df = linear_assoc_no_covar(filt_grg, Y_kept)
        mask_sample_df = linear_assoc_no_covar(self.grg, Y_miss)
        true_nans = true_sample_df["BETA"].isna().sum()
        mask_nans = mask_sample_df["BETA"].isna().sum()
        self.assertGreaterEqual(true_nans, mask_nans)
        true_beta = true_sample_df["BETA"].to_numpy()
        mask_beta = mask_sample_df["BETA"].to_numpy()
        for i in range(true_beta.shape[0]):
            if abs((true_beta[i] - mask_beta[i]) / mask_beta[i]) > 0.25:
                print(
                    f"Mismatch @ {true_sample_df.iloc[i]}: {true_beta[i]} vs. {mask_beta[i]}"
                )
        numpy.testing.assert_allclose(true_sample_df["BETA"], mask_sample_df["BETA"])

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
