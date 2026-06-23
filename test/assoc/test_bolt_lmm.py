import os
import unittest

import numpy as np
import pandas as pd

from grapp.assoc import read_pheno
from grapp.assoc.bolt_lmm import bolt_lmm_inf, lmm_inf_stats_to_dataframe
from grapp.assoc.bolt_inf_core import CovariateBasis
from grapp.grg_calculator import load_grg_calculator

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")

# Committed fixture GRGs (bolt.chr{c}.grg): the EUR_clean modelSnps for chr20/21/22,
# all 369 individuals, 10,539 SNPs total. The two scenarios share these GRGs:
#   - no-missing: bolt.pheno.txt (real phenotype) vs bolt.truth.tsv (grapp cuSPARSE)
#   - missing:    bolt.miss.pheno.txt (55 NA) vs bolt.miss.truth.tsv (C++ BOLT, Nused=314)
# In the missing case grapp must drop the 55 NA individuals from the whole analysis
# (matching the C++ BOLT), recomputing every quantity over Nused=314.
CHROMS = [20, 21, 22]
SEED = 2026

MERGE_KEYS = ["CHROM", "BP", "ALLELE1", "ALLELE0"]

# Per-column median-relative tolerance, shared by both scenarios.
#  - A1FREQ is RNG-independent and computed over the kept individuals, so it stays
#    tight (1e-3): the cheap guard that a data/allele-frequency regression (or wrong
#    individual-removal in the missing case) is caught.
#  - LINREG / SE are deterministic regression quantities: a few percent.
#  - BETA / CHISQ_BOLT_LMM_INF / P_BOLT_LMM_INF depend on the MC heritability fit and
#    calibration draw; grapp's numpy RNG diverges from BOLT's Boost RNG (and the
#    missing case amplifies it at N=314), so they get a looser 1e-1. These bounds are
#    intentionally loose enough that the no-missing case (grgl vs cuSPARSE) also
#    passes comfortably with the same numbers.
REL_TOL = {
    "A1FREQ": 1e-3,
    "CHISQ_LINREG": 5e-2,
    "P_LINREG": 5e-2,
    "SE": 5e-2,
    "BETA": 1e-1,
    "CHISQ_BOLT_LMM_INF": 1e-1,
    "P_BOLT_LMM_INF": 1e-1,
}
ABS_TOL = 5e-2  # median absolute error, all columns


class TestBoltLmmInf(unittest.TestCase):
    """grgl-backend regression tests for BOLT-LMM-inf (with and without missingness)."""

    @classmethod
    def setUpClass(cls):
        cls.grgs = [
            load_grg_calculator(os.path.join(INPUT_DIR, f"bolt.chr{c}.grg"))
            for c in CHROMS
        ]
        cls.chrom_grgs = list(zip(CHROMS, cls.grgs))
        cls.n = cls.grgs[0].num_individuals

    def _run_against_truth(self, pheno_file, truth_file, expected_missing):
        # read_pheno maps the NA missing token to NaN; the driver drops those
        # individuals (Nused = n - expected_missing) and rebuilds the intercept basis.
        y = read_pheno(os.path.join(INPUT_DIR, pheno_file))
        self.assertEqual(y.shape[0], self.n)
        self.assertEqual(
            int(np.isnan(y).sum()),
            expected_missing,
            f"{pheno_file}: expected {expected_missing} missing, got {int(np.isnan(y).sum())}",
        )
        cov = CovariateBasis.intercept_only(self.n)

        fit, cal, _, stats = bolt_lmm_inf(
            self.chrom_grgs,
            y,
            cov,
            seed=SEED,
            threads=1,
        )
        df = lmm_inf_stats_to_dataframe(stats, self.chrom_grgs)

        truth = pd.read_csv(os.path.join(INPUT_DIR, truth_file), sep="\t")
        merged = df.merge(truth, on=MERGE_KEYS, suffixes=("_got", "_exp"))
        self.assertEqual(len(df), len(truth), "row count mismatch")
        self.assertEqual(len(merged), len(truth), "variant-key merge dropped rows")

        for col, rel_tol in REL_TOL.items():
            got = merged[f"{col}_got"].astype(float).to_numpy()
            exp = merged[f"{col}_exp"].astype(float).to_numpy()
            # NaN must appear in exactly the same variants in both.
            np.testing.assert_array_equal(
                np.isnan(got), np.isnan(exp), err_msg=f"NaN-position mismatch in {col}"
            )
            mask = ~(np.isnan(got) | np.isnan(exp))
            abs_err = np.abs(got[mask] - exp[mask])
            rel_err = abs_err / (np.abs(exp[mask]) + 1e-12)
            # Median (not max/p99): near-zero BETA/CHISQ rows make relative error
            # meaningless in the tail, so a robust central statistic is used.
            med_abs = float(np.median(abs_err))
            med_rel = float(np.median(rel_err))
            self.assertLess(
                med_abs, ABS_TOL, f"{col}: median abs err {med_abs:.3e} >= {ABS_TOL}"
            )
            self.assertLess(
                med_rel, rel_tol, f"{col}: median rel err {med_rel:.3e} >= {rel_tol}"
            )

        # Sanity on the fitted variance-component scalars.
        self.assertTrue(0.0 <= fit.h2 <= 1.0, f"h2 out of range: {fit.h2}")
        self.assertGreater(
            cal.factor, 0.0, f"calibration factor not positive: {cal.factor}"
        )

    def test_no_missing(self):
        self._run_against_truth("bolt.pheno.txt", "bolt.truth.tsv", expected_missing=0)

    def test_missing(self):
        self._run_against_truth(
            "bolt.miss.pheno.txt", "bolt.miss.truth.tsv", expected_missing=55
        )


if __name__ == "__main__":
    unittest.main()
