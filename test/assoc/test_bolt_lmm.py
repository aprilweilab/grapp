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

# Chromosomes of the committed fixture GRGs (bolt.chr{c}.grg). The phenotype is the
# real one in bolt.pheno.txt (read in file order to align with GRG individual order).
# SEED is the variance-component / calibration RNG seed; it MUST match how the
# reference bolt.truth.tsv was produced.
CHROMS = [20, 21, 22]
SEED = 2026

# Merge key uniquely identifying a variant across grapp output and the truth.
MERGE_KEYS = ["CHROM", "BP", "ALLELE1", "ALLELE0"]

# Mixed-model + data columns compared at the median<5e-2 (abs and rel) threshold.
COMPARE_COLS = [
    "A1FREQ",
    "CHISQ_LINREG",
    "P_LINREG",
    "BETA",
    "SE",
    "CHISQ_BOLT_LMM_INF",
    "P_BOLT_LMM_INF",
]
# These do not depend on the variance-component (h2) fit, so they match the
# reference to ~1e-6 regardless of backend; guard them tighter to catch any
# data/LINREG-pipeline regression.
TIGHT_COLS = ["A1FREQ", "CHISQ_LINREG"]

MEDIAN_TOL = 5e-2
TIGHT_MEDIAN_REL_TOL = 1e-3


class TestBoltLmmInf(unittest.TestCase):
    """grgl-backend regression test for BOLT-LMM-inf against a committed truth."""

    @classmethod
    def setUpClass(cls):
        cls.grgs = [
            load_grg_calculator(os.path.join(INPUT_DIR, f"bolt.chr{c}.grg"))
            for c in CHROMS
        ]
        cls.chrom_grgs = list(zip(CHROMS, cls.grgs))

        n = cls.grgs[0].num_individuals
        # Real phenotype matching the fixture individuals (intercept-only model).
        # read_pheno takes the last column in file order, which is the GRG/FAM order.
        cls.y = read_pheno(os.path.join(INPUT_DIR, "bolt.pheno.txt"))
        assert cls.y.shape[0] == n, f"pheno length {cls.y.shape[0]} != num_individuals {n}"
        cls.cov = CovariateBasis.intercept_only(n)

        truth_path = os.path.join(INPUT_DIR, "bolt.truth.tsv")
        assert os.path.isfile(truth_path)
        cls.truth = pd.read_csv(truth_path, sep="\t")

    def test_bolt_lmm_inf_grgl_vs_truth(self):
        fit, cal, _, stats = bolt_lmm_inf(
            self.chrom_grgs,
            self.y,
            self.cov,
            seed=SEED,
            threads=1,
        )
        df = lmm_inf_stats_to_dataframe(stats, self.chrom_grgs)

        merged = df.merge(self.truth, on=MERGE_KEYS, suffixes=("_got", "_exp"))
        self.assertEqual(
            len(df),
            len(self.truth),
            f"row count mismatch: grapp {len(df)} vs truth {len(self.truth)}",
        )
        self.assertEqual(
            len(merged),
            len(self.truth),
            f"variant-key merge dropped rows: merged {len(merged)} vs truth {len(self.truth)}",
        )

        for col in COMPARE_COLS:
            got = merged[f"{col}_got"].astype(float).to_numpy()
            exp = merged[f"{col}_exp"].astype(float).to_numpy()

            # NaN must appear in exactly the same variants in both.
            np.testing.assert_array_equal(
                np.isnan(got), np.isnan(exp), err_msg=f"NaN-position mismatch in {col}"
            )

            mask = ~(np.isnan(got) | np.isnan(exp))
            a = got[mask]
            e = exp[mask]
            abs_err = np.abs(a - e)
            rel_err = abs_err / (np.abs(e) + 1e-12)

            # Median (not max/p99): near-zero BETA/CHISQ rows make relative error
            # meaningless in the tail, so a robust central statistic is used.
            med_abs = float(np.median(abs_err))
            med_rel = float(np.median(rel_err))

            self.assertLess(
                med_abs, MEDIAN_TOL, f"{col}: median abs err {med_abs:.3e} >= {MEDIAN_TOL}"
            )
            self.assertLess(
                med_rel, MEDIAN_TOL, f"{col}: median rel err {med_rel:.3e} >= {MEDIAN_TOL}"
            )

            if col in TIGHT_COLS:
                self.assertLess(
                    med_rel,
                    TIGHT_MEDIAN_REL_TOL,
                    f"{col}: median rel err {med_rel:.3e} >= {TIGHT_MEDIAN_REL_TOL} "
                    f"(data/LINREG layer should match the reference closely)",
                )

        # Sanity on the fitted variance-component scalars.
        self.assertTrue(0.0 <= fit.h2 <= 1.0, f"h2 out of range: {fit.h2}")
        self.assertGreater(cal.factor, 0.0, f"calibration factor not positive: {cal.factor}")


if __name__ == "__main__":
    unittest.main()
