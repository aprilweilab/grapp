import os
import unittest
import pytest

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


def _assert_bolt_comparison(case, df, truth, fit, cal):
    merged = df.merge(truth, on=MERGE_KEYS, suffixes=("_got", "_exp"))
    case.assertEqual(len(df), len(truth), "row count mismatch")
    case.assertEqual(len(merged), len(truth), "variant-key merge dropped rows")

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
        case.assertLess(
            med_abs, ABS_TOL, f"{col}: median abs err {med_abs:.3e} >= {ABS_TOL}"
        )
        case.assertLess(
            med_rel, rel_tol, f"{col}: median rel err {med_rel:.3e} >= {rel_tol}"
        )

    # Sanity on the fitted variance-component scalars.
    case.assertTrue(0.0 <= fit.h2 <= 1.0, f"h2 out of range: {fit.h2}")
    case.assertGreater(
        cal.factor, 0.0, f"calibration factor not positive: {cal.factor}"
    )

@pytest.mark.human_data
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

    def _run_against_truth(self, pheno_file, truth_file, expected_missing, cov=None):
        # read_pheno maps the NA missing token to NaN; the driver drops those
        # individuals (Nused = n - expected_missing) and rebuilds the intercept basis.
        y = read_pheno(os.path.join(INPUT_DIR, pheno_file))
        self.assertEqual(y.shape[0], self.n)
        self.assertEqual(
            int(np.isnan(y).sum()),
            expected_missing,
            f"{pheno_file}: expected {expected_missing} missing, got {int(np.isnan(y).sum())}",
        )
        if cov is None:
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
        _assert_bolt_comparison(self, df, truth, fit, cal)

    def test_no_missing(self):
        self._run_against_truth("bolt.pheno.txt", "bolt.truth.tsv", expected_missing=0)

    def test_missing(self):
        self._run_against_truth(
            "bolt.miss.pheno.txt", "bolt.miss.truth.tsv", expected_missing=55
        )

    def test_missing_with_real_covar_runs(self):
        # End-to-end smoke for the bolt runs with covariate and missingness
        y = read_pheno(os.path.join(INPUT_DIR, "bolt.miss.pheno.txt"))
        self.assertEqual(int(np.isnan(y).sum()), 55)
        rng = np.random.default_rng(SEED)
        C_full = np.column_stack([np.ones(self.n), rng.standard_normal((self.n, 2))])
        cov = CovariateBasis.from_matrix(
            C_full, covar_cols=(), q_covar_cols=("x", "y"), covar_max_levels=10
        )
        self.assertEqual(cov.cindep, 3)

        fit, cal, _, stats = bolt_lmm_inf(self.chrom_grgs, y, cov, seed=SEED, threads=1)
        df = lmm_inf_stats_to_dataframe(stats, self.chrom_grgs)

        self.assertGreater(len(df), 0)
        beta = df["BETA"].astype(float).to_numpy()
        chisq = df["CHISQ_BOLT_LMM_INF"].astype(float).to_numpy()
        self.assertTrue(
            np.isfinite(beta).all(), "non-finite BETA with covariates+missing"
        )
        self.assertTrue(
            np.isfinite(chisq).all(), "non-finite CHISQ with covariates+missing"
        )
        self.assertTrue(0.0 <= fit.h2 <= 1.0, f"h2 out of range: {fit.h2}")
        self.assertGreater(cal.factor, 0.0, f"calibration not positive: {cal.factor}")


@pytest.fixture(scope="module")
def simulated_bolt_inputs(tmp_path_factory):
    import sys
    from pathlib import Path
    import pygrgl
    from pygrgl_spmv import simple_convert
    from grapp.util.filter import grg_save_range
    from grg_pheno_sim.multi_grg_phenotype import sim_phenotypes_multi_grg

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from testing_utils import construct_grg

    directory = tmp_path_factory.mktemp("bolt-simulated")
    source = construct_grg("test-200-samples.vcf.gz", str(directory / "simulated.grg"))
    raw = pygrgl.load_immutable_grg(source, load_up_edges=True)
    positions = sorted(raw.get_mutation_by_id(j).position for j in range(raw.num_mutations))
    edges = [positions[0], positions[len(positions)//3], positions[2*len(positions)//3], positions[-1]+1]
    assert edges == [55829, 3345382, 6712483, 9999127]
    paths = []
    for chrom, lo, hi in zip(CHROMS, edges[:-1], edges[1:]):
        path = directory / f"chr{chrom}.grg"
        grg_save_range(raw, str(path), (lo, hi))
        paths.append(path)
    raw_parts = [pygrgl.load_immutable_grg(str(path)) for path in paths]
    assert sum(g.num_mutations for g in raw_parts) == 10893
    assert all(g.num_individuals == 200 for g in raw_parts)
    artifacts = [simple_convert(path, path.with_suffix(".grg_spmv")) for path in paths]
    frame = sim_phenotypes_multi_grg([str(p) for p in paths], load_all_ram=True,
                                    heritability=.6, standardized=True, random_seed=42)
    frame.to_csv(directory / "phenotypes.tsv", sep="\t", index=False)
    return raw_parts, artifacts, frame["phenotype"].to_numpy()


@pytest.fixture(scope="module", params=["complete", "missing", "covariates"])
def simulated_bolt_scenario(request, simulated_bolt_inputs):
    from grapp.grg_calculator import GRGCalculator

    raw, artifacts, phenotype = simulated_bolt_inputs
    y = phenotype.copy()
    if request.param != "complete":
        y[:30] = np.nan
    cov = CovariateBasis.intercept_only(len(y))
    if request.param == "covariates":
        matrix = np.column_stack((np.ones(len(y)), np.random.default_rng(2026).normal(size=(len(y), 2))))
        cov = CovariateBasis.from_matrix(matrix, covar_cols=(), q_covar_cols=("x", "y"), covar_max_levels=10)
    chromosomes = [(c, GRGCalculator(g)) for c, g in zip(CHROMS, raw)]
    fit, cal, _, stats = bolt_lmm_inf(chromosomes, y, cov, seed=SEED, threads=1)
    truth = lmm_inf_stats_to_dataframe(stats, chromosomes)
    assert 0 <= fit.h2 <= 1 and cal.factor > 0
    return artifacts, y, cov, truth


@pytest.mark.parametrize("backend,native,device", [
    pytest.param("reference", False, 0, marks=pytest.mark.reference),
    pytest.param("mkl", False, 0, marks=pytest.mark.mkl),
    *[pytest.param(b, native, 0, marks=getattr(pytest.mark, b))
      for b in ("cusparse", "rocsparse") for native in (False, True)],
    pytest.param("cusparse", True, 1, marks=pytest.mark.cusparse, id="cusparse-native-device1"),
])
def test_simulated_bolt_matches_grgl(simulated_bolt_scenario, backend, native, device):
    from grapp.grg_calculator import GRGSpMVCalculator
    from pygrgl_spmv import testing, make_runconfig_bolt

    if device:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("needs at least two GPU devices")
        assert cupy.cuda.runtime.getDevice() == 0
    artifacts, y, cov, truth = simulated_bolt_scenario
    kwargs = dict(capture=True, native=native, device=device) if backend in {"cusparse", "rocsparse"} else {}
    with testing.load_many(artifacts, backend=backend, req=make_runconfig_bolt(), **kwargs) as grgs:
        chromosomes = [(c, GRGSpMVCalculator(g)) for c, g in zip(CHROMS, grgs)]
        fit, cal, _, stats = bolt_lmm_inf(chromosomes, y, cov, seed=SEED, threads=1)
        frame = lmm_inf_stats_to_dataframe(stats, chromosomes)
        if device:
            assert cupy.cuda.runtime.getDevice() == 0
    _assert_bolt_comparison(unittest.TestCase(), frame, truth, fit, cal)


if __name__ == "__main__":
    unittest.main()
