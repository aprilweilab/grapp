"""BOLT-LMM-inf output formatting and top-level driver.

The math engine and the GRG-facing numeric computation (per-variant stats,
association statistics) live in ``grapp.assoc.bolt_inf_core``. This module holds the
optional DataFrame annotation step (which needs per-variant ``get_mutation_by_id``
metadata lookups) plus the ``bolt_lmm_inf`` orchestrator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from grapp.grg_calculator import GRGCalcInterface, _wrap_grg
from grapp.assoc.bolt_inf_core import (
    BOLT_RANDOM_SEED,
    DEFAULT_NUM_CALIB_SNPS,
    DEFAULT_H2_EST_MC_TRIALS,
    DEFAULT_CG_TOL,
    DEFAULT_MAX_ITERS,
    DEFAULT_PVALUE_METHOD,
    BoltChromInfStats,
    BoltVariantStats,
    BoltVariantStatsArray,
    BoltLmmOps,
    CalibrationResult,
    CgStats,
    CovariateBasis,
    VarianceFit,
    calibrate_lmm_inf,
    compute_bolt_variant_stats,
    compute_lmm_inf_stats,
    detect_cupy_backend,
    fit_bolt_variance_components,
    summarize_chisq,
    _nvtx,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataFrame annotation (optional, slower step)
# ---------------------------------------------------------------------------

def lmm_inf_stats_to_dataframe(
    chrom_stats: List[BoltChromInfStats],
    chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
) -> pd.DataFrame:
    """
    Convert fast ``BoltChromInfStats`` into the standard BOLT-LMM output DataFrame.

    This is the optional, slower step: it performs the per-variant
    ``grg.get_mutation_by_id`` metadata lookups to build ``SNP_ID``, ``BP``,
    ``ALLELE1``, ``ALLELE0`` and assembles the canonical columns.
    """
    grg_by_chrom = {chrom: grg for chrom, grg in chrom_grgs}

    frames: List[pd.DataFrame] = []
    for cs in chrom_stats:
        chrom = cs.chrom
        grg = grg_by_chrom[chrom]

        bp, allele1, allele0, snp_id = [], [], [], []
        for local_idx in cs.local_idx.tolist():
            mut = grg.get_mutation_by_id(local_idx)
            bp.append(mut.position)
            allele1.append(mut.allele)
            allele0.append(mut.ref_allele)
            snp_id.append(f"{chrom}:{mut.position}:{mut.allele}:{mut.ref_allele}")

        frames.append(pd.DataFrame({
            "SNP_ID": snp_id, "CHROM": chrom, "BP": bp,
            "ALLELE1": allele1, "ALLELE0": allele0, "A1FREQ": cs.a1freq,
            "CHISQ_LINREG": cs.chisq_linreg, "P_LINREG": cs.p_linreg,
            "BETA": cs.beta, "SE": cs.se,
            "CHISQ_BOLT_LMM_INF": cs.chisq_lmm_inf, "P_BOLT_LMM_INF": cs.p_lmm_inf,
        }))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def bolt_lmm_inf(
    chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
    y: np.ndarray,
    covariates: CovariateBasis,
    *,
    num_calib_snps: int = DEFAULT_NUM_CALIB_SNPS,
    mc_trials: int = DEFAULT_H2_EST_MC_TRIALS,
    cg_tol: float = DEFAULT_CG_TOL,
    max_iter: int = DEFAULT_MAX_ITERS,
    seed: int = BOLT_RANDOM_SEED,
    threads: int = 1,
    batched_apply_x: bool = False,
    pvalue_method: str = DEFAULT_PVALUE_METHOD,
) -> Tuple[VarianceFit, CalibrationResult, Dict, List[BoltChromInfStats]]:
    """
    Run BOLT-LMM-inf on one or more chromosomes.

    :param chrom_grgs: List of (chromosome_label, GRGCalcInterface), one per chromosome.
    :param y: Phenotype vector of length n_individuals.
    :param covariates: Orthonormal covariate basis (includes intercept).
    :param batched_apply_x: If True, generate the MC genetic probes with a single
        batched matmat over all trials; if False (default), loop one probe column
        per call. Both go through the Multi-X operator.
    :returns: ``(VarianceFit, CalibrationResult, residuals_dict, stats)`` where
        ``stats`` is a list of ``BoltChromInfStats`` (the fast numeric output).
        Pass it to :func:`lmm_inf_stats_to_dataframe` to get the annotated
        BOLT-LMM DataFrame.
    """
    cg_stats = CgStats()

    _y_arr = np.asarray(y, dtype=np.float64)
    logger.info(
        "Phenotype: N=%d mean=%.6g std=%.6g",
        _y_arr.size, float(_y_arr.mean()), float(_y_arr.std()),
    )

    # Detect the CuPy/NumPy backend once and pass it to every consumer.
    use_cupy = detect_cupy_backend(chrom_grgs[0][1])

    # Compute per-variant stats for each chromosome
    chrom_all_stats: List[BoltVariantStatsArray] = []
    t0 = time.perf_counter()
    with _nvtx("bolt:variant_stats"):
        grgs = [grg for _, grg in chrom_grgs]
        scheduler = _wrap_grg(chrom_grgs[0][1]).make_scheduler(grgs, threads)
        futures = [scheduler.submit(grg, compute_bolt_variant_stats, grg, covariates, grg.num_individuals, use_cupy) for _, grg in chrom_grgs]
        for future in futures:
            stats = future.result()
            chrom_all_stats.append(stats)
    logger.info("Time for computing variant statistics = %.2f sec", time.perf_counter() - t0)

    # Build ops
    t0 = time.perf_counter()
    with _nvtx("bolt:ops_setup"):
        ops = BoltLmmOps(chrom_grgs, chrom_all_stats, covariates, threads=threads, use_cupy=use_cupy).setup()
    logger.info(
        "Individuals N=%d, model SNPs M=%d across %d chroms", ops.n, ops.m_proj, len(ops.chroms),
    )
    logger.info(
        "Model SNPs per chrom: %s",
        ", ".join(f"{chrom}:{ops.m_proj_by_chrom.get(chrom, 0)}" for chrom in ops.chroms),
    )
    logger.info("Time for BoltLmmOps setup = %.2f sec", time.perf_counter() - t0)

    # Fit variance components (variance fitting CG uses 10x looser tolerance,
    # matching grg-spmv convention — secant search is insensitive to CG precision)
    t0 = time.perf_counter()
    with _nvtx("bolt:variance_fit"):
        fit = fit_bolt_variance_components(
            ops, y,
            mc_trials=mc_trials,
            seed=seed,
            batched_apply_x=batched_apply_x,
            rel_tol=10.0 * cg_tol,
            max_iter=max_iter,
            stats=cg_stats,
        )
    logger.info("Time for fitting variance components = %.2f sec", time.perf_counter() - t0)

    # LOCO + calibration
    residuals: Dict[Any, Any] = {}
    t0 = time.perf_counter()
    with _nvtx("bolt:calibration"):
        calibration = calibrate_lmm_inf(
            ops, y, residuals,
            fit=fit,
            count=num_calib_snps,
            seed=seed,
            rel_tol=cg_tol,
            max_iter=max_iter,
            stats=cg_stats,
        )
    logger.info("Time for LOCO calibration = %.2f sec", time.perf_counter() - t0)

    # Compute association statistics (fast numeric path; caller converts to a
    # DataFrame via lmm_inf_stats_to_dataframe when annotated output is wanted).
    t0 = time.perf_counter()
    with _nvtx("bolt:assoc_stats"):
        stats = compute_lmm_inf_stats(
            ops, chrom_grgs, chrom_all_stats, y, residuals, fit, calibration,
            pvalue_method=pvalue_method,
        )
    logger.info("Time for computing assoc stats = %.2f sec", time.perf_counter() - t0)

    summary = summarize_chisq(stats)
    logger.info(
        "Mean LINREG: %.6g (%d good SNPs)   lambdaGC: %.6g",
        summary["linreg"]["mean"], summary["linreg"]["n_good"], summary["linreg"]["lambda_gc"],
    )
    logger.info(
        "Mean BOLT_LMM_INF: %.6g (%d good SNPs)   lambdaGC: %.6g",
        summary["lmm_inf"]["mean"], summary["lmm_inf"]["n_good"], summary["lmm_inf"]["lambda_gc"],
    )

    return fit, calibration, residuals, stats
