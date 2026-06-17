"""BOLT-LMM-inf math engine: dataclasses, RNGs, CG solver, BoltLmmOps, and the
variance-component / calibration algorithm steps that operate purely through
``BoltLmmOps``.

GRG-facing computation (per-variant stats, association statistics) and output
formatting live in ``grapp.assoc.bolt_lmm`` alongside the ``bolt_lmm_inf`` driver.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2 as _scipy_chi2
from scipy.special import erfc as _erfc

import pygrgl

from grapp.grg_calculator import GRGCalcInterface, GRGSpMVCalculator, _wrap_grg
from grapp.util.simple import (
    allele_counts,
    allele_frequencies,
)


DTYPE = np.dtype(np.float64)
BOLT_RANDOM_SEED = 12345
BOLT_BAD_SNP_STAT = -1e9
DEFAULT_NUM_CALIB_SNPS = 30
DEFAULT_H2_EST_MC_TRIALS = 0
DEFAULT_CG_TOL = 5e-4
DEFAULT_MAX_ITERS = 10_000

_UP = pygrgl.TraversalDirection.UP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NVTX helper
# ---------------------------------------------------------------------------

try:  # CuPy is optional; NVTX ranges no-op on the NumPy backend.
    from cupy.cuda import nvtx as _cuda_nvtx  # type: ignore
except Exception:  # pragma: no cover - depends on runtime CUDA availability
    _cuda_nvtx = None


@contextlib.contextmanager
def _nvtx(name: str):
    """Push/pop an NVTX range (no-op if CuPy/CUDA is unavailable).

    Ranges use a ``bolt:`` prefix and ``:``-delimited level naming so the
    Nsight timeline reads stage -> sub-step -> inner op.
    """
    if _cuda_nvtx is None:
        yield
        return
    _cuda_nvtx.RangePush(name)
    try:
        yield
    finally:
        _cuda_nvtx.RangePop()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> float:
    if hasattr(value, "get"):
        return float(value.get())
    return float(value)


def _array_module(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy as cp
        return cp
    return np


def _dot(left, right) -> float:
    xp = _array_module(left)
    return _as_float(xp.sum(left * right))

def _to_np(arr) -> np.ndarray:
    """Convert CuPy or NumPy array to NumPy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def detect_cupy_backend(grg) -> bool:
    """True if this GRG calculator uses the CuPy/GPU backend."""
    return bool(isinstance(grg, GRGSpMVCalculator) and getattr(grg, "use_cupy", False))

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CgStats:
    solves: int = 0
    iterations: int = 0
    max_iterations: int = 0
    max_rel_resid: float = 0.0

    def add(self, iterations: int, rel_resid: float) -> None:
        iters = int(iterations)
        self.solves += 1
        self.iterations += iters
        self.max_iterations = max(self.max_iterations, iters)
        self.max_rel_resid = max(self.max_rel_resid, float(rel_resid))


@dataclass(frozen=True)
class VarianceFit:
    log_delta: float
    sigma_g2: float
    sigma_e2: float
    h2: float
    delta: float
    all_hinv_y: Any


@dataclass(frozen=True)
class McScalingResult:
    log_delta: float
    f_jacks: tuple
    f_rands_as_data: tuple
    sigma2_k: float
    all_hinv_y: Any

    @property
    def f_reml(self) -> float:
        return float(self.f_jacks[-1])


@dataclass(frozen=True)
class CalibrationResult:
    factor: float
    std: float
    ratio_of_medians: float
    median_of_ratios: float
    selected_snps: tuple
    tried_snps: int
    vinv_scale_by_chrom: dict


# ---------------------------------------------------------------------------
# CovariateBasis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CovariateBasis:
    """BOLT-style orthonormal covariate basis, including the all-ones vector."""

    basis: np.ndarray
    covar_cols: tuple
    q_covar_cols: tuple
    covar_max_levels: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.basis, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("covariate basis must be two-dimensional")
        if arr.shape[0] < 1:
            raise ValueError("covariate basis must include at least one sample")
        object.__setattr__(self, "basis", np.ascontiguousarray(arr))

    @property
    def nused(self) -> int:
        return int(self.basis.shape[0])

    @property
    def cindep(self) -> int:
        return int(self.basis.shape[1])

    @property
    def dim(self) -> int:
        return int(self.nused - self.cindep)

    @classmethod
    def intercept_only(cls, n: int) -> "CovariateBasis":
        n_int = int(n)
        if n_int < 1:
            raise ValueError("sample count must be positive")
        basis = np.full((n_int, 1), 1.0 / math.sqrt(float(n_int)), dtype=np.float64)
        return cls(basis=basis, covar_cols=(), q_covar_cols=(), covar_max_levels=10)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        covar_cols: Sequence,
        q_covar_cols: Sequence,
        covar_max_levels: int,
    ) -> "CovariateBasis":
        covars = np.asarray(matrix, dtype=np.float64)
        if covars.ndim != 2:
            raise ValueError("covariate matrix must be two-dimensional")
        if covars.shape[0] < 1 or covars.shape[1] < 1:
            raise ValueError("covariate matrix must be non-empty")
        if covars.shape[1] > covars.shape[0]:
            raise ValueError("number of covariate columns cannot exceed sample count")
        u, s, _vt = np.linalg.svd(np.asfortranarray(covars), full_matrices=False)
        if s.size == 0 or s[0] <= 0.0:
            raise ValueError("covariate matrix is rank-deficient with no independent columns")
        rank = int(np.count_nonzero(s >= (s[0] * 1e-8)))
        return cls(
            basis=u[:, :rank],
            covar_cols=tuple(str(v) for v in covar_cols),
            q_covar_cols=tuple(str(v) for v in q_covar_cols),
            covar_max_levels=int(covar_max_levels),
        )

    def project_host(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        if mat.shape[0] != self.nused:
            raise ValueError(f"vector has {mat.shape[0]} rows; expected {self.nused}")
        projected = mat - self.basis @ (self.basis.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_host_inplace(self, values: np.ndarray) -> np.ndarray:
        values[...] = self.project_host(values)
        return values

    def project_device(self, values):
        xp = _array_module(values)
        if xp is np:
            return self.project_host(values)
        q = xp.asarray(self.basis, dtype=DTYPE)
        arr = xp.asarray(values, dtype=DTYPE)
        was_vector = arr.ndim == 1
        mat = arr.reshape((self.nused, 1)) if was_vector else arr
        projected = mat - q @ (q.T @ mat)
        return projected[:, 0] if was_vector else projected

    def project_device_inplace(self, values):
        values[...] = self.project_device(values)
        return values


# ---------------------------------------------------------------------------
# CG solver
# ---------------------------------------------------------------------------

def bolt_conj_grad_solve(
    matvecs: Sequence[Any],
    rhs_columns: Sequence[Any],
    *,
    rel_tol: float,
    max_iter: int,
    stats: Optional[CgStats] = None,
    project,
) -> List[Any]:
    """Mirrors BOLT-LMM_v2.5 Bolt::conjGradSolve.

    The full CG iteration runs natively on the array backend of the inputs (CuPy
    when device arrays are passed in, NumPy otherwise); the GRG operator
    (``matvec`` = apply_k) and ``project`` operate on the same backend.
    """
    if len(matvecs) != len(rhs_columns):
        raise ValueError("matvec and RHS counts differ")
    if not rhs_columns:
        return []
    xp = _array_module(rhs_columns[0])
    # Mirrors BOLT-LMM_v2.5 Bolt::conjGradSolve: full-batch CG, no active-column
    # mask, no denominator guard, and no exception when maxIters is reached.
    b_cols = [project(xp.asarray(rhs, dtype=DTYPE).copy()) for rhs in rhs_columns]
    b = xp.column_stack(b_cols)
    x = xp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    hp = xp.empty_like(b)
    r2_orig = xp.sum(r * r, axis=0)
    r2_old = r2_orig.copy()
    rels = xp.sqrt(r2_old / r2_orig)
    for it in range(1, int(max_iter) + 1):
        for col, matvec in enumerate(matvecs):
            matvec(p[:, col], hp[:, col])
        denom = xp.sum(p * hp, axis=0)
        alpha = r2_old / denom
        x += p * alpha.reshape((1, -1))
        r -= hp * alpha.reshape((1, -1))
        r = project(r)
        r2_new = xp.sum(r * r, axis=0)
        rels = xp.sqrt(r2_new / r2_orig)
        if not bool(xp.any(rels > float(rel_tol))):
            if stats is not None:
                rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
                for rel in rel_values:
                    stats.add(it, float(rel))
            return [project(x[:, idx].copy()) for idx in range(x.shape[1])]
        beta = r2_new / r2_old
        p *= beta.reshape((1, -1))
        p += r
        r2_old = r2_new
    if stats is not None:
        rel_values = xp.asnumpy(rels) if hasattr(xp, "asnumpy") else np.asarray(rels)
        for rel in rel_values:
            stats.add(int(max_iter), float(rel))
    return [project(x[:, idx].copy()) for idx in range(x.shape[1])]


# ---------------------------------------------------------------------------
# BoltVariantStats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoltVariantStats:
    """Per-variant statistics needed for BOLT-LMM-inf."""
    local_idx: int
    mean: float
    mean_center_norm2: float
    proj_norm2: float
    norm_scale: float
    x_norm2: float

    @property
    def is_model_variant(self) -> bool:
        return (
            float(self.mean_center_norm2) > 0.0
            and float(self.proj_norm2) >= 0.1
            and float(self.norm_scale) > 0.0
            and float(self.x_norm2) > 0.0
        )


class BoltVariantStatsArray(Sequence):
    """Struct-of-arrays counterpart of ``List[BoltVariantStats]``.

    Holds parallel host-numpy arrays (one row per variant, in ``local_idx``
    order). Hot-path code reads whole arrays via the field properties; cold-path
    code that needs per-element ``BoltVariantStats`` objects uses
    ``__getitem__``/``__iter__``. Returned arrays are read-only by convention
    (consumers never mutate them in place).
    """

    __slots__ = ("_local_idx", "_mean", "_mean_center_norm2",
                 "_proj_norm2", "_norm_scale", "_x_norm2")

    def __init__(self, local_idx, mean, mean_center_norm2,
                 proj_norm2, norm_scale, x_norm2):
        n = len(local_idx)
        self._local_idx         = np.ascontiguousarray(local_idx, dtype=np.int64)
        self._mean              = np.ascontiguousarray(mean, dtype=np.float64)
        self._mean_center_norm2 = np.ascontiguousarray(mean_center_norm2, dtype=np.float64)
        self._proj_norm2        = np.ascontiguousarray(proj_norm2, dtype=np.float64)
        self._norm_scale        = np.ascontiguousarray(norm_scale, dtype=np.float64)
        self._x_norm2           = np.ascontiguousarray(x_norm2, dtype=np.float64)
        for name, arr in (("mean", self._mean),
                          ("mean_center_norm2", self._mean_center_norm2),
                          ("proj_norm2", self._proj_norm2),
                          ("norm_scale", self._norm_scale),
                          ("x_norm2", self._x_norm2)):
            if arr.shape != (n,):
                raise ValueError(
                    f"BoltVariantStatsArray.{name} shape {arr.shape} != ({n},)"
                )

    # whole-array accessors (names match the BoltVariantStats fields)
    @property
    def local_idx(self) -> np.ndarray:
        return self._local_idx

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def mean_center_norm2(self) -> np.ndarray:
        return self._mean_center_norm2

    @property
    def proj_norm2(self) -> np.ndarray:
        return self._proj_norm2

    @property
    def norm_scale(self) -> np.ndarray:
        return self._norm_scale

    @property
    def x_norm2(self) -> np.ndarray:
        return self._x_norm2

    @property
    def is_model_variant_mask(self) -> np.ndarray:
        # Vectorized mirror of BoltVariantStats.is_model_variant — keep in sync.
        return (
            (self._mean_center_norm2 > 0.0)
            & (self._proj_norm2 >= 0.1)
            & (self._norm_scale > 0.0)
            & (self._x_norm2 > 0.0)
        )

    def __len__(self) -> int:
        return int(self._local_idx.shape[0])

    def __getitem__(self, i) -> BoltVariantStats:
        # int index -> a real frozen BoltVariantStats (slices unsupported; no
        # call site needs them).
        i = int(i)
        return BoltVariantStats(
            local_idx=int(self._local_idx[i]),
            mean=float(self._mean[i]),
            mean_center_norm2=float(self._mean_center_norm2[i]),
            proj_norm2=float(self._proj_norm2[i]),
            norm_scale=float(self._norm_scale[i]),
            x_norm2=float(self._x_norm2[i]),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# BoltLmmOps
# ---------------------------------------------------------------------------

class BoltLmmOps:
    """
    GRG-backed BOLT-LMM-inf linear algebra using grapp's standardized operators.
    Accepts one GRG per chromosome for LOCO. 
    """

    def __init__(
        self,
        chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
        chrom_stats: List[BoltVariantStatsArray],
        covariates: CovariateBasis,
        threads: int = 1,
        use_cupy: Optional[bool] = None,
        sample_filter: Optional[List[int]] = None,
    ):
        if len(chrom_grgs) != len(chrom_stats):
            raise ValueError("chrom_grgs and chrom_stats must have the same length")
        self._chrom_grgs = chrom_grgs
        self._chrom_stats = chrom_stats
        self._covariates = covariates
        self._threads = threads
        self._use_cupy_arg = use_cupy
        # Non-missing INDIVIDUAL indices, or None to use all individuals. When set,
        # N_used = len(sample_filter) and every operator / stat is restricted to it.
        self._sample_filter = sample_filter

        self._n: int = 0
        self._m_proj: int = 0
        self._m_proj_by_chrom: Dict[Any, int] = {}
        self._xfro2: float = 0.0

        self._x_ops: Dict[Any, Any] = {}
        self._k_all_op: Any = None
        self._x_all_op: Any = None
        self._chrom_to_op_idx: Dict[Any, int] = {}
        self._model_stats_by_chrom: Dict[Any, BoltVariantStatsArray] = {}
        self._local_idx_to_pos: Dict[Any, Dict[int, int]] = {}
        self._is_cupy: bool = False
        self._xp = np

    def setup(self) -> "BoltLmmOps":
        grgs_list = [grg for _, grg in self._chrom_grgs]
        # N_used: the number of non-missing individuals when a sample_filter is set,
        # else all individuals in the GRG.
        if self._sample_filter is not None:
            self._n = len(self._sample_filter)
            samp = [s for i in self._sample_filter for s in (2 * i, 2 * i + 1)]
        else:
            self._n = int(grgs_list[0].num_individuals)
            samp = None

        if self._covariates.nused != self._n:
            raise ValueError(
                f"covariate sample count {self._covariates.nused} != N_used {self._n}"
            )

        # Backend: use the explicitly passed flag if any; otherwise auto-detect.
        if self._use_cupy_arg is None:
            self._is_cupy = bool(grgs_list) and detect_cupy_backend(grgs_list[0])
        else:
            self._is_cupy = bool(self._use_cupy_arg)
        if self._is_cupy:
            import cupy
            self._xp = cupy

        active_grgs: List[Any] = []
        active_freqs: List[np.ndarray] = []
        active_vars: List[np.ndarray] = []

        # Operator classes are chosen through the calculator's unified selector
        # (get_operator / get_multi_operator), which picks the SciPy or CuPy
        # backend itself.

        rep_calc = _wrap_grg(grgs_list[0])
        std_x_cls = rep_calc.get_operator("X", standardized=True)

        for (chrom, grg), stats in zip(self._chrom_grgs, self._chrom_stats):
            model_stats = stats
            self._model_stats_by_chrom[chrom] = model_stats
            li = model_stats.local_idx
            self._local_idx_to_pos[chrom] = {
                int(li[pos]): pos for pos in range(len(li))
            }

            if not model_stats:
                continue

            mcn2 = model_stats.mean_center_norm2
            var_c = mcn2 / float(self._n - 1)

            freqs_c = allele_frequencies(grg, sample_filter=samp)

            m_c = len(model_stats)
            self._m_proj += m_c
            self._m_proj_by_chrom[chrom] = m_c
            self._xfro2 += float(model_stats.x_norm2.sum())

            self._x_ops[chrom] = std_x_cls(
                grg, _UP, freqs_c,
                custom_variance=var_c,
                sample_filter=self._sample_filter,
            )

            self._chrom_to_op_idx[chrom] = len(active_grgs)
            active_grgs.append(grg)
            active_freqs.append(freqs_c)
            active_vars.append(var_c)

        if self._m_proj <= 0:
            raise ValueError("no eligible model variants")

        # K_all: multi-chromosome standardized XX^T with native LOCO via
        # set_exclude (MultiSciPyStdXXTOperator / MultiCuPyStdXXTOperator).
        self._k_all_op = rep_calc.get_multi_operator("XXT", standardized=True)(
            active_grgs, active_freqs,
            custom_variance=active_vars,
            threads=self._threads,
            sample_filter=self._sample_filter,
        )

        # Multi-chromosome standardized-X operator (sum_c X_c @ w_c) for batched
        # MC probe generation; the X analog of the K_all operator above.
        self._x_all_op = rep_calc.get_multi_operator("X", standardized=True)(
            active_grgs, _UP, active_freqs,
            custom_variance=active_vars,
            threads=self._threads,
            sample_filter=self._sample_filter,
        )

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._covariates.dim

    @property
    def m_proj(self) -> int:
        return self._m_proj

    @property
    def m_proj_by_chrom(self) -> Dict[Any, int]:
        return dict(self._m_proj_by_chrom)

    @property
    def xfro2(self) -> float:
        return self._xfro2

    @property
    def chroms(self) -> List[Any]:
        return [chrom for chrom, _ in self._chrom_grgs]

    @property
    def xp(self):
        return self._xp

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, v):
        return self._covariates.project_device(v)

    def project_inplace(self, v):
        return self._covariates.project_device_inplace(v)

    # ------------------------------------------------------------------
    # Per-chromosome operations
    # ------------------------------------------------------------------

    def _device_ctx(self, chrom):
        if not self._is_cupy:
            return contextlib.nullcontext()
        import cupy as cp
        dev = getattr(self._x_ops[chrom], "_device", None)
        return cp.cuda.Device(dev) if dev is not None else contextlib.nullcontext()

    def scores(self, chrom, v) -> np.ndarray:
        """X^T @ project(v) for model variants of this chromosome (compact array)."""
        v_proj = self.project(v)
        return self._x_ops[chrom].rmatvec(v_proj)

    def apply_x(self, chrom, w) -> np.ndarray:
        """project(X @ w) for model variants of this chromosome."""
        with self._device_ctx(chrom):
            w_dev = self._xp.asarray(w, dtype=DTYPE)
            result = self._x_ops[chrom].matvec(w_dev)
        if self._is_cupy:
            # TODO: projection should be done on the same device as grg if possible
            from grapp.linalg.ops_cupy import _xdev_asarray
            result = _xdev_asarray(result)
        return self._covariates.project_device(result)

    def apply_x_all(self, weights) -> Any:
        """project(sum_c X_c @ w_c) for a (m_proj, k) weight matrix whose rows are in
        active-chromosome (operator) order and whose columns are the k probe vectors.

        The Multi-X operator does the per-chromosome matmuls and the cross-device sum;
        projection is applied once here on the (n, k) result.
        """
        # Pass a HOST array: the operator uploads each chromosome's row-slice straight
        # to that chromosome's device (host->device), avoiding a cross-device GPU copy
        # of a (possibly non-contiguous) slice. The reduction + projection are pinned to
        # the operator's output device so the covariate basis and result share a device.
        W = np.asarray(weights, dtype=DTYPE)
        if not self._is_cupy:
            return self._covariates.project_device(self._x_all_op.matmat(W))
        import cupy as cp
        with cp.cuda.Device(self._x_all_op._output_device):
            result = self._x_all_op.matmat(W)  # (n, k), unprojected, on output device
            return self._covariates.project_device(result)

    def column(self, chrom, local_idx: int) -> np.ndarray:
        """The projected column x_i of the standardized X for one model variant."""
        pos = self._local_idx_to_pos[chrom][int(local_idx)]
        m = len(self._model_stats_by_chrom[chrom])
        with self._device_ctx(chrom):
            w = self._xp.zeros(m, dtype=DTYPE)
            w[pos] = 1.0
        return self.apply_x(chrom, w)

    # ------------------------------------------------------------------
    # Kinship
    # ------------------------------------------------------------------

    def apply_k(self, v, *, exclude_chrom=None) -> np.ndarray:
        """(1/m_loco) * X_loco @ X_loco^T @ project(v), then project."""
        if exclude_chrom is not None and exclude_chrom not in self._chrom_to_op_idx:
            raise KeyError(
                f"exclude_chrom={exclude_chrom!r} not in active chroms "
                f"{sorted(self._chrom_to_op_idx.keys())}"
            )
        v_dev = self._xp.asarray(v, dtype=DTYPE)
        v_proj = self._covariates.project_device(v_dev)
        exclude_idx = (
            self._chrom_to_op_idx[exclude_chrom]
            if exclude_chrom is not None else None
        )
        # Select the left-out chromosome (sticky state on the Multi-XXT operator),
        # apply, then clear the exclusion so the operator's resting state is the
        # full (no-exclude) sum.
        self._k_all_op.set_exclude(exclude_idx)
        try:
            xxt_v = self._k_all_op.matvec(v_proj)
        finally:
            self._k_all_op.set_exclude(None)

        m = self._m_proj
        if exclude_chrom is not None:
            m -= self._m_proj_by_chrom[exclude_chrom]
        if m <= 0:
            return self._xp.zeros(self._n, dtype=DTYPE)
        result = xxt_v / float(m)
        self.project_inplace(result)
        return result

    # ------------------------------------------------------------------
    # Accessor for stats
    # ------------------------------------------------------------------

    def model_stats_for(self, chrom):
        # Returns a BoltVariantStatsArray, or [] for an absent chrom (the empty
        # list still satisfies the len/truthiness/iterate uses at call sites).
        return self._model_stats_by_chrom.get(chrom, [])

    def all_model_stats(self) -> List[Tuple[Any, BoltVariantStats]]:
        result = []
        for chrom, _ in self._chrom_grgs:
            for s in self._model_stats_by_chrom.get(chrom, []):
                result.append((chrom, s))
        return result


# ---------------------------------------------------------------------------
# Variance component helpers
# ---------------------------------------------------------------------------

def log_delta_from_h2(ops: BoltLmmOps, h2: float) -> float:
    return math.log(float(ops.xfro2) / (float(ops.m_proj) * float(ops.dim)) * (1.0 - float(h2)) / float(h2))


def h2_from_log_delta(ops: BoltLmmOps, log_delta: float) -> float:
    return float(ops.xfro2) / (float(ops.xfro2) + float(ops.m_proj) * float(ops.dim) * math.exp(float(log_delta)))


def _sum_score_squares(ops: BoltLmmOps, vector) -> float:
    total = 0.0
    for chrom in ops.chroms:
        if not ops.model_stats_for(chrom):
            continue
        scores = ops.scores(chrom, vector)
        xp = _array_module(scores)
        total += _as_float(xp.sum(scores * scores))
    return float(total)


# ---------------------------------------------------------------------------
# MC component generation
# ---------------------------------------------------------------------------

def _generate_bolt_mc_components(
    ops: BoltLmmOps,
    y,
    *,
    trials: int,
    seed: int,
    batched_apply_x: bool = False,
) -> Tuple[List[Any], List[Any], Any]:
    xp = ops.xp
    inv_sqrt_m = 1.0 / math.sqrt(float(ops.m_proj))

    # Vectorized numpy draws. Order: all weights, then noise.
    weights_by_chrom: Dict[Any, np.ndarray] = {}
    gen = np.random.default_rng(int(seed) + 1)
    for chrom in ops.chroms:
        model_stats = ops.model_stats_for(chrom)
        if not model_stats:
            continue
        m_c = len(model_stats)
        weights_by_chrom[chrom] = (
            gen.standard_normal((int(trials), m_c)) * inv_sqrt_m
        )
    noise = gen.standard_normal((int(trials), int(ops.n)))

    e_rand: List[Any] = []

    # g_rand[t] = project(sum_c X_c @ w_{c,t}). Stack the per-chrom (trials, m_c) weight
    # blocks into a single (m_proj, trials) matrix in operator (active-chrom) order. Both
    # paths go through the Multi-X operator (ops.apply_x_all); they differ only in matmat
    # batch width: all trials at once vs. one probe column per call.
    ordered_w = [weights_by_chrom[chrom].T for chrom in ops.chroms if chrom in weights_by_chrom]
    W = np.concatenate(ordered_w, axis=0)  # (m_proj, trials)

    with _nvtx("bolt:mc_gen:apply_x"):
        if batched_apply_x:
            G = ops.apply_x_all(W)              # (n, trials), projected
            g_rand: List[Any] = [G[:, t].copy() for t in range(int(trials))]
        else:
            g_rand = []
            for t in range(int(trials)):
                g = ops.apply_x_all(W[:, t:t + 1])  # (n, 1), projected
                g_rand.append(g[:, 0].copy())

    for trial in range(int(trials)):
        e = xp.asarray(noise[trial], dtype=DTYPE)
        ops.project_inplace(e)
        e_rand.append(e)

    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    return g_rand, e_rand, y_dev


# ---------------------------------------------------------------------------
# MC scaling
# ---------------------------------------------------------------------------

def _compute_mc_scaling(
    ops: BoltLmmOps,
    y_dev,
    g_rand: Sequence[Any],
    e_rand: Sequence[Any],
    *,
    log_delta: float,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> McScalingResult:
    trials = len(g_rand)
    delta = math.exp(float(log_delta))
    sqrt_delta = math.sqrt(delta)

    def h_into(src, dst) -> None:
        result = ops.apply_k(src, exclude_chrom=None)
        dst[...] = result + delta * src

    rand_beta: List[float] = []
    rand_eps: List[float] = []

    with _nvtx("bolt:mc:rhs_build"):
        rhs_columns = []
        for g_t, e_t in zip(g_rand, e_rand):
            rhs = e_t * sqrt_delta + g_t
            ops.project_inplace(rhs)
            rhs_columns.append(rhs)
        rhs_columns.append(y_dev)

    with _nvtx("bolt:mc:cg_solve"):
        z_columns = bolt_conj_grad_solve(
            [h_into for _ in rhs_columns],
            rhs_columns,
            rel_tol=rel_tol,
            max_iter=max_iter,
            stats=stats,
            project=ops.project,
        )

    with _nvtx("bolt:mc:reductions"):
        for z_t in z_columns[:-1]:
            rand_beta.append(_sum_score_squares(ops, z_t))
            rand_eps.append(_dot(z_t, z_t))

        z_data = z_columns[-1]
        data_beta = _sum_score_squares(ops, z_data)
        data_eps = _dot(z_data, z_data)

    if min([data_beta, data_eps, *rand_beta, *rand_eps]) <= 0.0:
        raise RuntimeError("invalid BOLT MC-scaling objective component")

    rand_beta_total = float(sum(rand_beta))
    rand_eps_total = float(sum(rand_eps))
    f_reml = math.log((data_beta / data_eps) / (rand_beta_total / rand_eps_total))

    f_jacks: List[float] = []
    for jack in range(trials + 1):
        jack_rand_beta = 0.0
        jack_rand_eps = 0.0
        for trial in range(trials):
            if trial != jack:
                jack_rand_beta += rand_beta[trial]
                jack_rand_eps += rand_eps[trial]
        if jack_rand_beta <= 0.0 or jack_rand_eps <= 0.0:
            f_jacks.append(float("nan"))
        else:
            f_jacks.append(math.log((data_beta / data_eps) / (jack_rand_beta / jack_rand_eps)))
    f_jacks[-1] = f_reml

    f_rands_as_data: List[float] = []
    for trial in range(trials):
        f_rands_as_data.append(
            math.log((rand_beta[trial] / rand_eps[trial]) / (rand_beta_total / rand_eps_total))
        )

    sigma2_k = _dot(y_dev, z_data) / float(max(ops.dim, 1))
    return McScalingResult(
        log_delta=float(log_delta),
        f_jacks=tuple(float(v) for v in f_jacks),
        f_rands_as_data=tuple(float(v) for v in f_rands_as_data),
        sigma2_k=float(sigma2_k),
        all_hinv_y=z_data,
    )


# ---------------------------------------------------------------------------
# Variance-component fitting
# ---------------------------------------------------------------------------

def fit_bolt_variance_components(
    ops: BoltLmmOps,
    y,
    *,
    mc_trials: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
    batched_apply_x: bool = False,
) -> VarianceFit:
    if int(mc_trials) <= 0:
        # BOLT default (setMCtrials, Bolt.cpp:2128-2137): auto-size from N.
        trials = max(min(int(4e9 / ops.n / ops.n), 15), 3)
        logger.info("Using default number of MC trials: %d (for N = %d)", trials, ops.n)
    else:
        trials = max(2, int(mc_trials))
    logger.info("Estimating variance parameters: %d MC trials, CGtol=%.3g", trials, rel_tol)
    with _nvtx("bolt:vc:mc_components"):
        g_rand, e_rand, y_dev = _generate_bolt_mc_components(
            ops, y, trials=trials, seed=int(seed),
            batched_apply_x=batched_apply_x,
        )

    def evaluate(log_delta: float) -> McScalingResult:
        with _nvtx("bolt:vc:eval"):
            res = _compute_mc_scaling(
                ops, y_dev, g_rand, e_rand,
                log_delta=float(log_delta),
                rel_tol=rel_tol,
                max_iter=max_iter,
                stats=stats,
            )
        logger.info(
            "MCscaling: logDelta=%.4f h2=%.4f f=%.6g",
            res.log_delta, h2_from_log_delta(ops, res.log_delta), res.f_reml,
        )
        return res

    with _nvtx("bolt:vc:secant"):
        prev = evaluate(log_delta_from_h2(ops, 0.25))
        cur = evaluate(log_delta_from_h2(ops, 0.125 if prev.f_reml < 0.0 else 0.5))
        best = prev if abs(prev.f_reml) <= abs(cur.f_reml) else cur
        if abs(prev.f_reml) < abs(cur.f_reml):
            prev, cur = cur, prev

        # Mirror BOLT (Bolt.cpp:2179): clear bestVCs.fJacks so `best` must be re-adopted
        # from a secant iterate, while keeping best.log_delta for the exit check.
        best_is_empty = True
        converged = False
        for _step in range(5):
            if abs(cur.f_reml - prev.f_reml) < 1e-300:
                break
            next_log_delta = (prev.log_delta * cur.f_reml - cur.log_delta * prev.f_reml) / (cur.f_reml - prev.f_reml)
            next_log_delta = float(np.clip(next_log_delta, -10.0, 10.0))
            # Exit when the current point is the best found and the step is tiny
            # (Bolt.cpp:2213-2218).
            if best.log_delta == cur.log_delta and abs(next_log_delta - cur.log_delta) < 0.01:
                converged = True
                break
            prev = cur
            cur = evaluate(next_log_delta)
            # updateBestMCscalingF (Bolt.cpp:2010): adopt while empty or strictly better.
            if best_is_empty or abs(cur.f_reml) < abs(best.f_reml):
                best = cur
                best_is_empty = False

    if not converged:
        logger.warning("Secant iteration for h2 estimation may not have converged")
    logger.debug("Secant variance search complete")

    delta = math.exp(float(best.log_delta))
    sigma_g2 = float(best.sigma2_k)
    sigma_e2 = delta * sigma_g2
    h2 = h2_from_log_delta(ops, best.log_delta)
    logger.info(
        "Estimated heritability h2g=%.4f; sigma_g2=%.6g logDelta=%.6f f=%.6g",
        h2, sigma_g2, best.log_delta, best.f_reml,
    )
    return VarianceFit(
        log_delta=float(best.log_delta),
        sigma_g2=sigma_g2,
        sigma_e2=float(sigma_e2),
        h2=float(h2),
        delta=float(delta),
        all_hinv_y=best.all_hinv_y,
    )


# ---------------------------------------------------------------------------
# LOCO residuals
# ---------------------------------------------------------------------------

def solve_loco_hinv_y(
    ops: BoltLmmOps,
    y,
    *,
    fit: VarianceFit,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> Dict[Any, Any]:
    xp = ops.xp
    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE)))
    chroms = ops.chroms
    matvecs = []
    rhs_columns = []
    for chrom in chroms:
        def h_into(src, dst, left_out=chrom) -> None:
            result = ops.apply_k(src, exclude_chrom=left_out)
            dst[...] = result + float(fit.delta) * src

        matvecs.append(h_into)
        rhs_columns.append(y_dev)
    with _nvtx("bolt:loco:cg_solve"):
        solved = bolt_conj_grad_solve(
            matvecs, rhs_columns,
            rel_tol=rel_tol, max_iter=max_iter, stats=stats,
            project=ops.project,
        )
    return {chrom: value for chrom, value in zip(chroms, solved)}


# ---------------------------------------------------------------------------
# Calibration SNP selection
# ---------------------------------------------------------------------------

def select_bolt_calibration_snps(
    ops: BoltLmmOps,
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
) -> Tuple[List[Tuple[Any, BoltVariantStats]], int]:
    num_calib = int(count)
    if num_calib < 2:
        raise ValueError("at least two calibration SNPs are required")

    # Flat index over all model stats in chrom order, WITHOUT materializing a
    # BoltVariantStats per variant (all_model_stats would construct millions of
    # objects). Each segment is (chrom, flat_start, stats_array); we map a flat
    # index m -> (chrom, j) by a linear scan over the (few) chrom segments.
    segments: List[Tuple[Any, int, "BoltVariantStatsArray"]] = []
    offset = 0
    for chrom, _ in ops._chrom_grgs:
        arr = ops._model_stats_by_chrom.get(chrom, [])
        n_c = len(arr)
        if n_c == 0:
            continue
        segments.append((chrom, offset, arr))
        offset += n_c
    model_count = offset
    if model_count <= 0:
        raise ValueError("no eligible SNPs available for calibration")

    def _locate(m: int) -> Tuple[Any, "BoltVariantStatsArray", int]:
        for seg_chrom, seg_start, seg_arr in segments:
            if m < seg_start + len(seg_arr):
                return seg_chrom, seg_arr, m - seg_start
        raise IndexError(m)
    if num_calib > model_count:
        raise ValueError(
            f"requested {num_calib} calibration SNPs but only {model_count} model SNPs are available"
        )

    m_total = model_count
    m_first = [m_total] * (num_calib + 1)
    m_good = 0
    for m in range(m_total):
        block = num_calib * m_good // model_count
        if m_first[block] == m_total:
            m_first[block] = m
        m_good += 1
    if any(start == m_total for start in m_first[:-1]):
        raise RuntimeError("failed to build BOLT calibration SNP blocks")

    all_hinv_norm2 = _dot(fit.all_hinv_y, fit.all_hinv_y)
    if all_hinv_norm2 <= 0.0:
        raise RuntimeError("all-chromosome H^-1 y has nonpositive norm")

    # Compute GRAMMAR scores per chrom
    grammar_scores_by_chrom: Dict[Any, np.ndarray] = {}
    for chrom in ops.chroms:
        if not ops.model_stats_for(chrom):
            continue
        scores = ops.scores(chrom, fit.all_hinv_y)
        xp = _array_module(scores)
        if hasattr(xp, "asnumpy"):
            grammar_scores_by_chrom[chrom] = xp.asnumpy(scores)
        else:
            grammar_scores_by_chrom[chrom] = np.asarray(scores)

    rng = np.random.default_rng(int(seed) + 321)
    selected: List[Tuple[Any, BoltVariantStats]] = []
    tried = 0
    for block in range(num_calib):
        block_start = int(m_first[block])
        block_end = int(m_first[block + 1])
        block_width = block_end - block_start
        if block_width <= 0:
            raise RuntimeError(f"empty calibration block {block}")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 1_000_000:
                raise RuntimeError(f"could not select a calibration SNP from block {block}")
            m = block_start + int(rng.integers(block_width))
            chrom, arr, j = _locate(m)
            tried += 1
            # position within this chrom's compact score array
            pos = ops._local_idx_to_pos[chrom][int(arr.local_idx[j])]
            grammar_score = float(grammar_scores_by_chrom[chrom][pos])
            x_norm2 = float(arr.x_norm2[j])
            retro_stat = (grammar_score ** 2) / all_hinv_norm2 / x_norm2 * float(ops.dim)
            if retro_stat < 5.0:
                # Build the per-variant object only for the selected SNP.
                selected.append((chrom, arr[j]))
                break
    return selected, tried


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_lmm_inf(
    ops: BoltLmmOps,
    y,
    residuals: Dict[Any, Any],
    *,
    fit: VarianceFit,
    count: int,
    seed: int,
    rel_tol: float,
    max_iter: int,
    stats: CgStats,
) -> CalibrationResult:
    with _nvtx("bolt:calib:select_snps"):
        selected, tried = select_bolt_calibration_snps(ops, fit=fit, count=int(count), seed=int(seed))
    logger.info(
        "Selected %d calibration SNPs (tried %d, threw out %d)",
        len(selected), int(tried), int(tried) - len(selected),
    )
    pro_stats: List[float] = []
    retro_stats: List[float] = []
    ratios: List[float] = []
    n_minus_c = float(max(ops.dim, 1))

    xp = ops.xp
    y_dev = ops.project(xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    chroms = ops.chroms
    rhs_columns = []
    matvecs = []

    for chrom in chroms:
        def h_into(src, dst, left_out=chrom) -> None:
            result = ops.apply_k(src, exclude_chrom=left_out)
            dst[...] = result + float(fit.delta) * src

        matvecs.append(h_into)
        rhs_columns.append(y_dev)

    selected_columns = []
    with _nvtx("bolt:calib:columns"):
        for sel_chrom, sel_stat in selected:
            x = ops.column(sel_chrom, sel_stat.local_idx)

            def h_into(src, dst, left_out=sel_chrom) -> None:
                result = ops.apply_k(src, exclude_chrom=left_out)
                dst[...] = result + float(fit.delta) * src

            matvecs.append(h_into)
            rhs_columns.append(x)
            selected_columns.append(x)

    with _nvtx("bolt:calib:cg_solve"):
        solved_columns = bolt_conj_grad_solve(
            matvecs, rhs_columns,
            rel_tol=rel_tol, max_iter=max_iter, stats=stats,
            project=ops.project,
        )

    residuals.clear()
    for chrom, solved in zip(chroms, solved_columns[: len(chroms)]):
        residuals[chrom] = solved

    q_by_sel = {
        i: solved
        for i, solved in enumerate(solved_columns[len(chroms):])
    }
    x_by_sel = {i: x for i, x in enumerate(selected_columns)}

    with _nvtx("bolt:calib:moments"):
        h_norm2 = {chrom: _dot(value, value) for chrom, value in residuals.items()}
        phi_h_phi = {chrom: _dot(y_dev, value) for chrom, value in residuals.items()}

        for i, (sel_chrom, sel_stat) in enumerate(selected):
            x = x_by_sel[i]
            score_h = _dot(x, residuals[sel_chrom])
            x_norm2 = _dot(x, x)
            if h_norm2[sel_chrom] <= 0.0 or phi_h_phi[sel_chrom] <= 0.0:
                raise RuntimeError(f"invalid LOCO H^-1 y moments for chr{sel_chrom}")
            if x_norm2 <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive projected norm: {sel_stat.local_idx}")
            retro = n_minus_c * score_h * score_h / (h_norm2[sel_chrom] * x_norm2)
            if retro <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive retrospective stat: {sel_stat.local_idx}")
            q = q_by_sel[i]
            denom_h = _dot(x, q)
            if denom_h <= 0.0:
                raise RuntimeError(f"selected calibration SNP has nonpositive prospective denominator: {sel_stat.local_idx}")
            pro = n_minus_c * score_h * score_h / denom_h / phi_h_phi[sel_chrom]
            pro_stats.append(float(pro))
            retro_stats.append(float(retro))
            ratios.append(float(pro / retro))

    total_pro = float(sum(pro_stats))
    total_retro = float(sum(retro_stats))
    if total_pro <= 0.0 or total_retro <= 0.0:
        raise RuntimeError("calibration failed: prospective or retrospective sum is nonpositive")
    factor = total_pro / total_retro
    calibration_raw = factor
    calibration_jacks = [
        (total_pro - pro) / (total_retro - retro)
        for pro, retro in zip(pro_stats, retro_stats)
    ]
    jack_count = len(calibration_jacks)
    jack_sum = float(sum(calibration_jacks))
    jack_sum2 = float(sum(v * v for v in calibration_jacks))
    calibration_std = math.sqrt(
        max(0.0, (jack_sum2 - jack_sum * jack_sum / jack_count) * (jack_count - 1) / jack_count)
    )
    ratio_of_medians = float(
        np.median(np.asarray(pro_stats, dtype=np.float64))
        / np.median(np.asarray(retro_stats, dtype=np.float64))
    )
    median_of_ratios = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if calibration_std > 0.01:
        factor = ratio_of_medians
    if factor <= 0.0:
        raise RuntimeError(f"calibration factor is nonpositive: {factor}")

    vinv_scale_by_chrom = {}
    for chrom, residual in residuals.items():
        resid_norm2 = _dot(residual, residual)
        if resid_norm2 <= 0.0:
            raise RuntimeError(f"LOCO H^-1 y has nonpositive norm for chr{chrom}")
        resid_factor = math.sqrt(n_minus_c / resid_norm2 * factor)
        vinv_scale_by_chrom[chrom] = 1.0 / (resid_factor * float(fit.sigma_g2))

    logger.info(
        "AvgPro=%.3f AvgRetro=%.3f Calibration=%.3f (%.3f)  "
        "RatioOfMedians=%.3f MedianOfRatios=%.3f (%d SNPs)",
        float(np.mean(pro_stats)), float(np.mean(retro_stats)),
        calibration_raw, calibration_std,
        ratio_of_medians, median_of_ratios, len(pro_stats),
    )

    return CalibrationResult(
        factor=float(factor),
        std=float(calibration_std),
        ratio_of_medians=ratio_of_medians,
        median_of_ratios=median_of_ratios,
        selected_snps=tuple(str(stat.local_idx) for _, stat in selected),
        tried_snps=int(tried),
        vinv_scale_by_chrom=vinv_scale_by_chrom,
    )


# ---------------------------------------------------------------------------
# Per-variant statistics from GRG (replaces PLINK-based attach_bed_stats)
# ---------------------------------------------------------------------------

def compute_bolt_variant_stats(
    grg: GRGCalcInterface,
    covariates: CovariateBasis,
    n_individuals: int,
    use_cupy: Optional[bool] = None,
    sample_filter: Optional[List[int]] = None,
) -> BoltVariantStatsArray:
    """
    Compute BOLT-LMM-inf per-variant statistics from a GRG.

    Replaces the PLINK-based attach_bed_stats + attach_projected_bed_stats pipeline.
    Uses GRG traversals to compute mean_center_norm2 and proj_norm2 without
    reading BED files.

    Note that this is lightweight and is done sequentially currently, not using
    the linear operators.

    :param sample_filter: Optional list of non-missing INDIVIDUAL indices. When
        given, every quantity (counts, means, norms, covariate projection) is
        computed over only those individuals, matching BOLT's removal of
        missing-phenotype individuals. ``n_individuals`` must equal
        ``len(sample_filter)`` and ``covariates`` must be built over that many
        rows. ``None`` => use all individuals (unchanged behavior).
    """
    grg = _wrap_grg(grg)
    n = int(n_individuals)

    if use_cupy is None:
        use_cupy = detect_cupy_backend(grg)
    if use_cupy:
        import cupy as cp
        xp = cp
        dev_ctx = lambda: cp.cuda.Device(grg.device)
    else:
        xp = np
        dev_ctx = contextlib.nullcontext

    # Sample-level (haplotype) indices for allele_counts; per-individual 0/1 mask
    # for the by_individual xtx / covariate traversals. None => all individuals.
    if sample_filter is not None:
        samp = [s for i in sample_filter for s in (2 * i, 2 * i + 1)]
        indiv_mask = np.zeros(grg.num_individuals, dtype=np.float64)
        indiv_mask[sample_filter] = 1.0
    else:
        samp = None
        indiv_mask = np.ones(grg.num_individuals, dtype=np.float64)

    acount_raw, miss_raw = allele_counts(grg, return_missing=True, sample_filter=samp)
    acount = np.asarray(acount_raw, dtype=np.float64)
    miss = np.asarray(miss_raw, dtype=np.float64)

    # Effective sample count and diploid mean per variant
    n_eff = n - miss / grg.ploidy
    with np.errstate(divide="ignore", invalid="ignore"):
        diploid_mean = np.where(n_eff > 0, acount / n_eff, 0.0)

    # sumsq_g = diag(X_indiv^T X_indiv)_j = sum_i g_ij^2, g_ij in {0,1,2}.
    # init="xtx" with by_individual=True computes the individual-level squared sum,
    # matching the native BED-based computation (sumsq_lut[g].sum() = sum_i g_ij^2).
    # The input row is a per-individual 0/1 weight, so passing indiv_mask (1 on
    # non-missing) restricts the squared sum to the kept individuals exactly.
    with dev_ctx():
        inp = xp.asarray(indiv_mask).reshape(1, -1)

    sumsq_g = _to_np(grg.matmul(
        inp,
        pygrgl.TraversalDirection.UP,
        by_individual=True,
        init="xtx",
    )).squeeze().astype(np.float64)

    # mean_center_norm2 = sum_i (x_ij - mean_j)^2 (using n_eff for mean)
    mean_center_norm2 = sumsq_g - acount * diploid_mean

    # norm_scale uses n_individuals (not n_eff), matching BOLT's Bessel correction
    norm_scale = np.where(
        mean_center_norm2 > 0.0,
        np.sqrt((n - 1.0) / np.maximum(mean_center_norm2, 1e-300)),
        0.0,
    )

    # proj_norm2: subtract covariate contribution via c UP traversals
    # proj_norm2_i = mean_center_norm2_i - sum_k (X_i^T q_k - mean_i * sum(q_k))^2
    sum_sq_proj = np.zeros(grg.num_mutations, dtype=np.float64)
    Q = covariates.basis  # (N_used, cindep)
    for k in range(covariates.cindep):
        q_k = Q[:, k].astype(np.float64)
        sum_qk = float(np.sum(q_k))
        # The by_individual traversal needs a full length-num_individuals vector;
        # scatter the (kept-only) covariate column back, zero on missing, so the
        # score is summed over the kept individuals only.
        if sample_filter is not None:
            q_full = np.zeros(grg.num_individuals, dtype=np.float64)
            q_full[sample_filter] = q_k
        else:
            q_full = q_k
        with dev_ctx():
            q_dev = xp.asarray(q_full).reshape(1, -1)
        raw_scores = _to_np(grg.matmul(
            q_dev,
            pygrgl.TraversalDirection.UP,
            by_individual=True,
        )).squeeze().astype(np.float64)
        score_k = raw_scores - diploid_mean * sum_qk
        sum_sq_proj += score_k * score_k

    proj_norm2 = np.maximum(0.0, mean_center_norm2 - sum_sq_proj)
    x_norm2 = proj_norm2 * norm_scale * norm_scale

    result = BoltVariantStatsArray(
        local_idx=np.arange(grg.num_mutations, dtype=np.int64),
        mean=diploid_mean,
        mean_center_norm2=mean_center_norm2,
        proj_norm2=proj_norm2,
        norm_scale=norm_scale,
        x_norm2=x_norm2,
    )
    logger.debug("Per-variant stats (chrom) complete")
    return result


# ---------------------------------------------------------------------------
# Association statistics (fast numeric path)
# ---------------------------------------------------------------------------

DEFAULT_PVALUE_METHOD = "erfc"  # df=1 chi-square survival via closed form


def _chi2_sf_df1(x: np.ndarray, method: str = DEFAULT_PVALUE_METHOD) -> np.ndarray:
    """Survival function of a chi-square with 1 dof, vectorized.

    method="erfc" (default): exact closed form sf(x) = erfc(sqrt(x/2)); ~46x
        faster than scipy.stats.chi2.sf(x, df=1) and equal to ~1e-15. Inputs
        here are always >= 0 (squared scores over positive norms).
    method="scipy": scipy.stats.chi2.sf(x, df=1) (reference / slow).
    """
    if method == "erfc":
        return _erfc(np.sqrt(x * 0.5))
    if method == "scipy":
        return _scipy_chi2.sf(x, df=1)
    raise ValueError(f"unknown pvalue_method {method!r}")


@dataclass(frozen=True)
class BoltChromInfStats:
    """Per-chromosome BOLT-LMM-inf numeric stats as parallel arrays (all variants).

    Aligned 1:1 with ``all_stats`` order. Non-model variants carry placeholder
    values (``BOLT_BAD_SNP_STAT`` chi2, p=1.0, beta=0, se=nan). Contains no GRG
    mutation metadata; use ``lmm_inf_stats_to_dataframe`` (in ``grapp.assoc.bolt_lmm``)
    to annotate.
    """
    chrom: Any
    local_idx: np.ndarray      # int64, in all_stats order (== compact score position)
    a1freq: np.ndarray         # float64
    chisq_linreg: np.ndarray
    p_linreg: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    chisq_lmm_inf: np.ndarray
    p_lmm_inf: np.ndarray


# chi-square 1-df median, used to normalize the genomic control factor lambdaGC.
_CHI2_1DF_MEDIAN = 0.4549364231195732


def summarize_chisq(stats: List["BoltChromInfStats"]) -> Dict[str, Dict[str, float]]:
    """Mean chi-square and lambdaGC over good (model) SNPs, for LINREG and LMM-inf.

    Concatenates the per-chromosome chi-square arrays, drops placeholder
    (``BOLT_BAD_SNP_STAT``) entries carried by non-model variants, and returns
    ``{"linreg": {...}, "lmm_inf": {...}}`` with ``mean``, ``lambda_gc`` and
    ``n_good`` for each. One vectorized pass; cheap relative to the pipeline.
    """
    out: Dict[str, Dict[str, float]] = {}
    for key, attr in (("linreg", "chisq_linreg"), ("lmm_inf", "chisq_lmm_inf")):
        if stats:
            chisq = np.concatenate([np.asarray(getattr(cs, attr)) for cs in stats])
        else:
            chisq = np.empty(0, dtype=np.float64)
        good = chisq[chisq != BOLT_BAD_SNP_STAT]
        if good.size:
            mean = float(np.mean(good))
            lambda_gc = float(np.median(good) / _CHI2_1DF_MEDIAN)
        else:
            mean = float("nan")
            lambda_gc = float("nan")
        out[key] = {"mean": mean, "lambda_gc": lambda_gc, "n_good": int(good.size)}
    return out


def compute_lmm_inf_stats(
    ops: BoltLmmOps,
    chrom_grgs: List[Tuple[Any, GRGCalcInterface]],
    chrom_all_stats: List[BoltVariantStatsArray],
    y,
    residuals: Dict[Any, Any],
    fit: VarianceFit,
    calibration: CalibrationResult,
    pvalue_method: str = DEFAULT_PVALUE_METHOD,
) -> List[BoltChromInfStats]:
    """
    Compute per-variant BOLT-LMM-inf and linear-regression statistics (fast path).

    Returns one ``BoltChromInfStats`` per chromosome, holding the numeric stats
    as vectorized numpy arrays in ``all_stats`` order (all variants; non-model
    variants carry placeholder values). Contains everything derivable from
    ``BoltLmmOps`` and ``BoltVariantStats`` without any per-variant
    ``get_mutation_by_id`` lookup. Use ``lmm_inf_stats_to_dataframe`` (in
    ``grapp.assoc.bolt_lmm``) to attach GRG mutation metadata and produce the standard
    BOLT-LMM DataFrame.

    Backend-correct: the operator dispatch inside ``ops.scores`` follows
    ``ops._is_cupy``.
    """
    y_dev = ops.project(ops.xp.asarray(np.asarray(y, dtype=DTYPE).copy()))
    y_norm2 = _dot(y_dev, y_dev)
    if y_norm2 <= 0.0:
        raise RuntimeError("phenotype has nonpositive projected norm")

    results: List[BoltChromInfStats] = []
    for (chrom, grg), all_stats in zip(chrom_grgs, chrom_all_stats):
      with _nvtx("bolt:assoc:chrom"):
        vinv_scale = float(calibration.vinv_scale_by_chrom[chrom])
        if vinv_scale <= 0.0:
            raise RuntimeError(f"nonpositive VinvScaleFactor for chr{chrom}: {vinv_scale}")

        # Compact scores via the existing per-chrom op. ops.scores internally
        # projects; passing already-projected vectors is safe because the
        # orthogonal projection is idempotent. Arrays are full-length and
        # aligned 1:1 with all_stats order (== compact score position).
        linreg_scores = ops.scores(chrom, y_dev)
        lmm_scores    = ops.scores(chrom, residuals[chrom])

        xp_s = _array_module(linreg_scores)
        if hasattr(xp_s, "asnumpy"):
            linreg_scores = xp_s.asnumpy(linreg_scores)
            lmm_scores    = xp_s.asnumpy(lmm_scores)
        else:
            linreg_scores = np.asarray(linreg_scores)
            lmm_scores    = np.asarray(lmm_scores)

        # A1FREQ must be over the kept (non-missing) individuals, matching BOLT.
        if ops._sample_filter is not None:
            _samp = [s for i in ops._sample_filter for s in (2 * i, 2 * i + 1)]
        else:
            _samp = None
        freqs_full = allele_frequencies(grg, sample_filter=_samp)

        # Per-variant arrays in all_stats order. These are read-only views into
        # the BoltVariantStatsArray; downstream arithmetic allocates new arrays
        # (never mutates these in place).
        local_idx = all_stats.local_idx
        ns   = all_stats.norm_scale
        pn2  = all_stats.proj_norm2
        xn2  = all_stats.x_norm2

        model_mask = all_stats.is_model_variant_mask

        a1freq = freqs_full[local_idx].astype(np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            linreg_chi2 = (linreg_scores * linreg_scores) / y_norm2 / xn2 * float(ops.dim)

            vinv_score_raw = (lmm_scores / ns) / float(fit.sigma_g2)
            lmm_chi2 = ((vinv_score_raw / vinv_scale) ** 2) / pn2
            beta     = vinv_score_raw / (pn2 * vinv_scale * vinv_scale)
            se       = 1.0 / (np.sqrt(pn2) * vinv_scale)

        linreg_p = _chi2_sf_df1(linreg_chi2, pvalue_method)
        lmm_p    = _chi2_sf_df1(lmm_chi2, pvalue_method)

        # Placeholders for non-model variants (match scalar version).
        bad = BOLT_BAD_SNP_STAT
        linreg_chi2 = np.where(model_mask, linreg_chi2, bad)
        lmm_chi2    = np.where(model_mask, lmm_chi2, bad)
        linreg_p    = np.where(model_mask, linreg_p, 1.0)
        lmm_p       = np.where(model_mask, lmm_p, 1.0)
        beta        = np.where(model_mask, beta, 0.0)
        se          = np.where(model_mask, se, np.nan)

        results.append(BoltChromInfStats(
            chrom=chrom,
            local_idx=local_idx,
            a1freq=a1freq,
            chisq_linreg=np.ascontiguousarray(linreg_chi2, dtype=np.float64),
            p_linreg=np.ascontiguousarray(linreg_p, dtype=np.float64),
            beta=np.ascontiguousarray(beta, dtype=np.float64),
            se=np.ascontiguousarray(se, dtype=np.float64),
            chisq_lmm_inf=np.ascontiguousarray(lmm_chi2, dtype=np.float64),
            p_lmm_inf=np.ascontiguousarray(lmm_p, dtype=np.float64),
        ))

    return results
