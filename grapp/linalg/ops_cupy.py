"""
CuPy-native linear operators for Spmv-GRG matrix multiplications.

All operators require a CUDA-capable GRG (grg.device must not be None).
CuPy is a hard dependency; an ImportError at import time indicates it is
not installed.

Memory model
------------
For single-GRG operators, the input and output buffers are assumed to be
on the GRG's device. For multi-GRG operators, the input and output buffers
are assumed to be on the "primary device", which should be the device of
the first GRG.

Stream model
------------
All operations are expected to perform on or sync with the default stream
of the corresponding GRG's device.
"""

import cupy as xp
from cupy import cuda
from contextlib import contextmanager
from cupyx.scipy.sparse.linalg import LinearOperator
from pygrgl import TraversalDirection
from typing import List, Optional, Tuple, Union
import numpy

from grapp.grg_calculator import (
    GRGCalcInterface as _GRGCalcInterface,
    _wrap_grg,
)

# When True, cross-device copies are routed D2H + H2D instead of using
# cudaMemcpyPeer / NVLink.  Set before any operator _matmat call to benchmark
# PCIe-only transfers.
force_host_xdev_copy: bool = False

_DOWN = TraversalDirection.DOWN
_UP = TraversalDirection.UP


def _flip_dir(direction: TraversalDirection) -> TraversalDirection:
    return _UP if direction == _DOWN else _DOWN


def _transpose_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    return (shape[1], shape[0])


# ---------------------------------------------------------------------------
# NVTX helper
# ---------------------------------------------------------------------------


@contextmanager
def _nvtx(name: str):
    """Context manager that always pops an NVTX range, even on exception."""
    xp.cuda.nvtx.RangePush(name)
    try:
        yield
    finally:
        xp.cuda.nvtx.RangePop()


# ---------------------------------------------------------------------------
# Cross-device helpers
# ---------------------------------------------------------------------------


def _xdev_copy(dst, src) -> None:
    """Write src into dst, routing through host when force_host_xdev_copy is set.

    When copying across devices, the source device's current stream is
    synchronized first so that all kernels producing src are guaranteed
    complete before the D2D transfer begins.
    """
    if force_host_xdev_copy:
        with cuda.Device(dst.device):
            dst[:] = xp.asarray(src.get())
    else:
        if (
            hasattr(src, "device")
            and hasattr(dst, "device")
            and src.device != dst.device
        ):
            with cuda.Device(src.device):
                xp.cuda.get_current_stream().synchronize()
        with cuda.Device(dst.device):
            dst[:] = src
            xp.cuda.get_current_stream().synchronize()


def _xdev_asarray(src):
    """Return src as a CuPy array on the current device, copying if necessary."""
    if force_host_xdev_copy and hasattr(src, "get"):
        return xp.asarray(src.get())
    dev = getattr(src, "device", None)

    if dev is not None and hasattr(dev, "id") and dev.id != xp.cuda.Device().id:
        with cuda.Device(dev):
            src = xp.ascontiguousarray(src)
            xp.cuda.get_current_stream().synchronize()
        return xp.array(src)  # explicit D2D copy to current device
    return xp.asarray(src)


# ---------------------------------------------------------------------------
# GRGOpFilter
# ---------------------------------------------------------------------------


class GRGOpFilter:
    """Handles optional mutation and sample sub-selection for GRG operators."""

    def __init__(
        self,
        grg: "_GRGCalcInterface",
        haploid: bool,
        mutation_filter: Optional[Union[List[int], numpy.ndarray, "xp.ndarray"]],
        sample_filter: Optional[Union[List[int], numpy.ndarray, "xp.ndarray"]],
    ):
        if mutation_filter is not None:
            if isinstance(mutation_filter, (numpy.ndarray, xp.ndarray)):
                mutation_filter = mutation_filter.tolist()
            assert len(set(mutation_filter)) == len(
                mutation_filter
            ), "Duplicate IDs in mutation_filter"
        self.mutation_filter = mutation_filter

        if sample_filter is not None:
            if isinstance(sample_filter, (numpy.ndarray, xp.ndarray)):
                sample_filter = sample_filter.tolist()
            assert len(set(sample_filter)) == len(
                sample_filter
            ), "Duplicate IDs in sample_filter"
        self.sample_filter = sample_filter

        sample_count = grg.num_samples if haploid else grg.num_individuals
        self.grg_shape = (sample_count, grg.num_mutations)
        self.shape = (
            self.grg_shape[0] if sample_filter is None else len(sample_filter),
            self.grg_shape[1] if mutation_filter is None else len(mutation_filter),
        )
        self.is_filtering = (
            self.sample_filter is not None or self.mutation_filter is not None
        )
        self._device = getattr(grg, "device", None)

    def prep_input(self, input_matrix, mult_dir: TraversalDirection):
        """Zero-pad input_matrix to the full GRG dimension, scattering filtered indices."""
        if mult_dir == _DOWN:
            if self.mutation_filter is not None:
                with cuda.Device(self._device):
                    new_matrix = xp.zeros(
                        (input_matrix.shape[0], self.grg_shape[1]),
                        dtype=input_matrix.dtype,
                    )
                    new_matrix[:, self.mutation_filter] = input_matrix
                return new_matrix
        else:
            assert mult_dir == _UP
            if self.sample_filter is not None:
                with cuda.Device(self._device):
                    new_matrix = xp.zeros(
                        (input_matrix.shape[0], self.grg_shape[0]),
                        dtype=input_matrix.dtype,
                    )
                    new_matrix[:, self.sample_filter] = input_matrix
                return new_matrix
        return input_matrix

    def adjust_output(self, output_matrix, mult_dir: TraversalDirection):
        """Select only the filtered indices from the output matrix."""
        if mult_dir == _UP:
            if self.mutation_filter is not None:
                return output_matrix[:, self.mutation_filter]
        else:
            assert mult_dir == _DOWN
            if self.sample_filter is not None:
                return output_matrix[:, self.sample_filter]
        return output_matrix


# ---------------------------------------------------------------------------
# Non-standardized operators
# ---------------------------------------------------------------------------


class CuPyXOperator(LinearOperator):
    """
    LinearOperator on the genotype matrix X (NxM) or its transpose X^T (MxN).

    direction=UP   → shape (N, M), _matmat computes X @ A
    direction=DOWN → shape (M, N), _matmat computes X^T @ A

    :param grg: The GRG to multiply against. Must expose a `device` attribute.
    :param direction: Selects X (UP) or X^T (DOWN).
    :param dtype: Output dtype.
    :param haploid: When True use {0,1} haploid matrix; otherwise {0..ploidy}.
    :param miss_values: 1-D vector of per-mutation imputation values for
        missing data. Ignored when the GRG has no missing data.
    :param mutation_filter: Subset of mutation IDs to expose; shapes X to NxP.
    :param sample_filter: Subset of sample IDs to expose; shapes X to QxM.
    """

    def __init__(
        self,
        grg: _GRGCalcInterface,
        direction: TraversalDirection,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
    ):
        self.grg = _wrap_grg(grg)
        self._device = getattr(self.grg, "device", None)
        if self._device is None:
            raise ValueError(
                "CuPyXOperator requires grg.device to be set (not None). "
                "Use a GPU-backed GRG such as GRGSpMVCalculator."
            )
        self.filter = GRGOpFilter(self.grg, haploid, mutation_filter, sample_filter)
        self.haploid = haploid
        self.direction = direction
        assert (
            miss_values is None or miss_values.ndim == 1
        ), '"miss_values" must be a 1-D vector'
        with cuda.Device(self._device):
            self.miss_values = (
                xp.asarray(miss_values) if miss_values is not None else None
            )
        shape = self.filter.shape
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(self, other_matrix, mult_dir: TraversalDirection, out=None):
        label = "UP" if mult_dir == _UP else "DOWN"
        with _nvtx(f"XOp_{label}"):
            with cuda.Device(self._device):
                # Move input to self._device; no-op if already there.
                A = self.filter.prep_input(_xdev_asarray(other_matrix).T, mult_dir)

                kwargs = {}
                use_M = self.grg.has_missing_data and self.miss_values is not None
                if use_M:
                    if mult_dir == _DOWN:
                        M = xp.array([self.miss_values]) * A
                    else:
                        M = xp.zeros(
                            (A.shape[0], self.grg.num_mutations), dtype=self.dtype
                        )
                    kwargs["miss"] = M

                with cuda.Device(self._device):
                    result = self.grg.get_raw().matmul(
                        A,
                        self.grg._convert_dir(mult_dir),
                        by_individual=not self.haploid,
                        **kwargs,
                    )

                if mult_dir == _UP and use_M:
                    result += M * self.miss_values

                result = self.filter.adjust_output(result, mult_dir).T

                if out is not None:
                    # Sync self._device stream; queue copy on out.device's null stream.
                    _xdev_copy(out, result)
                    return out
            return result

    def _matmat(self, other_matrix, out=None):
        return self._matmat_helper(other_matrix, _flip_dir(self.direction), out)

    def _rmatmat(self, other_matrix, out=None):
        return self._matmat_helper(other_matrix, self.direction, out)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class CuPyXTXOperator(LinearOperator):
    """LinearOperator for X^T X (MxM non-centred correlation matrix)."""

    def __init__(
        self,
        grg: _GRGCalcInterface,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
    ):
        self.x_op = CuPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
        )
        self._device = self.x_op._device
        super().__init__(dtype=dtype, shape=(self.x_op.shape[1], self.x_op.shape[1]))

    def _matmat(self, other_matrix):
        with _nvtx("XTXOp_matmat"):
            D = self.x_op._matmat(other_matrix)
            return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class CuPyXXTOperator(LinearOperator):
    """LinearOperator for X X^T (NxN genetic relatedness matrix)."""

    def __init__(
        self,
        grg: _GRGCalcInterface,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
    ):
        self.x_op = CuPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
        )
        self._device = self.x_op._device
        self.grg = self.x_op.grg
        super().__init__(dtype=dtype, shape=(self.x_op.shape[0], self.x_op.shape[0]))

    def _matmat(self, other_matrix, out=None):
        with _nvtx("XXTOp_matmat"):
            D = self.x_op._rmatmat(other_matrix)
            result = self.x_op._matmat(D)
            if out is not None:
                _xdev_copy(out, result)
                return out
            return result

    def _rmatmat(self, other_matrix, out=None):
        return self._matmat(other_matrix, out)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


# ---------------------------------------------------------------------------
# Standardized operators
# ---------------------------------------------------------------------------


class _CuPyStandardizedOperator(LinearOperator):
    """
    Base class that pre-computes and stores allele-frequency-derived scalings
    on the GRG's device.

    Scaling: inverse_sigma[i] = sqrt(variance[i]^alpha), where variance is
    binomial by default (mult_const * freq * (1-freq)).  Entries where
    variance == 0 (fixed alleles) remain 0 to avoid NaN.

    :param freqs: 1-D allele frequency vector of length M.
    :param alpha: Power applied to variance; default -1 gives 1/sigma scaling.
    :param custom_variance: Replace binomial variance with caller-supplied values.
    """

    def __init__(
        self,
        grg: _GRGCalcInterface,
        freqs: numpy.ndarray,
        shape: Tuple[int, int],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        alpha: float = -1,
        custom_variance: Optional[numpy.ndarray] = None,
    ):
        self.haploid = haploid
        self.grg = _wrap_grg(grg)
        self._device = getattr(self.grg, "device", None)
        if self._device is None:
            raise ValueError(
                "Standardized CuPy operators require grg.device to be set (not None)."
            )
        self.mult_const = 1 if haploid else self.grg.ploidy

        with cuda.Device(self._device):
            self.freqs = xp.asarray(freqs)
            if custom_variance is not None:
                assert (
                    custom_variance.shape == freqs.shape
                ), "custom_variance must have the same shape as freqs"
                variance = xp.asarray(custom_variance)
            else:
                variance = self.mult_const * self.freqs * (1.0 - self.freqs)

            mask = variance != 0
            self.inverse_sigma = xp.zeros(variance.shape, dtype=variance.dtype)
            self.inverse_sigma[mask] = xp.sqrt(xp.power(variance[mask], alpha))

        super().__init__(dtype=dtype, shape=shape)


class CuPyStdXOperator(_CuPyStandardizedOperator):
    """
    LinearOperator on the standardized genotype matrix X or X^T.

    Standardization centers each column by its mean (mult_const * freq) and
    scales by 1/sigma, where sigma = sqrt(variance^alpha).

    direction=UP   → shape (N, M), _matmat computes stdX @ A
    direction=DOWN → shape (M, N), _matmat computes stdX^T @ A

    :param direction: Selects standardized X (UP) or its transpose (DOWN).
    :param freqs: Per-mutation allele frequencies, shape (M,).
    :param mutation_filter: Subset of mutation IDs; shapes X to NxP.
    :param sample_filter: Subset of sample IDs; shapes X to QxM.
    :param alpha: See _CuPyStandardizedOperator.
    :param custom_variance: See _CuPyStandardizedOperator.
    """

    def __init__(
        self,
        grg: _GRGCalcInterface,
        direction: TraversalDirection,
        freqs: numpy.ndarray,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.ndarray] = None,
    ):
        self.filter = GRGOpFilter(
            _wrap_grg(grg), haploid, mutation_filter, sample_filter
        )
        self.direction = direction
        shape = self.filter.shape
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(
            grg,
            freqs,
            shape,
            dtype=dtype,
            haploid=haploid,
            alpha=alpha,
            custom_variance=custom_variance,
        )

    def _matmat_direction(self, other_matrix, direction: TraversalDirection, out=None):
        label = "UP" if direction == _UP else "DOWN"
        with _nvtx(f"StdXOp_{label}"):
            with cuda.Device(self._device):
                # Move input to self._device; no-op if already there.
                other_matrix = _xdev_asarray(other_matrix)
                mult_dir = _flip_dir(direction)

                if direction == _UP:
                    with _nvtx("StdX_prep_scale"):
                        # prep_input allocates on self._device (inside _device_ctx).
                        # inverse_sigma and freqs are on self._device.
                        vS = (
                            self.filter.prep_input(other_matrix.T, mult_dir)
                            * self.inverse_sigma
                        )

                    with _nvtx("StdX_matmul"):
                        with cuda.Device(self._device):
                            XvS = self.grg.get_raw().matmul(
                                vS,
                                self.grg._convert_dir(mult_dir),
                                by_individual=not self.haploid,
                            )
                            xp.cuda.get_current_stream().synchronize()

                    with _nvtx("StdX_post"):
                        with cuda.Device(self._device):
                            consts = xp.sum(
                                self.mult_const * self.freqs * vS, axis=1, keepdims=True
                            )
                            result = self.filter.adjust_output(XvS - consts, mult_dir).T

                else:  # DOWN
                    with _nvtx("StdX_prep"):
                        m = self.filter.prep_input(other_matrix.T, mult_dir)

                    with _nvtx("StdX_matmul"):
                        # See UP branch: re-assert self._device around the native
                        # matmul so the post-matmul math stays on the right device.
                        with cuda.Device(self._device):
                            SXv_raw = self.grg.get_raw().matmul(
                                m,
                                self.grg._convert_dir(mult_dir),
                                by_individual=not self.haploid,
                            )
                            xp.cuda.get_current_stream().synchronize()
                            SXv = SXv_raw * self.inverse_sigma

                    with _nvtx("StdX_post"):
                        with cuda.Device(self._device):
                            col_const = xp.sum(m.T, axis=0, keepdims=True).T
                            sub_const2 = (
                                self.mult_const * self.freqs * self.inverse_sigma
                            ) * col_const
                            result = self.filter.adjust_output(
                                SXv - sub_const2, mult_dir
                            ).T

                if out is not None:
                    # Sync self._device stream; queue copy on out.device's null stream.
                    _xdev_copy(out, result)
                    return out
                # out is None: ensure the post-matmul scaling/centering kernels have
                # completed on self._device before any (possibly cross-device) consumer
                # reads `result`.
                xp.cuda.get_current_stream().synchronize()
            return result

    def _matmat(self, other_matrix, out=None):
        return self._matmat_direction(other_matrix, self.direction, out)

    def _rmatmat(self, other_matrix, out=None):
        return self._matmat_direction(other_matrix, _flip_dir(self.direction), out)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class CuPyStdXTXOperator(LinearOperator):
    """LinearOperator for the standardized X^T X (MxM correlation matrix)."""

    def __init__(
        self,
        grg: _GRGCalcInterface,
        freqs: numpy.ndarray,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.ndarray] = None,
    ):
        self.std_x_op = CuPyStdXOperator(
            grg,
            _UP,
            freqs,
            dtype=dtype,
            haploid=haploid,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        self._device = self.std_x_op._device
        super().__init__(
            dtype=dtype, shape=(self.std_x_op.shape[1], self.std_x_op.shape[1])
        )

    def _matmat(self, other_matrix):
        with _nvtx("StdXTXOp_matmat"):
            D = self.std_x_op._matmat(other_matrix)
            return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class CuPyStdXXTOperator(LinearOperator):
    """LinearOperator for the standardized X X^T (NxN genetic relatedness matrix)."""

    def __init__(
        self,
        grg: _GRGCalcInterface,
        freqs: numpy.ndarray,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.ndarray] = None,
    ):
        self.std_x_op = CuPyStdXOperator(
            grg,
            _UP,
            freqs,
            dtype=dtype,
            haploid=haploid,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        self._device = self.std_x_op._device
        self.grg = self.std_x_op.grg
        super().__init__(
            dtype=dtype, shape=(self.std_x_op.shape[0], self.std_x_op.shape[0])
        )

    def _matmat(self, other_matrix, out=None):
        with _nvtx("StdXXTOp_matmat"):
            D = self.std_x_op._rmatmat(other_matrix)
            result = self.std_x_op._matmat(D)
            if out is not None:
                _xdev_copy(out, result)
                return out
            return result

    def _rmatmat(self, other_matrix, out=None):
        return self._matmat(other_matrix, out)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


# ---------------------------------------------------------------------------
# Multi-GRG operators
# ---------------------------------------------------------------------------


def _build_per_grg_mut_filt(mutation_filter, prev_max_mut, g):
    """Remap global mutation_filter indices to per-GRG local indices."""
    if mutation_filter is None:
        return None, False
    grg_mut_filt = list(
        map(
            lambda m: m - prev_max_mut,
            filter(
                lambda m: m >= prev_max_mut and m < prev_max_mut + g.num_mutations,
                mutation_filter,
            ),
        )
    )
    return grg_mut_filt, len(grg_mut_filt) == 0


class MultiCuPyXOperator(LinearOperator):
    """
    LinearOperator on multiple GRGs. If GRGs have mutation counts M1, M2, ...,
    the implicit genotype matrix has dimensions Nx(M1+M2+...).

    :param grgs: GRGs with the same samples (e.g., one per chromosome).
    :param direction: UP -> X (NxM); DOWN -> X^T (MxN).
    :param dtype: Output dtype.
    :param haploid: Use {0,1} haploid matrix.
    :param miss_values: Per-mutation imputation vector (all GRGs concatenated).
    :param mutation_filter: Global mutation indices to expose; remapped per GRG.
    :param sample_filter: Sample indices to expose; applied to all GRGs.
    :param threads: Thread-pool size for parallelising across GRGs.
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        direction: TraversalDirection,
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        self.direction = direction
        self.operators: List[CuPyXOperator] = []
        prev_miss_start = 0
        prev_max_mut = 0
        for g in grgs:
            assert (
                g.num_samples == grgs[0].num_samples
            ), "All GRGs must use the same samples"
            grg_mut_filt, skip = _build_per_grg_mut_filt(
                mutation_filter, prev_max_mut, g
            )
            if not skip:
                effective_muts = (
                    len(grg_mut_filt) if grg_mut_filt is not None else g.num_mutations
                )
                if miss_values is not None:
                    grg_miss = miss_values[
                        prev_miss_start : prev_miss_start + effective_muts
                    ]
                    prev_miss_start += effective_muts
                else:
                    grg_miss = None
                self.operators.append(
                    CuPyXOperator(
                        g,
                        direction,
                        dtype,
                        haploid=haploid,
                        miss_values=grg_miss,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                    )
                )
            prev_max_mut += g.num_mutations
        self._output_device = self.operators[0]._device
        self.scheduler = _wrap_grg(grgs[0]).make_scheduler(grgs, threads)
        if direction == _UP:
            shape = (
                self.operators[0].shape[0],
                sum(op.shape[1] for op in self.operators),
            )
        else:
            shape = (
                sum(op.shape[0] for op in self.operators),
                self.operators[0].shape[1],
            )
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(self, other_matrix, direction, op_method):
        k = other_matrix.shape[1]
        label = "UP" if direction == _UP else "DOWN"
        futures = []
        with _nvtx(f"MultiXOp_{label}"):
            if direction == _UP:
                # Reduce branch: split the input along each GRG's mutation axis
                # and sum the per-GRG sample-space outputs. The mutation/sample
                # axes of each sub-operator depend on self.direction (UP exposes
                # shape (N, M_i); DOWN exposes shape (M_i, N)).
                op_muts = (
                    (lambda op: op.shape[0])
                    if self.direction == _DOWN
                    else (lambda op: op.shape[1])
                )
                n_rows = (
                    self.operators[0].shape[1]
                    if self.direction == _DOWN
                    else self.operators[0].shape[0]
                )
                parts = []
                for _ in self.operators:
                    with cuda.Device(self._output_device):
                        parts.append(xp.empty((n_rows, k), dtype=self.dtype))
                start = 0
                for op, part in zip(self.operators, parts):
                    end = start + op_muts(op)
                    sub = other_matrix[start:end, :]
                    with cuda.Device(op._device):
                        sub = _xdev_asarray(sub)
                    futures.append(
                        self.scheduler.submit(op.grg, op_method, op, sub, part)
                    )
                    start = end
                returned_parts = []
                for f in futures:
                    returned_parts.append(f.result())
                with cuda.Device(self._output_device):
                    xp.cuda.get_current_stream().synchronize()
                    result = sum(returned_parts[1:], returned_parts[0])
                    # Sync so that sum kernels complete before any cross-device read of result.
                    xp.cuda.get_current_stream().synchronize()
                return result
            else:
                # Each op gets the full input; outputs are concatenated.
                op_rows = (
                    (lambda op: op.shape[0])
                    if self.direction == _DOWN
                    else (lambda op: op.shape[1])
                )
                total_rows = sum(op_rows(op) for op in self.operators)
                with cuda.Device(self._output_device):
                    output = xp.empty((total_rows, k), dtype=self.dtype)
                start = 0
                for op in self.operators:
                    end = start + op_rows(op)
                    out_slice = output[start:end, :]
                    with cuda.Device(op._device):
                        op_input = _xdev_asarray(other_matrix)
                    futures.append(
                        self.scheduler.submit(
                            op.grg, op_method, op, op_input, out_slice
                        )
                    )
                    start = end
                for f in futures:
                    f.result()
                # Sync so that in-flight D2D copies to output complete before any
                # cross-device read of output (e.g. by the XTX DOWN pass).
                with cuda.Device(self._output_device):
                    xp.cuda.get_current_stream().synchronize()
                return output

    def _matmat(self, other_matrix):
        return self._matmat_helper(other_matrix, self.direction, CuPyXOperator._matmat)

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, _flip_dir(self.direction), CuPyXOperator._rmatmat
        )

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class MultiCuPyXTXOperator(LinearOperator):
    """
    LinearOperator for X^T X across multiple GRGs (MxM, symmetric).
    Equivalent to CuPyXTXOperator but across a list of GRGs sharing the same samples.
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
    ):
        self.x_op = MultiCuPyXOperator(
            grgs,
            _UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            threads=threads,
        )
        self._output_device = self.x_op._output_device
        super().__init__(dtype=dtype, shape=(self.x_op.shape[1], self.x_op.shape[1]))

    def _matmat(self, other_matrix):
        with _nvtx("MultiXTXOp_matmat"):
            D = self.x_op._matmat(other_matrix)
            return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class MultiCuPyXXTOperator(LinearOperator):
    """
    LinearOperator for X X^T across multiple GRGs (NxN, symmetric).

    Dispatches to one CuPyXXTOperator per GRG and sums N*K results on
    _output_device, costing 2 D2D copies per GRG (broadcast input + gather
    result) instead of 4 (the old Multi-X approach required an intermediate
    M*K transfer in each direction).
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.ndarray] = None,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        self.operators: List[CuPyXXTOperator] = []
        prev_miss_start = 0
        prev_max_mut = 0
        for g in grgs:
            assert (
                g.num_samples == grgs[0].num_samples
            ), "All GRGs must use the same samples"
            grg_mut_filt, skip = _build_per_grg_mut_filt(
                mutation_filter, prev_max_mut, g
            )
            if not skip:
                effective_muts = (
                    len(grg_mut_filt) if grg_mut_filt is not None else g.num_mutations
                )
                if miss_values is not None:
                    grg_miss = miss_values[
                        prev_miss_start : prev_miss_start + effective_muts
                    ]
                    prev_miss_start += effective_muts
                else:
                    grg_miss = None
                self.operators.append(
                    CuPyXXTOperator(
                        g,
                        dtype,
                        haploid=haploid,
                        miss_values=grg_miss,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                    )
                )
            prev_max_mut += g.num_mutations
        self._output_device = self.operators[0]._device
        self.scheduler = _wrap_grg(grgs[0]).make_scheduler(grgs, threads)
        n = self.operators[0].shape[0]
        super().__init__(dtype=dtype, shape=(n, n))

    def _matmat(self, other_matrix):
        n, k = self.shape[0], other_matrix.shape[1]
        futures = []
        with _nvtx("MultiXXTOp_matmat"):
            parts = []
            for _ in self.operators:
                with cuda.Device(self._output_device):
                    parts.append(xp.empty((n, k), dtype=self.dtype))
            for op, part in zip(self.operators, parts):
                futures.append(
                    self.scheduler.submit(
                        op.grg, CuPyXXTOperator._matmat, op, other_matrix, part
                    )
                )
            for f in futures:
                f.result()
            with cuda.Device(self._output_device):
                result = sum(parts[1:], parts[0])
                xp.cuda.get_current_stream().synchronize()
            return result

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class MultiCuPyStdXOperator(LinearOperator):
    """
    LinearOperator on the standardized genotype matrix across multiple GRGs.

    :param grgs: GRGs sharing the same samples.
    :param direction: UP -> stdX; DOWN -> stdX^T.
    :param freqs: List of per-GRG allele-frequency arrays (one per GRG).
    :param mutation_filter: Global mutation indices; remapped per GRG.
    :param threads: Thread-pool size.
    :param alpha: Variance power; default -1 gives 1/sigma scaling.
    :param custom_variance: Custom variance. Can be a single array of length num_mutations
        (applied to all GRGs) or a list of per-GRG arrays.
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        direction: TraversalDirection,
        freqs: List[numpy.ndarray],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[Union[numpy.ndarray, List[numpy.ndarray]]] = None,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        assert len(grgs) == len(freqs), "Must provide allele frequencies for every GRG"
        if isinstance(custom_variance, list):
            assert len(custom_variance) == len(
                grgs
            ), "custom_variance list must have one entry per GRG"
        self.direction = direction
        self.operators: List[CuPyStdXOperator] = []
        prev_max_mut = 0
        for i, (g, f) in enumerate(zip(grgs, freqs)):
            assert (
                g.num_samples == grgs[0].num_samples
            ), "All GRGs must use the same samples"
            grg_mut_filt, skip = _build_per_grg_mut_filt(
                mutation_filter, prev_max_mut, g
            )
            grg_custom_var = (
                custom_variance[i]
                if isinstance(custom_variance, list)
                else custom_variance
            )
            if not skip:
                self.operators.append(
                    CuPyStdXOperator(
                        g,
                        direction,
                        f,
                        dtype,
                        haploid=haploid,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                        alpha=alpha,
                        custom_variance=grg_custom_var,
                    )
                )
            prev_max_mut += g.num_mutations
        self._output_device = self.operators[0]._device
        self.scheduler = _wrap_grg(grgs[0]).make_scheduler(grgs, threads)
        if direction == _UP:
            shape = (
                self.operators[0].shape[0],
                sum(op.shape[1] for op in self.operators),
            )
        else:
            shape = (
                sum(op.shape[0] for op in self.operators),
                self.operators[0].shape[1],
            )
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(self, other_matrix, direction, op_method):
        k = other_matrix.shape[1]
        label = "UP" if direction == _UP else "DOWN"
        futures = []
        with _nvtx(f"MultiStdXOp_{label}"):
            if direction == _UP:
                # Reduce branch: split the input along each GRG's mutation axis
                # and sum the per-GRG sample-space outputs. The mutation/sample
                # axes of each sub-operator depend on self.direction (UP exposes
                # shape (N, M_i); DOWN exposes shape (M_i, N)).
                op_muts = (
                    (lambda op: op.shape[0])
                    if self.direction == _DOWN
                    else (lambda op: op.shape[1])
                )
                n_rows = (
                    self.operators[0].shape[1]
                    if self.direction == _DOWN
                    else self.operators[0].shape[0]
                )
                parts = []
                for _ in self.operators:
                    with cuda.Device(self._output_device):
                        parts.append(xp.empty((n_rows, k), dtype=self.dtype))
                start = 0
                for op, part in zip(self.operators, parts):
                    end = start + op_muts(op)
                    sub = other_matrix[start:end, :]
                    with cuda.Device(op._device):
                        sub = _xdev_asarray(sub)
                    futures.append(
                        self.scheduler.submit(op.grg, op_method, op, sub, part)
                    )
                    start = end
                for f in futures:
                    f.result()
                with cuda.Device(self._output_device):
                    result = sum(parts[1:], parts[0])
                    xp.cuda.get_current_stream().synchronize()
                return result
            else:
                op_rows = (
                    (lambda op: op.shape[0])
                    if self.direction == _DOWN
                    else (lambda op: op.shape[1])
                )
                total_rows = sum(op_rows(op) for op in self.operators)
                with cuda.Device(self._output_device):
                    output = xp.empty((total_rows, k), dtype=self.dtype)
                start = 0
                for op in self.operators:
                    end = start + op_rows(op)
                    out_slice = output[start:end, :]
                    with cuda.Device(op._device):
                        op_input = _xdev_asarray(other_matrix)
                    futures.append(
                        self.scheduler.submit(
                            op.grg, op_method, op, op_input, out_slice
                        )
                    )
                    start = end
                for f in futures:
                    f.result()
                with cuda.Device(self._output_device):
                    xp.cuda.get_current_stream().synchronize()
                return output

    def _matmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, self.direction, CuPyStdXOperator._matmat
        )

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, _flip_dir(self.direction), CuPyStdXOperator._rmatmat
        )

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class MultiCuPyStdXTXOperator(LinearOperator):
    """
    LinearOperator for standardized X^T X across multiple GRGs (MxM, symmetric).
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        freqs: List[numpy.ndarray],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[numpy.ndarray] = None,
    ):
        self.std_x_op = MultiCuPyStdXOperator(
            grgs,
            _UP,
            freqs,
            dtype=dtype,
            haploid=haploid,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            threads=threads,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        self._output_device = self.std_x_op._output_device
        super().__init__(
            dtype=dtype, shape=(self.std_x_op.shape[1], self.std_x_op.shape[1])
        )

    def _matmat(self, other_matrix):
        with _nvtx("MultiStdXTXOp_matmat"):
            D = self.std_x_op._matmat(other_matrix)
            return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)


class MultiCuPyStdXXTOperator(LinearOperator):
    """
    LinearOperator for standardized X X^T across multiple GRGs (NxN, symmetric).

    Uses one CuPyStdXXTOperator per GRG and sums results on _output_device,
    costing 2 D2D copies per GRG instead of 4 (see MultiCuPyXXTOperator).

    :param freqs: List of per-GRG allele-frequency arrays.
    :param custom_variance: Per-GRG custom variance. Can be a single array of length
        num_mutations (applied to all GRGs) or a list of per-GRG arrays.
    """

    def __init__(
        self,
        grgs: List[_GRGCalcInterface],
        freqs: List[numpy.ndarray],
        dtype: numpy.typing.DTypeLike = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        sample_filter: Optional[Union[List[int], numpy.ndarray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[Union[numpy.ndarray, List[numpy.ndarray]]] = None,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        assert len(grgs) == len(freqs), "Must provide allele frequencies for every GRG"
        if isinstance(custom_variance, list):
            assert len(custom_variance) == len(
                grgs
            ), "custom_variance list must have one entry per GRG"
        self.operators: List[CuPyStdXXTOperator] = []
        prev_max_mut = 0
        for i, (g, f) in enumerate(zip(grgs, freqs)):
            assert (
                g.num_samples == grgs[0].num_samples
            ), "All GRGs must use the same samples"
            grg_mut_filt, skip = _build_per_grg_mut_filt(
                mutation_filter, prev_max_mut, g
            )
            grg_custom_var = (
                custom_variance[i]
                if isinstance(custom_variance, list)
                else custom_variance
            )
            if not skip:
                self.operators.append(
                    CuPyStdXXTOperator(
                        g,
                        f,
                        dtype,
                        haploid=haploid,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                        alpha=alpha,
                        custom_variance=grg_custom_var,
                    )
                )
            prev_max_mut += g.num_mutations
        self._output_device = self.operators[0]._device
        self.scheduler = _wrap_grg(grgs[0]).make_scheduler(grgs, threads, gated=True)
        # Operator indices to leave out of the next product (set via set_exclude).
        self._exclude: set = set()
        n = self.operators[0].shape[0]
        super().__init__(dtype=dtype, shape=(n, n))

    def set_exclude(self, exclude=None):
        """Set which operator indices (chromosomes) to leave out of subsequent
        products. Pass ``None`` or an empty list to include all chromosomes.
        Returns self for chaining."""
        if exclude is None:
            self._exclude = set()
        elif isinstance(exclude, int):
            self._exclude = {int(exclude)}
        else:
            self._exclude = {int(i) for i in exclude}
        for idx in self._exclude:
            if idx < 0 or idx >= len(self.operators):
                raise IndexError(
                    f"exclude index {idx} out of range for "
                    f"{len(self.operators)} operators"
                )
        return self

    def _matmat(self, other_matrix):
        n, k = self.shape[0], other_matrix.shape[1]
        with _nvtx("MultiStdXXTOp_matmat"):
            active = [
                (i, op) for i, op in enumerate(self.operators) if i not in self._exclude
            ]
            if not active:
                with cuda.Device(self._output_device):
                    return xp.zeros((n, k), dtype=self.dtype)
            parts = []
            for _ in active:
                with cuda.Device(self._output_device):
                    parts.append(xp.empty((n, k), dtype=self.dtype))
            self.scheduler.reset()
            futures = []
            for (_, op), part in zip(active, parts):
                futures.append(
                    self.scheduler.submit(
                        op.grg, CuPyStdXXTOperator._matmat, op, other_matrix, part
                    )
                )
            self.scheduler.start()
            for f in futures:
                f.result()
            with cuda.Device(self._output_device):
                cuda.Device(self._output_device).synchronize()
                result = sum(parts[1:], parts[0])
                xp.cuda.get_current_stream().synchronize()
            return result

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            with cuda.Device(self._output_device):
                vect = _xdev_asarray(vect).reshape(-1, 1)
        return self._rmatmat(vect)
