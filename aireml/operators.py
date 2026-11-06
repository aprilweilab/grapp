"""
Helpers for building the ``scipy.sparse.linalg.LinearOperator`` objects that
:mod:`aireml` consumes.

The AI-REML solver never looks inside an operator: it only ever needs
``operator @ vector`` (and ``operator @ matrix``).  Anything that can supply
matrix-vector products -- a dense array, a sparse matrix, a genotype matrix
held on disk, an ARG or GRG traversal -- can therefore be plugged in.
"""

from typing import Callable, Optional, Sequence, Union

import numpy
import numpy.typing
from scipy.sparse.linalg import LinearOperator, aslinearoperator

#: What the helpers here accept in place of a matrix.  :class:`LinearOperator`
#: is the case that matters -- it is how an ARG, a GRG, or a memory-mapped
#: genotype file gets in without ever being materialized.  Dense arrays are
#: accepted too because they are convenient for small problems and tests, and
#: ``aslinearoperator`` additionally handles scipy sparse matrices, so callers
#: rarely have to wrap anything themselves.
ArrayLike = Union[numpy.typing.NDArray, LinearOperator]

#: Variance-component values: anything array-like of floats.
Coefficients = Union[Sequence[float], numpy.typing.NDArray]


def _from_matmat(
    size: int,
    matmat: Callable[[numpy.typing.NDArray], numpy.typing.NDArray],
    dtype=numpy.float64,
) -> LinearOperator:
    """
    Wrap a block-product function as a symmetric ``size x size`` operator.

    Every operator in this module is symmetric and defined by a single
    block-product function, so this collapses the five-way
    ``matvec``/``matmat``/``rmatvec``/``rmatmat``/``dtype`` boilerplate that
    :class:`LinearOperator` otherwise needs at each construction site.

    ``pylops.FunctionOperator`` looks like a drop-in replacement but is not:
    it supplies only ``_matvec``, so the inherited ``matmat`` falls back to
    looping over columns one at a time.  Measured at ``N = 4000`` on a
    50-column block, that is 8.70 ms against 2.00 ms here -- a 4.4x penalty
    on exactly the operation this package leans on, since trace estimation
    and the Nystrom sketch apply operators to 30-100 columns at once.  The
    cost is worse for a real ARG or GRG operator, where one block product
    amortizes a graph traversal across every column.  pylops' operator
    algebra (``a * Op1 + b * Op2``) degrades the same way.  Keeping the
    explicit block ``matmat`` here is deliberate.
    """

    def _as_vector(vector: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return matmat(numpy.asarray(vector).reshape(-1, 1)).ravel()

    return LinearOperator(
        shape=(size, size),
        matvec=_as_vector,
        matmat=matmat,
        rmatvec=_as_vector,
        rmatmat=matmat,
        dtype=dtype,
    )


def operator_size(operator: LinearOperator) -> int:
    """Return ``N`` for an ``N x N`` operator, raising if it is not square."""
    rows, cols = operator.shape
    if rows != cols:
        raise ValueError(f"Expected a square operator, got shape {operator.shape}")
    return int(rows)


def as_symmetric_operator(matrix: ArrayLike, dtype=numpy.float64) -> LinearOperator:
    """
    Coerce ``matrix`` into a square :class:`LinearOperator` of the given dtype.

    Squareness is checked, since that is cheap.  Symmetry and positive
    semi-definiteness are *assumed* and never verified: unlike shape, they are
    not carried by the operator and establishing them would mean materializing
    the matrix, which is the whole thing this package avoids.  Supplying an
    asymmetric or indefinite operator produces wrong variance components
    rather than an error.
    """
    base = aslinearoperator(matrix)
    size = operator_size(base)
    if base.dtype == dtype:
        return base

    def _matmat(other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return numpy.asarray(base.matmat(other)).astype(dtype, copy=False)

    return _from_matmat(size, _matmat, dtype=dtype)


def identity_operator(size: int, dtype=numpy.float64) -> LinearOperator:
    """The ``size x size`` identity, as a :class:`LinearOperator`."""

    def _matmat(other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return numpy.asarray(other).astype(dtype, copy=False)

    return _from_matmat(size, _matmat, dtype=dtype)


class LinearCombinationOperator(LinearOperator):
    """
    ``sum_i coefficient[i] * component[i]``, evaluated lazily.

    Used to form the phenotypic covariance ``V = sum_i theta_i V_i`` without
    materializing it.  Construction is pure bookkeeping -- it wraps the
    components and stores the coefficients, touching no data -- so callers
    build a fresh instance whenever the variance components move rather than
    mutating one in place.
    """

    def __init__(
        self,
        components: Sequence[LinearOperator],
        coefficients: Optional[Coefficients] = None,
        dtype=numpy.float64,
    ):
        if len(components) == 0:
            raise ValueError("Need at least one component")
        self.components = [as_symmetric_operator(op, dtype=dtype) for op in components]
        size = operator_size(self.components[0])
        for component in self.components[1:]:
            if component.shape != (size, size):
                raise ValueError(
                    f"Component shapes disagree: {component.shape} vs {(size, size)}"
                )
        super().__init__(dtype=dtype, shape=(size, size))
        values = (
            numpy.ones(len(self.components), dtype=dtype)
            if coefficients is None
            else numpy.asarray(coefficients, dtype=dtype)
        )
        if values.shape != (len(self.components),):
            raise ValueError("Need one coefficient per component")
        self.coefficients: numpy.typing.NDArray = values

    def _matmat(self, other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        result = numpy.zeros((self.shape[0], other.shape[1]), dtype=self.dtype)
        for coefficient, component in zip(self.coefficients, self.components):
            result += coefficient * numpy.asarray(component.matmat(other))
        return result

    def _matvec(self, other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        return self._matmat(numpy.asarray(other).reshape(-1, 1)).ravel()

    def _adjoint(self) -> LinearOperator:
        return self


def grm_from_genotypes(
    genotypes: ArrayLike,
    scale: Optional[float] = None,
    center: bool = False,
    column_means: Optional[numpy.typing.NDArray] = None,
) -> LinearOperator:
    """
    Build a relatedness operator ``Z Z^T / scale`` from an ``N x M`` operator.

    This is the bridge between :mod:`aireml` and any library that exposes a
    genotype matrix as a linear operator (``grapp``'s GRG-backed operators, a
    tree-sequence product, a memory-mapped ``.bed`` file, ...).  ``aireml``
    itself never needs to know where the genotypes came from.

    :param genotypes: ``N x M`` array or :class:`LinearOperator`.
    :param scale: Divisor applied to ``Z Z^T``; defaults to the number of
        columns ``M``.
    :param center: If True, use ``(Z - 1 mu^T)(Z - 1 mu^T)^T / scale``.
    :param column_means: Column means to use when ``center`` is True.  If not
        given they are computed with one pass (``Z^T 1 / N``).
    """
    operator = aslinearoperator(genotypes)
    num_rows, num_cols = operator.shape
    if scale is None:
        scale = float(num_cols)
    if scale <= 0:
        raise ValueError("scale must be positive")

    if center and column_means is None:
        column_means = operator.rmatvec(numpy.ones(num_rows)) / num_rows
    means = None if not center else numpy.asarray(column_means, dtype=numpy.float64)

    def _matmat(other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        # (Z - 1 mu^T) (Z - 1 mu^T)^T @ other, evaluated right to left.
        right = numpy.asarray(operator.rmatmat(other))
        if means is not None:
            right = right - numpy.outer(means, other.sum(axis=0))
        result = numpy.asarray(operator.matmat(right))
        if means is not None:
            result = result - numpy.tensordot(means, right, axes=(0, 0))
        return result / scale

    return _from_matmat(num_rows, _matmat)


def centered_operator(operator: ArrayLike) -> LinearOperator:
    """
    Row- and column-center a symmetric operator: ``C B C`` with
    ``C = I - 1 1^T / N``.

    This is the "centered GRM" of Lee et al. (2026), useful when relating the
    genetic variance component to an additive genetic variance.
    """
    base = as_symmetric_operator(operator)
    size = operator_size(base)

    def _matmat(other: numpy.typing.NDArray) -> numpy.typing.NDArray:
        centered = other - other.mean(axis=0, keepdims=True)
        product = numpy.asarray(base.matmat(centered))
        return product - product.mean(axis=0, keepdims=True)

    return _from_matmat(size, _matmat)


def materialize(operator: LinearOperator) -> numpy.typing.NDArray:
    """
    Apply ``operator`` to the identity, returning a dense array.

    Only for testing and small problems: this costs ``N`` matrix-vector
    products and ``O(N^2)`` memory.
    """
    base = aslinearoperator(operator)
    _, cols = base.shape
    return numpy.asarray(base.matmat(numpy.eye(cols, dtype=numpy.float64)))
