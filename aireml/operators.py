"""
Helpers for building the ``scipy.sparse.linalg.LinearOperator`` objects that
:mod:`aireml` consumes.

The AI-REML solver never looks inside an operator: it only ever needs
``operator @ vector`` (and ``operator @ matrix``).  Anything that can supply
matrix-vector products -- a dense array, a sparse matrix, a genotype matrix
held on disk, an ARG or GRG traversal -- can therefore be plugged in.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy
from scipy.sparse.linalg import LinearOperator, aslinearoperator

ArrayLike = Union[numpy.ndarray, LinearOperator]
#: Variance-component values: anything array-like of floats.
Coefficients = Union[Sequence[float], numpy.ndarray]

__all__ = [
    "as_symmetric_operator",
    "Coefficients",
    "centered_operator",
    "grm_from_genotypes",
    "identity_operator",
    "LinearCombinationOperator",
    "materialize",
    "operator_size",
]


def operator_size(operator: LinearOperator) -> int:
    """Return ``N`` for an ``N x N`` operator, raising if it is not square."""
    rows, cols = operator.shape
    if rows != cols:
        raise ValueError(f"Expected a square operator, got shape {operator.shape}")
    return int(rows)


def as_symmetric_operator(matrix: ArrayLike, dtype=numpy.float64) -> LinearOperator:
    """
    Coerce ``matrix`` into a square :class:`LinearOperator` of the given dtype.

    The caller is responsible for the operator actually being symmetric
    positive semi-definite; that is assumed throughout and is not checked,
    because checking would require materializing the matrix.
    """
    base = aslinearoperator(matrix)
    operator_size(base)
    if base.dtype == dtype:
        return base

    def _matmat(other: numpy.ndarray) -> numpy.ndarray:
        return numpy.asarray(base.matmat(other)).astype(dtype, copy=False)

    return LinearOperator(
        shape=base.shape,
        matvec=lambda x: _matmat(x.reshape(-1, 1)).ravel(),
        matmat=_matmat,
        rmatvec=lambda x: _matmat(x.reshape(-1, 1)).ravel(),
        rmatmat=_matmat,
        dtype=dtype,
    )


def identity_operator(size: int, dtype=numpy.float64) -> LinearOperator:
    """The ``size x size`` identity, as a :class:`LinearOperator`."""

    def _matmat(other: numpy.ndarray) -> numpy.ndarray:
        return numpy.asarray(other).astype(dtype, copy=False)

    return LinearOperator(
        shape=(size, size),
        matvec=lambda x: _matmat(x),
        matmat=_matmat,
        rmatvec=lambda x: _matmat(x),
        rmatmat=_matmat,
        dtype=dtype,
    )


class LinearCombinationOperator(LinearOperator):
    """
    ``sum_i coefficient[i] * operator[i]``, evaluated lazily.

    Used to form the phenotypic covariance ``V = sum_i theta_i V_i`` without
    materializing it.  The coefficients can be replaced in place (see
    :meth:`set_coefficients`), so a single instance is reused across AI-REML
    iterations.
    """

    def __init__(
        self,
        operators: Sequence[LinearOperator],
        coefficients: Optional[Coefficients] = None,
        dtype=numpy.float64,
    ):
        if len(operators) == 0:
            raise ValueError("Need at least one operator")
        self.operators = [as_symmetric_operator(op, dtype=dtype) for op in operators]
        size = operator_size(self.operators[0])
        for op in self.operators[1:]:
            if op.shape != (size, size):
                raise ValueError(
                    f"Operator shapes disagree: {op.shape} vs {(size, size)}"
                )
        super().__init__(dtype=dtype, shape=(size, size))
        self.coefficients: numpy.ndarray = numpy.zeros(len(self.operators), dtype=dtype)
        self.set_coefficients(
            numpy.ones(len(self.operators)) if coefficients is None else coefficients
        )

    def set_coefficients(self, coefficients: Coefficients) -> None:
        values = numpy.asarray(coefficients, dtype=self.dtype)
        if values.shape != self.coefficients.shape:
            raise ValueError("Need one coefficient per operator")
        self.coefficients = values

    def _matmat(self, other: numpy.ndarray) -> numpy.ndarray:
        result = numpy.zeros((self.shape[0], other.shape[1]), dtype=self.dtype)
        for coefficient, operator in zip(self.coefficients, self.operators):
            if coefficient == 0.0:
                continue
            result += coefficient * numpy.asarray(operator.matmat(other))
        return result

    def _matvec(self, other: numpy.ndarray) -> numpy.ndarray:
        return self._matmat(numpy.asarray(other).reshape(-1, 1)).ravel()

    def _adjoint(self) -> LinearOperator:
        return self


def grm_from_genotypes(
    genotypes: ArrayLike,
    scale: Optional[float] = None,
    center: bool = False,
    column_means: Optional[numpy.ndarray] = None,
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

    def _matmat(other: numpy.ndarray) -> numpy.ndarray:
        # (Z - 1 mu^T) (Z - 1 mu^T)^T @ other, evaluated right to left.
        right = numpy.asarray(operator.rmatmat(other))
        if means is not None:
            right = right - numpy.outer(means, other.sum(axis=0))
        result = numpy.asarray(operator.matmat(right))
        if means is not None:
            result = result - numpy.tensordot(means, right, axes=(0, 0))
        return result / scale

    return LinearOperator(
        shape=(num_rows, num_rows),
        matvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
        matmat=_matmat,
        rmatvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
        rmatmat=_matmat,
        dtype=numpy.float64,
    )


def centered_operator(operator: ArrayLike) -> LinearOperator:
    """
    Row- and column-center a symmetric operator: ``C B C`` with
    ``C = I - 1 1^T / N``.

    This is the "centered GRM" of Lee et al. (2026), useful when relating the
    genetic variance component to an additive genetic variance.
    """
    base = as_symmetric_operator(operator)
    size = operator_size(base)

    def _matmat(other: numpy.ndarray) -> numpy.ndarray:
        centered = other - other.mean(axis=0, keepdims=True)
        product = numpy.asarray(base.matmat(centered))
        return product - product.mean(axis=0, keepdims=True)

    return LinearOperator(
        shape=(size, size),
        matvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
        matmat=_matmat,
        rmatvec=lambda x: _matmat(numpy.asarray(x).reshape(-1, 1)).ravel(),
        rmatmat=_matmat,
        dtype=numpy.float64,
    )


def materialize(operator: LinearOperator) -> numpy.ndarray:
    """
    Apply ``operator`` to the identity, returning a dense array.

    Only for testing and small problems: this costs ``N`` matrix-vector
    products and ``O(N^2)`` memory.
    """
    base = aslinearoperator(operator)
    _, cols = base.shape
    return numpy.asarray(base.matmat(numpy.eye(cols, dtype=numpy.float64)))


def check_operator_shapes(operators: Sequence[LinearOperator]) -> Tuple[int, int]:
    """Validate that all operators are square and agree; return their shape."""
    if len(operators) == 0:
        raise ValueError("Need at least one operator")
    shape = aslinearoperator(operators[0]).shape
    for op in operators:
        if aslinearoperator(op).shape != shape:
            raise ValueError("Operator shapes disagree")
    if shape[0] != shape[1]:
        raise ValueError(f"Expected square operators, got {shape}")
    return shape
