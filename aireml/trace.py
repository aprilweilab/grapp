"""
Stochastic estimation of ``trace(A)`` for an implicitly defined operator.

AI-REML needs ``trace(P V_i)`` for each variance component, and those are the
only quantities in the gradient that cannot be reduced to a handful of
matrix-vector products (Lee et al. 2026, equation 16).  Three unbiased
estimators are provided:

* ``"hutchinson"`` -- the classical estimator (Hutchinson 1990).  Variance
  decays like ``1/m``.
* ``"hutchpp"`` -- Hutch++ (Meyer et al. 2021), Algorithm 1.  Splits the
  budget three ways: a sketch that captures the dominant eigenspace, an exact
  trace on that subspace, and a Hutchinson correction on its orthogonal
  complement.  Variance decays like ``1/m^2`` for matrices with decaying
  spectra.
* ``"xtrace"`` -- the exchangeable estimator of Epperly et al. (2024), which
  reuses every test vector both for the sketch and for the residual
  correction via a leave-one-out identity.  Also ``1/m^2``, usually with a
  smaller constant than Hutch++, but it needs two products per test vector.

**Matrix-vector cost.**  ``num_vectors`` is the *matrix-vector budget* for
``"hutchinson"`` and ``"hutchpp"`` (the latter matching the ``neval`` argument
of ``pylops.utils.estimators.trace_hutchpp``), but the *number of test
vectors* for ``"xtrace"``, which spends ``2 * num_vectors`` products.  Every
product here is a conjugate-gradient solve, so compare estimators at equal
``TraceEstimate.num_matvecs``, not at equal ``num_vectors``.

``"exact"`` materializes the operator and is only useful for testing.
"""

from typing import NamedTuple, Optional

import numpy
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator, aslinearoperator

__all__ = ["TraceEstimate", "estimate_trace", "rademacher"]


class TraceEstimate(NamedTuple):
    """Result of a stochastic trace estimate."""

    value: float
    #: Heuristic standard error.  For XTrace the per-vector estimates are
    #: correlated, so this understates the true error somewhat; Epperly et al.
    #: (2024) use the same heuristic.
    stderr: float
    #: Number of matrix-vector products consumed.
    num_matvecs: int


def rademacher(
    num_rows: int, num_cols: int, rng: numpy.random.Generator
) -> numpy.ndarray:
    """``num_rows x num_cols`` matrix of independent +/-1 entries."""
    return (
        rng.integers(0, 2, size=(num_rows, num_cols)).astype(numpy.float64) * 2.0 - 1.0
    )


def _hutchinson(
    operator: LinearOperator, num_vectors: int, rng: numpy.random.Generator
) -> TraceEstimate:
    size = operator.shape[0]
    test = rademacher(size, num_vectors, rng)
    products = numpy.asarray(operator.matmat(test))
    samples = numpy.einsum("ij,ij->j", test, products)
    return TraceEstimate(
        value=float(samples.mean()),
        stderr=(
            float(samples.std(ddof=1) / numpy.sqrt(num_vectors))
            if num_vectors > 1
            else float("nan")
        ),
        num_matvecs=num_vectors,
    )


def _xtrace(
    operator: LinearOperator, num_vectors: int, rng: numpy.random.Generator
) -> TraceEstimate:
    """
    XTrace (Epperly et al. 2024).

    For each test vector ``w_i`` let ``Q_{-i}`` be an orthonormal basis for the
    sketch built from all *other* test vectors.  The leave-one-out estimate

        t_i = trace(Q_{-i}^T A Q_{-i}) + w_i^T Pi_i A Pi_i w_i,
        Pi_i = I - Q_{-i} Q_{-i}^T

    is unbiased because ``w_i`` is independent of ``Q_{-i}``; XTrace averages
    the ``m`` of them.  All ``m`` leave-one-out bases are obtained from a
    single QR: if ``Y = A Omega = Q R`` then the unit vector in ``range(Y)``
    orthogonal to ``range(Y_{-i})`` is ``Q s_i / ||s_i||`` with ``s_i`` the
    i-th column of ``R^{-T}``, so ``Q_{-i} Q_{-i}^T = Q Q^T - u_i u_i^T``.
    """
    size = operator.shape[0]
    test = rademacher(size, num_vectors, rng)
    sketch = numpy.asarray(operator.matmat(test))  # Y = A @ Omega
    basis, upper = numpy.linalg.qr(sketch)  # Y = Q R

    diagonal = numpy.abs(numpy.diag(upper))
    if diagonal.min() <= numpy.finfo(float).eps * 8 * max(diagonal.max(), 1.0):
        # Rank-deficient sketch: the leave-one-out downdate is not defined.
        samples = numpy.einsum("ij,ij->j", test, sketch)
        return TraceEstimate(
            value=float(samples.mean()),
            stderr=float(samples.std(ddof=1) / numpy.sqrt(num_vectors)),
            num_matvecs=num_vectors,
        )

    # S = R^{-T} with unit-norm columns; column i spans range(Y) . range(Y_{-i}).
    inverse_t = solve_triangular(upper, numpy.eye(num_vectors), lower=False, trans="T")
    inverse_t /= numpy.linalg.norm(inverse_t, axis=0, keepdims=True)

    projected = numpy.asarray(operator.matmat(basis))  # Z = A @ Q
    small = basis.T @ projected  # H = Q^T A Q
    weights = basis.T @ test  # W = Q^T Omega
    cross = projected.T @ test  # T, T[j, i] = w_i^T A q_j

    # c_i = (I - s_i s_i^T) w_i, so that Pi_i w_i = w_i - Q c_i.
    overlap = numpy.einsum("ji,ji->i", inverse_t, weights)
    residual_coeffs = weights - inverse_t * overlap

    low_rank = numpy.trace(small) - numpy.einsum(
        "ji,ji->i", inverse_t, small @ inverse_t
    )
    quadratic = numpy.einsum("ij,ij->j", test, sketch)  # w_i^T A w_i
    correction = (
        quadratic
        - numpy.einsum("ji,ji->i", cross, residual_coeffs)
        - numpy.einsum("ji,ji->i", upper, residual_coeffs)
        + numpy.einsum("ji,ji->i", residual_coeffs, small @ residual_coeffs)
    )
    samples = low_rank + correction
    return TraceEstimate(
        value=float(samples.mean()),
        stderr=float(samples.std(ddof=1) / numpy.sqrt(num_vectors)),
        num_matvecs=2 * num_vectors,
    )


def _hutchpp(
    operator: LinearOperator, num_vectors: int, rng: numpy.random.Generator
) -> TraceEstimate:
    """
    Hutch++ (Meyer et al. 2021), Algorithm 1.

    With a budget of ``k`` products and ``c = k // 3``:

    1. draw sketching matrices ``S`` and ``G``, each ``n x c``;
    2. ``Q`` = orthonormal basis of ``A S`` (``c`` products);
    3. return ``trace(Q^T A Q)`` (``c`` products) plus the Hutchinson estimate
       of ``trace((I - Q Q^T) A (I - Q Q^T))`` using ``G`` (``c`` products).

    The first term is exact on the subspace the sketch found, so all of the
    remaining variance comes from the complement, where the spectrum has
    already been deflated.  Unbiased because ``G`` is independent of ``S`` and
    hence of ``Q``.
    """
    size = operator.shape[0]
    count = num_vectors // 3
    sketch_vectors = rademacher(size, count, rng)
    probe_vectors = rademacher(size, count, rng)

    sketch = numpy.asarray(operator.matmat(sketch_vectors))
    basis, _ = numpy.linalg.qr(sketch)
    projected = numpy.asarray(operator.matmat(basis))
    low_rank = float(numpy.trace(basis.T @ projected))

    # (I - Q Q^T) G, then one product each: g^T (I-P) A (I-P) g = w^T A w.
    deflated = probe_vectors - basis @ (basis.T @ probe_vectors)
    images = numpy.asarray(operator.matmat(deflated))
    samples = numpy.einsum("ij,ij->j", deflated, images)

    return TraceEstimate(
        value=low_rank + float(samples.mean()),
        stderr=(
            float(samples.std(ddof=1) / numpy.sqrt(count))
            if count > 1
            else float("nan")
        ),
        num_matvecs=3 * count,
    )


def _exact(operator: LinearOperator) -> TraceEstimate:
    size = operator.shape[0]
    dense = numpy.asarray(operator.matmat(numpy.eye(size)))
    return TraceEstimate(value=float(numpy.trace(dense)), stderr=0.0, num_matvecs=size)


def estimate_trace(
    operator: LinearOperator,
    num_vectors: int = 50,
    method: str = "xtrace",
    rng: Optional[numpy.random.Generator] = None,
) -> TraceEstimate:
    """
    Estimate ``trace(operator)``.

    :param operator: Square :class:`LinearOperator`.  It need not be
        symmetric: AI-REML needs ``trace(P V_i)``, which is not.
    :param num_vectors: Matrix-vector budget for ``"hutchinson"`` and
        ``"hutchpp"``; number of test vectors (costing two products each) for
        ``"xtrace"``.  See the module docstring.
    :param method: ``"xtrace"``, ``"hutchpp"``, ``"hutchinson"``, or
        ``"exact"``.
    :param rng: Random generator; a fresh default one is used if omitted.
    """
    operator = aslinearoperator(operator)
    rows, cols = operator.shape
    if rows != cols:
        raise ValueError(f"Expected a square operator, got shape {operator.shape}")
    if method == "exact":
        return _exact(operator)
    if rng is None:
        rng = numpy.random.default_rng()
    if num_vectors < 1:
        raise ValueError("num_vectors must be positive")
    num_vectors = min(num_vectors, rows)
    if method == "hutchinson":
        return _hutchinson(operator, num_vectors, rng)
    if method == "hutchpp":
        if num_vectors < 3:
            raise ValueError("hutchpp needs num_vectors >= 3")
        return _hutchpp(operator, num_vectors, rng)
    if method == "xtrace":
        if num_vectors < 2:
            return _hutchinson(operator, num_vectors, rng)
        return _xtrace(operator, num_vectors, rng)
    raise ValueError(f"Unknown trace estimator: {method!r}")
