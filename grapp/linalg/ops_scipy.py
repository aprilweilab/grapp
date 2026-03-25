"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
from pygrgl import TraversalDirection
from typing import Tuple, Union, List, Optional
from grapp.grg_calculator import (
    GRGCalcInterface as _GRGCalcInterface,
    _wrap_grg,
)
import concurrent.futures
import numpy
import pygrgl

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias  # type: ignore


_DOWN = TraversalDirection.DOWN
_UP = TraversalDirection.UP


def _flip_dir(direction: TraversalDirection) -> TraversalDirection:
    return _UP if direction == _DOWN else _DOWN


def _transpose_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    return (shape[1], shape[0])


class GRGOpFilter:
    def __init__(
        self,
        grg: _GRGCalcInterface,
        haploid: bool,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]],
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]],
    ):
        if mutation_filter is not None:
            if isinstance(mutation_filter, numpy.ndarray):
                mutation_filter = mutation_filter.tolist()
            assert len(set(mutation_filter)) == len(
                mutation_filter
            ), "Duplicate IDs in mutation_filter"
        self.mutation_filter = mutation_filter
        if sample_filter is not None:
            if isinstance(sample_filter, numpy.ndarray):
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

    def prep_input(
        self, input_matrix: numpy.typing.NDArray, mult_dir: TraversalDirection
    ):
        if mult_dir == _DOWN:
            if self.mutation_filter is not None:
                new_matrix = numpy.zeros(
                    (input_matrix.shape[0], self.grg_shape[1]), dtype=input_matrix.dtype
                )
                new_matrix[:, self.mutation_filter] = input_matrix  # type: ignore
                return new_matrix
        else:
            assert mult_dir == _UP
            if self.sample_filter is not None:
                new_matrix = numpy.zeros(
                    (input_matrix.shape[0], self.grg_shape[0]), dtype=input_matrix.dtype
                )
                new_matrix[:, self.sample_filter] = input_matrix  # type: ignore
                return new_matrix
        return input_matrix

    def adjust_output(
        self, output_matrix: numpy.typing.NDArray, mult_dir: TraversalDirection
    ):
        if mult_dir == _UP:
            if self.mutation_filter is not None:
                return output_matrix[:, self.mutation_filter]
        else:
            assert mult_dir == _DOWN
            if self.sample_filter is not None:
                return output_matrix[:, self.sample_filter]
        return output_matrix


# General type for GRG objects.
GRGType = Union[pygrgl.GRG, _GRGCalcInterface]


class SciPyXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the genotype matrix represented by the GRG, which allows for
    multiplication between the GRG and a matrix or vector. This is for the non-standardized matrix, which
    just contains discrete allele counts.

    Can perform the operation :math:`X \\times A` (_matmat) or :math:`X \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
        :math:`X^T` (pygrgl.TraversalDirection.DOWN).
    :type direction: pygrgl.TraversalDirection
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param miss_values: If non-None, must be a vector of length num_mutations, which provides a per-
        mutation value for missingness (applied per haplotype). Usually the per-Mutation mean value
        (e.g., missingness-adjusted allele frequency) is provided. Default: None.
    :type miss_values: Optional[numpy.typing.NDArray]
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    """

    def __init__(
        self,
        grg: GRGType,
        direction: TraversalDirection,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.typing.NDArray] = None,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
    ):
        self.filter = GRGOpFilter(grg, haploid, mutation_filter, sample_filter)
        self.haploid = haploid
        self.grg = _wrap_grg(grg)
        self.direction = direction
        assert (
            miss_values is None or miss_values.ndim == 1
        ), 'If "miss_values" is provided, it must be a vector'
        self.miss_values = miss_values
        shape = self.filter.shape
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(
        self, other_matrix: numpy.typing.NDArray, mult_dir: TraversalDirection
    ):
        kwargs = {}
        A = self.filter.prep_input(other_matrix.T, mult_dir)

        # When doing the multiplication AX^T (DOWN), we need to initialize the missingness node
        # data values ("miss") with the input matrix * the missingness mean values per mutation.
        use_M = self.grg.has_missing_data and self.miss_values is not None
        if use_M:
            if mult_dir == _DOWN:
                M = numpy.array([self.miss_values]) * A
            else:
                M = numpy.zeros((A.shape[0], self.grg.num_mutations))
            kwargs["miss"] = M

        result = self.grg.matmul(
            A,
            mult_dir,
            by_individual=not self.haploid,
            **kwargs,
        )

        # When doing the multiplication AX (UP), we need to adjust the final output with the
        # missingness node data values ("miss") * the missingness mean values per mutation.
        if mult_dir == _UP and use_M:
            result += M * self.miss_values

        return self.filter.adjust_output(result, mult_dir).T

    def _matmat(self, other_matrix):
        return self._matmat_helper(other_matrix, _flip_dir(self.direction))

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(other_matrix, self.direction)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


class SciPyXTXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`X^TX` represented by the GRG.
    This is for the non-standardized matrix, which just contains discrete allele counts, but it is not
    centered at the mean, so it is not quite the covariance matrix.

    Can perform the operation :math:`X^T \\times X \\times A` (_matmat) or
    :math:`X^T \\times X \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param miss_values: If non-None, must be a vector of length num_mutations, which provides a per-
        mutation value for missingness (applied per haplotype). Usually the per-Mutation mean value
        (e.g., missingness-adjusted allele frequency) is provided. Default: None.
    :type miss_values: Optional[numpy.typing.NDArray]
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    """

    def __init__(
        self,
        grg: GRGType,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.typing.NDArray] = None,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
    ):
        self.x_op = SciPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
        )
        xtx_shape = (self.x_op.shape[1], self.x_op.shape[1])
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.x_op._matmat(other_matrix)
        return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


class SciPyXXTOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`XX^T` represented by the GRG.
    This is for the non-standardized matrix, which just contains discrete allele counts.

    Can perform the operation :math:`X \\times X^T \\times A` (_matmat) or
    :math:`X \\times X^T \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param miss_values: If non-None, must be a vector of length num_mutations, which provides a per-
        mutation value for missingness (applied per haplotype). Usually the per-Mutation mean value
        (e.g., missingness-adjusted allele frequency) is provided. Default: None.
    :type miss_values: Optional[numpy.typing.NDArray]
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    """

    def __init__(
        self,
        grg: GRGType,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.typing.NDArray] = None,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
    ):
        self.x_op = SciPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
        )
        xxt_shape = (self.x_op.shape[0], self.x_op.shape[0])
        super().__init__(dtype=dtype, shape=xxt_shape)

    def _matmat(self, other_matrix):
        D = self.x_op._rmatmat(other_matrix)
        return self.x_op._matmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


# (Abstract) base class for GRG-based scipy LinearOperators that standardize the underlying
# genotype matrix.
class _SciPyStandardizedOperator(LinearOperator):
    """
    Private base class for standardizing the genotype matrix.

    :param grg: The GRG
    :param freqs: The allele frequencies, used to determine the mean and variance.
    :param shape: The shape of the operator.
    :param dtype: The datatype to use for the result.
    :param haploid: When true, treat the matrix as haplotypes, not individuals of ploid P.
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    """

    def __init__(
        self,
        grg: GRGType,
        freqs: numpy.typing.NDArray,
        shape: Tuple[int, int],
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.haploid = haploid
        self.grg = _wrap_grg(grg)
        self.freqs = freqs
        self.mult_const = 1 if self.haploid else grg.ploidy

        if custom_variance is not None:
            assert (
                custom_variance.shape == self.freqs.shape
            ), "custom_variance must have same dimension as freqs (1xM)"
            variance = custom_variance
        else:
            variance = self.mult_const * self.freqs * (1.0 - self.freqs)

        # Multiplicative scaling factor is sqrt(variance^alpha), we maintain 0 values
        # (which means the frequency was either 1 or 0 for the mutation) to avoid NaNs
        with numpy.errstate(invalid="raise"):
            self.inverse_sigma = numpy.zeros(variance.shape)
            numpy.sqrt(
                numpy.power(
                    variance, alpha, out=self.inverse_sigma, where=variance != 0
                ),
                out=self.inverse_sigma,
            )
        super().__init__(dtype=dtype, shape=shape)


# Operator on the standardized GRG X or X^T (based on the direction chosen)
class SciPyStdXOperator(_SciPyStandardizedOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the genotype matrix represented by the GRG, which allows for
    multiplication between the GRG and a matrix or vector. This is for the standardized matrix, which is
    centered to the mean (based on allele frequencies) and standard devation (based on the binomial distribution
    where each individual is the result of :math:`p`, the ploidy, trials).

    Can perform the operation :math:`X \\times A` (_matmat) or :math:`X \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
        :math:`X^T` (pygrgl.TraversalDirection.DOWN).
    :type direction: pygrgl.TraversalDirection
    :param freqs: A vector of length num_mutations, containing the allele frequency for all mutations.
        Indexed by the mutation ID of the mutation.
    :type freqs: numpy.ndarray
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grg: GRGType,
        direction: pygrgl.TraversalDirection,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.filter = GRGOpFilter(grg, haploid, mutation_filter, sample_filter)
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

    def _matmat_direction(self, other_matrix, direction):
        mult_dir = _flip_dir(direction)
        with numpy.errstate(divide="raise"):
            if direction == _UP:
                vS = (
                    self.filter.prep_input(other_matrix.T, mult_dir)
                    * self.inverse_sigma
                )
                XvS = self.grg.matmul(
                    vS,
                    mult_dir,
                    by_individual=not self.haploid,
                )
                consts = numpy.array(
                    [numpy.sum(self.mult_const * self.freqs * vS, axis=1)]
                ).T
                return self.filter.adjust_output(XvS - consts, mult_dir).T
            else:
                assert direction == _DOWN
                m = self.filter.prep_input(other_matrix.T, mult_dir)
                SXv = (
                    self.grg.matmul(
                        m,
                        mult_dir,
                        by_individual=not self.haploid,
                    )
                    * self.inverse_sigma
                )
                col_const = numpy.sum(m.T, axis=0, keepdims=True).T
                sub_const2 = (
                    self.mult_const * self.freqs * self.inverse_sigma
                ) * col_const
                return self.filter.adjust_output(SXv - sub_const2, mult_dir).T

    def _matmat(self, other_matrix):
        return self._matmat_direction(other_matrix, self.direction)

    def _rmatmat(self, other_matrix):
        return self._matmat_direction(other_matrix, _flip_dir(self.direction))

    def _matvec(self, vect):
        # Assume direction == UP, then we are operating on X. Given this, we have X: NxM and
        # the input vector must be of length M.
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        # Assume direction == UP, then we are operating on X^T for rmatvec. Given this, we
        # have X^T: MxN and the input vector must be of length N.
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


# Correlation matrix X^T*X operator on the standardized GRG
class SciPyStdXTXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`X^TX` represented by the GRG.
    This is for the standardized matrix, which is centered to the mean (based on allele
    frequencies) and standard devation (based on the binomial distribution where each individual
    is the result of :math:`p`, the ploidy, trials).

    This operator performs multiplications against the correlation matrix of the genotype matrix
    underlying the GRG. Can perform the operation :math:`X^T \\times X \\times A` (_matmat) or
    :math:`X \\times X \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param freqs: A vector of length num_mutations, containing the allele frequency for all mutations.
        Indexed by the mutation ID of the mutation.
    :type freqs: numpy.ndarray
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grg: GRGType,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.filter = GRGOpFilter(grg, haploid, mutation_filter, sample_filter)
        self.std_x_op = SciPyStdXOperator(
            grg,
            _UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        xtx_shape = (self.std_x_op.shape[1], self.std_x_op.shape[1])
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.std_x_op._matmat(other_matrix)
        return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


# Genetic relatedness matrix X*X^T operator on the standardized GRG
class SciPyStdXXTOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`XX^T` represented by the GRG.
    This is for the standardized matrix, which is centered to the mean (based on allele
    frequencies) and standard devation (based on the binomial distribution where each individual
    is the result of :math:`p`, the ploidy, trials).

    This operator performs multiplications against the correlation matrix of the genotype matrix
    underlying the GRG. Can perform the operation :math:`X \\times X^T \\times A` (_matmat) or
    :math:`X \\times X^T \\times \\overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param freqs: A vector of length num_mutations, containing the allele frequency for all mutations.
        Indexed by the mutation ID of the mutation.
    :type freqs: numpy.ndarray
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grg: GRGType,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.std_x_op = SciPyStdXOperator(
            grg,
            _UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        xxt_shape = (self.std_x_op.shape[0], self.std_x_op.shape[0])
        super().__init__(dtype=dtype, shape=xxt_shape)

    def _matmat(self, other_matrix):
        D = self.std_x_op._rmatmat(other_matrix)
        return self.std_x_op._matmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


class MultiSciPyXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on multiple GRGs. Same as SciPyXOperator, except if the input
    GRGs have mutation counts M1, M2, ..., MK, then the dimension of the implicit underlying genotype
    matrix is Nx(M1 + M2 + ... + MK).

    :param grgs: The GRGs the operator will multiply against. They must all have the same samples,
        and the mutations are expected to differ (e.g., one GRG per chromosome of the same dataset).
    :type grgs: List[pygrgl.GRG]
    :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
        :math:`X^T` (pygrgl.TraversalDirection.DOWN).
    :type direction: pygrgl.TraversalDirection
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param miss_values: If non-None, must be a vector of length num_mutations (for all GRGs), which provides
        a per-mutation value for missingness (applied per haplotype). Usually the per-Mutation mean value
        (e.g., missingness-adjusted allele frequency) is provided. Default: None.
    :type miss_values: Optional[numpy.typing.NDArray]
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Here the mutation filter follows the same numbering as the
        input/output matrices: for example, if grgs=[grg1, grg2] then indexes 0...(grg1.num_mutations-1)
        will be for grg1, and grg1.num_mutations...(grg1.num_mutations+grg2.num_mutations-1) will be the
        mutations for grg2. Then if you have a mutation_filter containing the number ``grg1.num_mutations + 4``
        it means it will keep grg2's mutation with ID ``4``. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Since all GRGs have the same samples, this behavior is the
        same as the non-Multi operators. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[GRGType],
        direction: pygrgl.TraversalDirection,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.typing.NDArray] = None,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        threads: int = 1,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        self.direction = direction
        self.num_mutations = sum([g.num_mutations for g in grgs])
        if mutation_filter is not None:
            assert len(mutation_filter) <= self.num_mutations
            self.num_mutations = len(mutation_filter)
        self.operators = []
        prev_miss_start = 0
        prev_max_mut = 0
        for g in grgs:
            assert (
                g.num_samples == grgs[0].num_samples
            ), "All GRGs must use the same samples"
            if mutation_filter is not None:
                grg_mut_filt = list(
                    map(
                        lambda m: m - prev_max_mut,
                        filter(
                            lambda m: m >= prev_max_mut
                            and m < prev_max_mut + g.num_mutations,
                            mutation_filter,
                        ),
                    )
                )
                # If we have an overall filter, but no filter for _this_ GRG, then we just skip it.
                skip = len(grg_mut_filt) == 0
            else:
                grg_mut_filt = None
                skip = False
            if not skip:
                effective_muts = (
                    len(grg_mut_filt) if grg_mut_filt is not None else g.num_mutations
                )
                if miss_values is not None:
                    grg_miss_values = miss_values[
                        prev_miss_start : prev_miss_start + effective_muts
                    ]
                    prev_miss_start += effective_muts
                else:
                    grg_miss_values = None
                self.operators.append(
                    SciPyXOperator(
                        g,
                        direction,
                        dtype,
                        haploid=haploid,
                        miss_values=grg_miss_values,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                    )
                )
            prev_max_mut += g.num_mutations
        # Should we concatenate the result for _matmat, or add them together?
        self.concat = self.direction == pygrgl.TraversalDirection.DOWN
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        shape = (self.operators[0].shape[0], self.num_mutations)
        if direction == _UP:
            shape = (
                self.operators[0].shape[0],
                sum(map(lambda op: op.shape[1], self.operators)),
            )
        else:
            assert direction == _DOWN
            shape = (
                sum(map(lambda op: op.shape[0], self.operators)),
                self.operators[0].shape[1],
            )
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(self, other_matrix, direction, op_method):
        # For UP, we have "(N x M) x (M x k)", so we need to split the other_matrix into chunks of the
        # appropriate size <= M.
        futures = []
        if direction == pygrgl.TraversalDirection.UP:
            start = 0
            for op in self.operators:
                end = start + op.shape[1]
                assert end <= other_matrix.shape[0]
                sub_matrix = other_matrix[start:end, :]
                futures.append(self.executor.submit(op_method, op, sub_matrix))
                start = end
            result = None
            for future in futures:
                if result is None:
                    result = future.result()
                else:
                    result += future.result()
            return result
        # For DOWN, we have "(M x N) x (N x k)", so it is much simpler (no splitting)
        else:
            for op in self.operators:
                futures.append(self.executor.submit(op_method, op, other_matrix))
            result = [f.result() for f in futures]
            return numpy.concatenate(result)

    def _matmat(self, other_matrix):
        return self._matmat_helper(other_matrix, self.direction, SciPyXOperator._matmat)

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, _flip_dir(self.direction), SciPyXOperator._rmatmat
        )

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._rmatmat(vect)


class MultiSciPyXTXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on multiple GRGs. Same as SciPyXTXOperator, except if the input
    GRGs have mutation counts M1, M2, ..., MK, then the dimension of the implicit underlying genotype
    matrix is Nx(M1 + M2 + ... + MK).

    :param grgs: The GRGs the operator will multiply against. They must all have the same samples,
        and the mutations are expected to differ (e.g., one GRG per chromosome of the same dataset).
    :type grgs: List[pygrgl.GRG]
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param miss_values: If non-None, must be a vector of length num_mutations (for all GRGs), which provides
        a per-mutation value for missingness (applied per haplotype). Usually the per-Mutation mean value
        (e.g., missingness-adjusted allele frequency) is provided. Default: None.
    :type miss_values: Optional[numpy.typing.NDArray]
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Here the mutation filter follows the same numbering as the
        input/output matrices: for example, if grgs=[grg1, grg2] then indexes 0...(grg1.num_mutations-1)
        will be for grg1, and grg1.num_mutations...(grg1.num_mutations+grg2.num_mutations-1) will be the
        mutations for grg2. Then if you have a mutation_filter containing the number ``grg1.num_mutations + 4``
        it means it will keep grg2's mutation with ID ``4``. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Since all GRGs have the same samples, this behavior is the
        same as the non-Multi operators. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[GRGType],
        dtype=numpy.float64,
        haploid: bool = False,
        miss_values: Optional[numpy.typing.NDArray] = None,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        threads: int = 1,
    ):
        self.x_op = MultiSciPyXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            dtype=dtype,
            haploid=haploid,
            miss_values=miss_values,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            threads=threads,
        )
        xtx_shape = (self.x_op.shape[1], self.x_op.shape[1])
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.x_op._matmat(other_matrix)
        return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matvec(vect)


class MultiSciPyStdXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on multiple GRGs. Same as SciPyStdXOperator, except if the input
    GRGs have mutation counts M1, M2, ..., MK, then the dimension of the implicit underlying genotype
    matrix is Nx(M1 + M2 + ... + MK).

    :param grgs: The GRGs the operator will multiply against. They must all have the same samples,
        and the mutations are expected to differ (e.g., one GRG per chromosome of the same dataset).
    :type grgs: List[pygrgl.GRG]
    :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
        :math:`X^T` (pygrgl.TraversalDirection.DOWN).
    :type direction: pygrgl.TraversalDirection
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Here the mutation filter follows the same numbering as the
        input/output matrices: for example, if grgs=[grg1, grg2] then indexes 0...(grg1.num_mutations-1)
        will be for grg1, and grg1.num_mutations...(grg1.num_mutations+grg2.num_mutations-1) will be the
        mutations for grg2. Then if you have a mutation_filter containing the number ``grg1.num_mutations + 4``
        it means it will keep grg2's mutation with ID ``4``. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Since all GRGs have the same samples, this behavior is the
        same as the non-Multi operators. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grgs: List[GRGType],
        direction: pygrgl.TraversalDirection,
        freqs: List[numpy.typing.NDArray],
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        assert len(grgs) >= 1, "Must provide at least one GRG"
        assert len(grgs) == len(freqs), "Must provide allele frequencies for every GRG"
        self.direction = direction
        self.num_mutations = sum([g.num_mutations for g in grgs])
        num_samples = grgs[0].num_samples
        num_indivs = grgs[0].num_individuals
        if mutation_filter is not None:
            assert len(mutation_filter) <= self.num_mutations  # type: ignore
            self.num_mutations = len(mutation_filter)  # type: ignore
        prev_max_mut = 0
        self.operators = []
        for g, f in zip(grgs, freqs):
            assert g.num_samples == num_samples, "All GRGs must use the same samples"
            if mutation_filter is not None:
                grg_mut_filt = list(
                    map(
                        lambda m: m - prev_max_mut,
                        filter(
                            lambda m: m >= prev_max_mut
                            and m < prev_max_mut + g.num_mutations,
                            mutation_filter,
                        ),
                    )
                )
                # If we have an overall filter, but no filter for _this_ GRG, then we just skip it.
                skip = len(grg_mut_filt) == 0
            else:
                grg_mut_filt = None
                skip = False
            if not skip:
                self.operators.append(
                    SciPyStdXOperator(
                        g,
                        direction,
                        f,
                        dtype,
                        haploid=haploid,
                        mutation_filter=grg_mut_filt,
                        sample_filter=sample_filter,
                        alpha=alpha,
                        custom_variance=custom_variance,
                    )
                )
            prev_max_mut += g.num_mutations
        # Should we concatenate the result for _matmat, or add them together?
        self.concat = self.direction == pygrgl.TraversalDirection.DOWN
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        sample_count = num_samples if haploid else num_indivs
        shape = (sample_count, self.num_mutations)
        if direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(self, other_matrix, direction, op_method):
        # For UP, we have "(N x M) x (M x k)", so we need to split the other_matrix into chunks of the
        # appropriate size <= M.
        futures = []
        if direction == pygrgl.TraversalDirection.UP:
            start = 0
            for op in self.operators:
                end = start + op.shape[1]
                assert end <= other_matrix.shape[0]
                sub_matrix = other_matrix[start:end, :]
                futures.append(self.executor.submit(op_method, op, sub_matrix))
                start = end
            result = None
            for future in futures:
                if result is None:
                    result = future.result()
                else:
                    result += future.result()
            return result
        # For DOWN, we have "(M x N) x (N x k)", so it is much simpler (no splitting)
        else:
            for op in self.operators:
                futures.append(self.executor.submit(op_method, op, other_matrix))
            result = [f.result() for f in futures]
            return numpy.concatenate(result)

    def _matmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, self.direction, SciPyStdXOperator._matmat
        )

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(
            other_matrix, _flip_dir(self.direction), SciPyStdXOperator._rmatmat
        )

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._rmatmat(vect)


class MultiSciPyStdXTXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on multiple GRGs. Same as SciPyStdXTXOperator, except if the input
    GRGs have mutation counts M1, M2, ..., MK, then the dimension of the implicit underlying genotype
    matrix is Nx(M1 + M2 + ... + MK).

    :param grgs: The GRGs the operator will multiply against. They must all have the same samples,
        and the mutations are expected to differ (e.g., one GRG per chromosome of the same dataset).
    :type grgs: List[pygrgl.GRG]
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Here the mutation filter follows the same numbering as the
        input/output matrices: for example, if grgs=[grg1, grg2] then indexes 0...(grg1.num_mutations-1)
        will be for grg1, and grg1.num_mutations...(grg1.num_mutations+grg2.num_mutations-1) will be the
        mutations for grg2. Then if you have a mutation_filter containing the number ``grg1.num_mutations + 4``
        it means it will keep grg2's mutation with ID ``4``. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Since all GRGs have the same samples, this behavior is the
        same as the non-Multi operators. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grgs: List[GRGType],
        freqs: List[numpy.typing.NDArray],
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.std_x_op = MultiSciPyStdXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            threads=threads,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        xtx_shape = (self.std_x_op.shape[1], self.std_x_op.shape[1])
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.std_x_op._matmat(other_matrix)
        return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matvec(vect)


class MultiSciPyStdXXTOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on multiple GRGs. Same as SciPyStdXXTOperator, except if the input
    GRGs have mutation counts M1, M2, ..., MK, then the dimension of the implicit underlying genotype
    matrix is Nx(M1 + M2 + ... + MK).

    :param grgs: The GRGs the operator will multiply against. They must all have the same samples,
        and the mutations are expected to differ (e.g., one GRG per chromosome of the same dataset).
    :type grgs: List[pygrgl.GRG]
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Here the mutation filter follows the same numbering as the
        input/output matrices: for example, if grgs=[grg1, grg2] then indexes 0...(grg1.num_mutations-1)
        will be for grg1, and grg1.num_mutations...(grg1.num_mutations+grg2.num_mutations-1) will be the
        mutations for grg2. Then if you have a mutation_filter containing the number ``grg1.num_mutations + 4``
        it means it will keep grg2's mutation with ID ``4``. Default: no filter.
    :type mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param sample_filter: Changes the dimensions of :math:`X` to be QxM (where Q is the length of
        sample_filter) instead of NxM. Since all GRGs have the same samples, this behavior is the
        same as the non-Multi operators. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    :param alpha: Alpha model coefficient (e.g., Speed, et. al., 2012) for variance, which multiplicatively
        scales the genotype matrix by sqrt(variance^alpha). By default alpha=-1, which
        corresponds to the "standard" binomial variance scaling.
    :type alpha: float
    :param custom_variance: Instead of using binomial variance, use provided custom variance
        for mutations. Must be an array of length num_mutations, for example the result from
        grapp.util.variance(). Default: None.
    :type custom_variance: numpy.ndarray
    """

    def __init__(
        self,
        grgs: List[GRGType],
        freqs: List[numpy.typing.NDArray],
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
        threads: int = 1,
        alpha: float = -1,
        custom_variance: Optional[numpy.typing.NDArray] = None,
    ):
        self.std_x_op = MultiSciPyStdXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
            sample_filter=sample_filter,
            threads=threads,
            alpha=alpha,
            custom_variance=custom_variance,
        )
        xxt_shape = (self.std_x_op.shape[0], self.std_x_op.shape[0])
        super().__init__(dtype=dtype, shape=xxt_shape)

    def _matmat(self, other_matrix):
        D = self.std_x_op._rmatmat(other_matrix)
        return self.std_x_op._matmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        if vect.ndim != 2:
            vect = numpy.array([vect]).T
        return self._matvec(vect)
