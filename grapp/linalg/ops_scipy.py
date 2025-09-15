"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
from typing import Tuple, Union, List
from pygrgl import TraversalDirection

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias
import numpy
import pygrgl


_DOWN = TraversalDirection.DOWN
_UP = TraversalDirection.UP


def _flip_dir(direction: TraversalDirection) -> TraversalDirection:
    return _UP if direction == _DOWN else _DOWN


def _transpose_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    return (shape[1], shape[0])


class GRGOperatorFilter:
    """
    Used to specify a subset of data (samples, mutations) in a GRG that an operation should
    apply to. Can specify just samples, just mutations, or both, to take a subset of the
    genotype matrix that the GRG represents.

    :param sample_filter: List of sample IDs (NodeIDs for sample ndoes) to keep, or empty list
        to keep all samples.
    :type sample_filter: Union[List[int], numpy.typing.NDArray]
    :param mutation_filter: List of MutationIDs to keep, or empty list to keep all mutations.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    """

    def __init__(
        self,
        sample_filter: Union[List[int], numpy.typing.NDArray] = [],
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
    ):
        self.sample_filter = sample_filter
        self.mutation_filter = mutation_filter

    def __bool__(self):
        """
        True if the filter is specified, False if the filter is a no-op.
        """
        return (len(self.sample_filter) + len(self.mutation_filter)) > 0

    def num_samples(self, grg_samples: int) -> int:
        return len(self.sample_filter) if self.sample_filter else grg_samples

    def num_mutations(self, grg_mutations: int) -> int:
        return len(self.mutation_filter) if self.mutation_filter else grg_mutations

    def shape(self, grg_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        The shape of the matrix that the filtered GRG represents. If the filter keeps L samples and
        P mutations, then the shape is LxP.

        :param grg_shape: The shape of the underlying GRG (see grg.shape).
        :type grg_shape: Tuple[int, int]
        :return: The tuple (L, P).
        :rtype: Tuple[int, int]
        """
        return (
            len(self.sample_filter) if self.sample_filter else grg_shape[0],
            len(self.mutation_filter) if self.mutation_filter else grg_shape[1],
        )

    def expand_mat(
        self,
        other_matrix: numpy.typing.NDArray,
        grg_shape: Tuple[int, int],
        direction: TraversalDirection,
    ):
        """
        Given a matrix that is "multiplication-compatible" with shape self.shape (:math:`L \\times P`), return
        a matrix compatible with shape grg_shape (:math:`N \\times M`). When direction is UP, the input matrix
        will be :math:`K \\times L` and output matrix will be :math:`K \\times N`, where K is an arbitrary
        number of rows. When direction is down, the input matrix will be :math:`K \\times P` and output will
        be :math:`K \\times M`.

        :param other_matrix: The matrix to expand.
        :type other_matrix: numpy.ndarray
        :param grg_shape: The shape of the GRG, a tuple (N, M) where N is number of samples
            (or individuals) and M is the number of mutations.
        :type grg_shape: Tuple[int, int]
        :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
            :math:`X^T` (pygrgl.TraversalDirection.DOWN).
        :type direction: pygrgl.TraversalDirection
        :return: The expanded NxM matrix.
        :rtype: numpy.ndarray
        """
        K = other_matrix.shape[0]
        if direction == _UP:
            if self.sample_filter:
                result = numpy.zeros((K, grg_shape[0]), dtype=other_matrix.dtype)
                result[:, self.sample_filter] = other_matrix  # type: ignore
            else:
                result = other_matrix
        else:
            assert direction == _DOWN
            if self.mutation_filter:
                result = numpy.zeros((K, grg_shape[1]), dtype=other_matrix.dtype)
                result[:, self.mutation_filter] = other_matrix  # type: ignore
            else:
                result = other_matrix
        return result

    def contract_mat(
        self,
        other_matrix: numpy.typing.NDArray,
        direction: TraversalDirection,
    ):
        """
        Given a matrix that is "multiplication-compatible" with the GRG's shape (:math:`N \\times M`), return
        a matrix compatible with self.shape (:math:`L \\times P`). When direction is UP, the input matrix
        will be :math:`K \\times N` and output matrix will be :math:`K \\times L`, where K is an arbitrary
        number of rows. When direction is down, the input matrix will be :math:`K \\times M` and output will
        be :math:`K \\times P`.

        :param other_matrix: The matrix to contract.
        :type other_matrix: numpy.ndarray
        :param direction: Determines whether the matrix is :math:`X` (pygrgl.TraversalDirection.UP) or
            :math:`X^T` (pygrgl.TraversalDirection.DOWN).
        :type direction: pygrgl.TraversalDirection
        :return: The expanded :math:`K \\times N` or :math:`K \\times M` matrix.
        :rtype: numpy.ndarray
        """
        if direction == _DOWN:
            if self.sample_filter:
                result = other_matrix[:, self.sample_filter]  # type: ignore
            else:
                result = other_matrix
        else:
            assert direction == _UP
            if self.mutation_filter:
                result = other_matrix[:, self.mutation_filter]  # type: ignore
            else:
                result = other_matrix
        return result


class SciPyXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the genotype matrix represented by the GRG, which allows for
    multiplication between the GRG and a matrix or vector. This is for the non-standardized matrix, which
    just contains discrete allele counts.

    Can perform the operation :math:`X \\times A` (_matmat) or :math:`X \\times \overrightarrow{v}` (_matvec).

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
    :param filter: Changes the dimensions of :math:`X` to match the provided filter, which can ignore both mutations
        and samples/individuals when performing the matrix operations. Default: empty filter.
    :type filter: GRGOperatorFilter
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: TraversalDirection,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        filter: GRGOperatorFilter = GRGOperatorFilter(),
    ):
        self.haploid = haploid
        self.grg = grg
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        self.direction = direction
        self.filter = filter
        self.grg_shape = (self.sample_count, grg.num_mutations)
        shape = self.filter.shape(self.grg_shape)
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(
        self, other_matrix: numpy.typing.NDArray, mult_dir: TraversalDirection
    ):
        if self.filter:
            A = self.filter.expand_mat(other_matrix.T, self.grg_shape, mult_dir)
        else:
            A = other_matrix.T
        result = pygrgl.matmul(
            self.grg,
            A,
            mult_dir,
            by_individual=not self.haploid,
        )
        if self.filter:
            Y = self.filter.contract_mat(result, mult_dir).T
        else:
            Y = result.T
        return Y

    def _matmat(self, other_matrix):
        return self._matmat_helper(other_matrix, _flip_dir(self.direction))

    def _rmatmat(self, other_matrix):
        return self._matmat_helper(other_matrix, self.direction)

    def _matvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


class SciPyXTXOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`X^TX` represented by the GRG.
    This is for the non-standardized matrix, which just contains discrete allele counts, but it is not
    centered at the mean, so it is not quite the covariance matrix.

    Can perform the operation :math:`X^T \\times X \\times A` (_matmat) or
    :math:`X \\times X \\times \overrightarrow{v}` (_matvec).

    :param grg: The GRG the operator will multiply against.
    :type grg: pygrgl.GRG
    :param dtype: The numpy.dtype to use.
    :type dtype: TypeAlias
    :param haploid: Perform calculations on the {0, 1} haploid genotype matrix, instead of the {0, ..., grg.ploidy}
        genotype matrix. Default: False.
    :type haploid: bool
    :param filter: Changes the dimensions of :math:`X` to match the provided filter, which can ignore both mutations
        and samples/individuals when performing the matrix operations. Default: empty filter.
    :type filter: GRGOperatorFilter
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        filter: GRGOperatorFilter = GRGOperatorFilter(),
    ):
        num_muts = filter.num_mutations(grg.num_mutations)
        xtx_shape = (num_muts, num_muts)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.x_op = SciPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            filter=filter,
        )

    def _matmat(self, other_matrix):
        D = self.x_op._matmat(other_matrix)
        return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        # Assume direction == UP, then we are operating on X. Given this, we have X: NxM and
        # the input vector must be of length M.
        vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        # Assume direction == UP, then we are operating on X^T for rmatvec. Given this, we
        # have X^T: MxN and the input vector must be of length N.
        vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


# (Abstract) base class for GRG-based scipy LinearOperators that standardize the underlying
# genotype matrix.
class _SciPyStandardizedOperator(LinearOperator):
    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        shape: Tuple[int, int],
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
    ):
        self.haploid = haploid
        self.grg = grg
        self.freqs = freqs
        self.mult_const = 1 if self.haploid else grg.ploidy
        self.sample_count = grg.num_samples if haploid else grg.num_individuals

        # TODO: there might be other normalization approachs besides this. For example, FlashPCA2 has different
        # options for what to use (this is the P-trial binomial).
        raw = self.mult_const * self.freqs * (1.0 - self.freqs)

        # Two versions of sigma, the second flips 0 values (which means the frequency was
        # either 1 or 0 for the mutation) to 1 values so we can use it for division.
        original_sigma = numpy.sqrt(raw)
        self.sigma_corrected = numpy.where(
            original_sigma == 0,
            1,
            original_sigma,
        )
        super().__init__(dtype=dtype, shape=shape)


# Operator on the standardized GRG X or X^T (based on the direction chosen)
class SciPyStdXOperator(_SciPyStandardizedOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the genotype matrix represented by the GRG, which allows for
    multiplication between the GRG and a matrix or vector. This is for the standardized matrix, which is
    centered to the mean (based on allele frequencies) and standard devation (based on the binomial distribution
    where each individual is the result of :math:`p`, the ploidy, trials).

    Can perform the operation :math:`X \\times A` (_matmat) or :math:`X \\times \overrightarrow{v}` (_matvec).

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
    :param filter: Changes the dimensions of :math:`X` to match the provided filter, which can ignore both mutations
        and samples/individuals when performing the matrix operations. Default: empty filter.
    :type filter: GRGOperatorFilter
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: pygrgl.TraversalDirection,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        filter: GRGOperatorFilter = GRGOperatorFilter(),
    ):
        self.direction = direction
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        self.filter = filter
        self.grg_shape = (self.sample_count, grg.num_mutations)
        self.grg_shape = (self.sample_count, grg.num_mutations)
        shape = self.filter.shape(self.grg_shape)
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(grg, freqs, shape, dtype=dtype, haploid=haploid)

    def _matmat_direction(self, other_matrix, direction):
        mult_dir = _flip_dir(direction)

        def expandm(matrix):
            if self.filter:
                return self.filter.expand_mat(matrix, self.grg_shape, mult_dir)
            return matrix

        def contractm(matrix):
            if self.filter:
                return self.filter.contract_mat(matrix, mult_dir)
            return matrix

        with numpy.errstate(divide="raise"):
            if direction == _UP:
                vS = expandm(other_matrix.T) / self.sigma_corrected
                XvS = pygrgl.matmul(
                    self.grg,
                    vS,
                    mult_dir,
                    by_individual=not self.haploid,
                )
                consts = numpy.array(
                    [numpy.sum(self.mult_const * self.freqs * vS, axis=1)]
                ).T
                return contractm(XvS - consts).T
            else:
                assert direction == _DOWN
                SXv = (
                    pygrgl.matmul(
                        self.grg,
                        expandm(other_matrix.T),
                        mult_dir,
                        by_individual=not self.haploid,
                    )
                    / self.sigma_corrected
                )
                col_const = numpy.sum(other_matrix, axis=0, keepdims=True).T
                sub_const2 = (
                    self.mult_const * self.freqs / self.sigma_corrected
                ) * col_const
                return contractm(SXv - sub_const2).T

    def _matmat(self, other_matrix):
        return self._matmat_direction(other_matrix, self.direction)

    def _rmatmat(self, other_matrix):
        return self._matmat_direction(other_matrix, _flip_dir(self.direction))

    def _matvec(self, vect):
        # Assume direction == UP, then we are operating on X. Given this, we have X: NxM and
        # the input vector must be of length M.
        vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        # Assume direction == UP, then we are operating on X^T for rmatvec. Given this, we
        # have X^T: MxN and the input vector must be of length N.
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
    :math:`X \\times X \\times \overrightarrow{v}` (_matvec).

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
    :param filter: Changes the dimensions of :math:`X` to match the provided filter, which can ignore both mutations
        and samples/individuals when performing the matrix operations. Default: empty filter.
    :type filter: GRGOperatorFilter
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        filter: GRGOperatorFilter = GRGOperatorFilter(),
    ):
        num_muts = filter.num_mutations(grg.num_mutations)
        xtx_shape = (num_muts, num_muts)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.std_x_op = SciPyStdXOperator(
            grg, _UP, freqs, haploid=haploid, dtype=dtype, filter=filter
        )

    def _matmat(self, other_matrix):
        D = self.std_x_op._matmat(other_matrix)
        return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)
