"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
from pygrgl import TraversalDirection
from typing import Tuple, Union, List
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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: TraversalDirection,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
    ):
        self.haploid = haploid
        self.grg = grg
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        self.direction = direction
        self.mutation_filter = mutation_filter
        self.grg_shape = (self.sample_count, grg.num_mutations)
        shape = (
            self.grg_shape
            if not mutation_filter
            else (self.grg_shape[0], len(self.mutation_filter))
        )
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat_helper(
        self, other_matrix: numpy.typing.NDArray, mult_dir: TraversalDirection
    ):
        if self.mutation_filter and mult_dir == _DOWN:
            A = numpy.zeros(
                (other_matrix.shape[1], self.grg_shape[1]), dtype=other_matrix.dtype
            )
            A[:, self.mutation_filter] = other_matrix.T  # type: ignore
        else:
            A = other_matrix.T
        result = pygrgl.matmul(
            self.grg,
            A,
            mult_dir,
            by_individual=not self.haploid,
        )
        Y = result
        if self.mutation_filter and mult_dir == _UP:
            Y = Y[:, self.mutation_filter]  # type: ignore
        return Y.T

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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
    ):
        num_muts = (
            grg.num_mutations if not (mutation_filter is None) else len(mutation_filter)
        )
        xtx_shape = (num_muts, num_muts)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.x_op = SciPyXOperator(
            grg,
            _UP,
            dtype=dtype,
            haploid=haploid,
            mutation_filter=mutation_filter,
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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: pygrgl.TraversalDirection,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
    ):
        if isinstance(mutation_filter, numpy.ndarray):
            mutation_filter = mutation_filter.tolist()
        self.direction = direction
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        self.mutation_filter = mutation_filter
        self.grg_shape = (self.sample_count, grg.num_mutations)
        shape = (
            self.grg_shape
            if not mutation_filter
            else (self.grg_shape[0], len(self.mutation_filter))
        )
        if self.direction == _DOWN:
            shape = _transpose_shape(shape)
        super().__init__(grg, freqs, shape, dtype=dtype, haploid=haploid)

    def _matmat_direction(self, other_matrix, direction):
        mult_dir = _flip_dir(direction)

        def expandm(matrix):
            if self.mutation_filter and mult_dir == _DOWN:
                result = numpy.zeros(
                    (matrix.shape[0], self.grg_shape[1]), dtype=matrix.dtype
                )
                result[:, self.mutation_filter] = matrix  # type: ignore
                return result
            return matrix

        def contractm(matrix):
            if self.mutation_filter and mult_dir == _UP:
                return matrix[:, self.mutation_filter]  # type: ignore
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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
    ):
        if isinstance(mutation_filter, numpy.ndarray):
            mutation_filter = mutation_filter.tolist()
        num_muts = grg.num_mutations if not mutation_filter else len(mutation_filter)
        xtx_shape = (num_muts, num_muts)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.std_x_op = SciPyStdXOperator(
            grg,
            _UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
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


# Genetic relatedness matrix X*X^T operator on the standardized GRG
class SciPyStdXXTOperator(LinearOperator):
    """
    A scipy.sparse.linalg.LinearOperator on the matrix :math:`XX^T` represented by the GRG.
    This is for the standardized matrix, which is centered to the mean (based on allele
    frequencies) and standard devation (based on the binomial distribution where each individual
    is the result of :math:`p`, the ploidy, trials).

    This operator performs multiplications against the correlation matrix of the genotype matrix
    underlying the GRG. Can perform the operation :math:`X \\times X^T \\times A` (_matmat) or
    :math:`X \\times X^T \\times \overrightarrow{v}` (_matvec).

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
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        dtype: TypeAlias = numpy.float64,
        haploid: bool = False,
    ):
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        xxt_shape = (self.sample_count, self.sample_count)
        super().__init__(dtype=dtype, shape=xxt_shape)
        self.std_x_op = SciPyStdXOperator(grg, _UP, freqs, haploid=haploid, dtype=dtype)

    def _matmat(self, other_matrix):
        D = self.std_x_op._rmatmat(other_matrix)
        return self.std_x_op._matmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[pygrgl.GRG],
        direction: pygrgl.TraversalDirection,
        dtype=numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
        threads: int = 1,
    ):
        if isinstance(mutation_filter, numpy.ndarray):
            mutation_filter = mutation_filter.tolist()
        assert len(grgs) >= 1, "Must provide at least one GRG"
        self.direction = direction
        self.num_mutations = sum([g.num_mutations for g in grgs])
        self.mutation_filter = mutation_filter
        if self.mutation_filter:
            assert len(self.mutation_filter) <= self.num_mutations
            self.num_mutations = len(self.mutation_filter)
        self.operators = []
        num_samples = grgs[0].num_samples
        prev_max_mut = 0
        for g in grgs:
            assert g.num_samples == num_samples, "All GRGs must use the same samples"
            if self.mutation_filter:
                grg_mut_filt = list(
                    map(
                        lambda m: m - prev_max_mut,
                        filter(
                            lambda m: m >= prev_max_mut
                            and m < prev_max_mut + g.num_mutations,
                            self.mutation_filter,
                        ),
                    )
                )
                # If we have an overall filter, but no filter for _this_ GRG, then we just skip it.
                skip = len(grg_mut_filt) == 0
            else:
                grg_mut_filt = []
                skip = False
            if not skip:
                self.operators.append(
                    SciPyXOperator(
                        g,
                        direction,
                        dtype,
                        haploid=haploid,
                        mutation_filter=grg_mut_filt,
                    )
                )
            prev_max_mut += g.num_mutations
        # Should we concatenate the result for _matmat, or add them together?
        self.concat = self.direction == pygrgl.TraversalDirection.DOWN
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

    def _matmat_helper(self, other_matrix, direction, op_method):
        # For UP, we have "(N x M) x (M x k)", so we need to split the other_matrix into chunks of the
        # appropriate size <= M.
        futures = []
        if direction == pygrgl.TraversalDirection.UP:
            start = 0
            for op in self.operators:
                end = start + op.shape[1]
                # assert end <= other_matrix.shape[0] # FIX ME: this will cause error when using mutation_filter
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
        vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
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
    :param mutation_filter: Changes the dimensions of :math:`X` to be NxP (where P is the length of
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[pygrgl.GRG],
        dtype=numpy.float64,
        haploid: bool = False,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
        threads: int = 1,
    ):
        self.x_op = MultiSciPyXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            dtype=dtype,
            haploid=haploid,
            mutation_filter=mutation_filter,
            threads=threads,
        )
        xtx_shape = (self.x_op.num_mutations, self.x_op.num_mutations)
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.x_op._matmat(other_matrix)
        return self.x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        return self._matvec(vect)


# FIXME this can be factored into a common base class with MultiSciPyXOperator
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
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[pygrgl.GRG],
        direction: pygrgl.TraversalDirection,
        freqs: List[numpy.typing.NDArray],
        haploid: bool = False,
        dtype=numpy.float64,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
        threads: int = 1,
    ):
        if isinstance(mutation_filter, numpy.ndarray):
            mutation_filter = mutation_filter.tolist()
        assert len(grgs) >= 1, "Must provide at least one GRG"
        assert len(grgs) == len(freqs), "Must provide allele frequencies for every GRG"
        self.direction = direction
        self.num_mutations = sum([g.num_mutations for g in grgs])
        num_samples = grgs[0].num_samples
        self.mutation_filter = mutation_filter
        if mutation_filter:
            assert len(self.mutation_filter) <= self.num_mutations
            self.num_mutations = len(self.mutation_filter)
        prev_max_mut = 0
        self.operators = []
        for g, f in zip(grgs, freqs):
            assert g.num_samples == num_samples, "All GRGs must use the same samples"
            if self.mutation_filter:
                grg_mut_filt = list(
                    map(
                        lambda m: m - prev_max_mut,
                        filter(
                            lambda m: m >= prev_max_mut
                            and m < prev_max_mut + g.num_mutations,
                            self.mutation_filter,
                        ),
                    )
                )
                # If we have an overall filter, but no filter for _this_ GRG, then we just skip it.
                skip = len(grg_mut_filt) == 0
            else:
                grg_mut_filt = []
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
                    )
                )
            prev_max_mut += g.num_mutations
        # Should we concatenate the result for _matmat, or add them together?
        self.concat = self.direction == pygrgl.TraversalDirection.DOWN
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)

    def _matmat_helper(self, other_matrix, direction, op_method):
        # For UP, we have "(N x M) x (M x k)", so we need to split the other_matrix into chunks of the
        # appropriate size <= M.
        futures = []
        if direction == pygrgl.TraversalDirection.UP:
            start = 0
            for op in self.operators:
                end = start + op.shape[1]
                # assert end <= other_matrix.shape[0] # FIX ME: this will cause error when using mutation_filter
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
        vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
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
        mutation_filter) instead of NxM. Default: empty filter.
    :type mutation_filter: Union[List[int], numpy.typing.NDArray]
    :param threads: Number of threads for performing the multiplication. Each GRG can be done in
        parallel.
    :type threads: int
    """

    def __init__(
        self,
        grgs: List[pygrgl.GRG],
        freqs: List[numpy.typing.NDArray],
        haploid: bool = False,
        dtype=numpy.float64,
        mutation_filter: Union[List[int], numpy.typing.NDArray] = [],
        threads: int = 1,
    ):
        self.std_x_op = MultiSciPyStdXOperator(
            grgs,
            pygrgl.TraversalDirection.UP,
            freqs,
            haploid=haploid,
            dtype=dtype,
            mutation_filter=mutation_filter,
            threads=threads,
        )
        xtx_shape = (self.std_x_op.num_mutations, self.std_x_op.num_mutations)
        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
        D = self.std_x_op._matmat(other_matrix)
        return self.std_x_op._rmatmat(D)

    def _rmatmat(self, other_matrix):
        return self._matmat(other_matrix)

    def _matvec(self, vect):
        vect = numpy.array([vect]).T
        return self._matmat(vect)

    def _rmatvec(self, vect):
        return self._matvec(vect)
