"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
from typing import Tuple
import numpy
import pygrgl

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias  # type: ignore


def _flip_dir(direction: pygrgl.TraversalDirection) -> pygrgl.TraversalDirection:
    return (
        pygrgl.TraversalDirection.UP
        if direction == pygrgl.TraversalDirection.DOWN
        else pygrgl.TraversalDirection.DOWN
    )


class SciPyXOperator(LinearOperator):
    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: pygrgl.TraversalDirection,
        dtype=numpy.float64,
        haploid: bool = False,
    ):
        self.haploid = haploid
        self.grg = grg
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        self.direction = direction
        if self.direction == pygrgl.TraversalDirection.UP:
            shape = (self.sample_count, grg.num_mutations)
        else:
            shape = (grg.num_mutations, self.sample_count)
        super().__init__(dtype=dtype, shape=shape)

    def _matmat(self, other_matrix):
        return numpy.transpose(
            pygrgl.matmul(
                self.grg,
                other_matrix.T,
                _flip_dir(self.direction),
                by_individual=not self.haploid,
            )
        )

    def _rmatmat(self, other_matrix):
        return numpy.transpose(
            pygrgl.matmul(
                self.grg,
                other_matrix.T,
                self.direction,
                by_individual=not self.haploid,
            )
        )

    def _matvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Mx1)
        return self._matmat(vect)

    def _rmatvec(self, vect):
        vect = numpy.array([vect]).T  # Column vector (Nx1)
        return self._rmatmat(vect)


class SciPyXTXOperator(LinearOperator):
    def __init__(
        self,
        grg: pygrgl.GRG,
        dtype=numpy.float64,
        haploid: bool = False,
    ):
        xtx_shape = (grg.num_mutations, grg.num_mutations)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.x_op = SciPyXOperator(
            grg,
            pygrgl.TraversalDirection.UP,
            dtype=dtype,
            haploid=haploid,
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


class SciPyStandardizedOperator(LinearOperator):
    """
    (Abstract) base class for GRG-based scipy LinearOperators that standardize the underlying
    genotype matrix.
    """

    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        shape: Tuple[int, int],
        dtype=numpy.float64,
        haploid: bool = False,
    ):
        self.haploid = haploid
        self.grg = grg
        self.freqs = freqs
        self.mult_const = 1 if self.haploid else grg.ploidy

        # TODO: there might be other normalization approachs besides this. For example, FlashPCA2 has different
        # options for what to use (this is the P-trial binomial).
        raw = self.mult_const * freqs * (1.0 - freqs)

        # Two versions of sigma, the second flips 0 values (which means the frequency was
        # either 1 or 0 for the mutation) to 1 values so we can use it for division.
        self.original_sigma = numpy.sqrt(raw)
        self.sigma_corrected = numpy.where(
            self.original_sigma == 0,
            1,
            self.original_sigma,
        )
        super().__init__(dtype=dtype, shape=shape)


# Operator on the standardized GRG X or X^T (based on the direction chosen)
class SciPyStdXOperator(SciPyStandardizedOperator):
    def __init__(
        self,
        grg: pygrgl.GRG,
        direction: pygrgl.TraversalDirection,
        freqs: numpy.typing.NDArray,
        haploid: bool = False,
        dtype=numpy.float64,
    ):
        """
        Construct a LinearOperator compatible with scipy's sparse linear algebra module.
        Let X be the genotype matrix, as represented by the GRG, then this operator computes either
        the product (transpose(X) * v) or (X * v), where v is a vector of length num_mutations or
        num_samples depending on the direction.

        :param grg: The GRG the operator will multiply against.
        :type grg: pygrgl.GRG
        :param direction: The direction of GRG traversal, which defines whether we are multiplying against
            the X matrix (NxM, the UP direction) or the X^T matrix (MxN, the DOWN direction).
        :type direction: pygrgl.TraversalDirection
        :param freqs: A vector of length num_mutations, containing the allele frequency for all mutations.
            Indexed by the mutation ID of the mutation.
        :type freqs: numpy.ndarray
        :param haploid: Set to True to perform haploid computations instead of the ploidy of the individuals
            in the GRG.
        :type haploid: bool
        :param dtype: The numpy.dtype to use for the computation.
        :type dtype: numpy.dtype
        """
        self.direction = direction
        self.sample_count = grg.num_samples if haploid else grg.num_individuals
        if self.direction == pygrgl.TraversalDirection.UP:
            shape = (self.sample_count, grg.num_mutations)
        else:
            shape = (grg.num_mutations, self.sample_count)
        super().__init__(grg, freqs, shape, dtype=dtype, haploid=haploid)

    def _matmat_direction(self, other_matrix, direction):
        with numpy.errstate(divide="raise"):
            if direction == pygrgl.TraversalDirection.UP:
                vS = other_matrix.T / self.sigma_corrected
                XvS = numpy.transpose(
                    pygrgl.matmul(
                        self.grg,
                        vS,
                        _flip_dir(direction),
                        by_individual=not self.haploid,
                    )
                )
                consts = numpy.sum(self.mult_const * self.freqs * vS, axis=1)
                return XvS - consts.T
            else:
                assert direction == pygrgl.TraversalDirection.DOWN
                SXv = (
                    pygrgl.matmul(
                        self.grg,
                        other_matrix.T,
                        _flip_dir(direction),
                        by_individual=not self.haploid,
                    )
                    / self.sigma_corrected
                )
                col_const = numpy.sum(other_matrix, axis=0, keepdims=True).T
                sub_const2 = (
                    self.mult_const * self.freqs / self.sigma_corrected
                ) * col_const
                result = numpy.transpose(SXv - sub_const2)
                return result

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
    def __init__(
        self,
        grg: pygrgl.GRG,
        freqs: numpy.typing.NDArray,
        haploid: bool = False,
        dtype=numpy.float64,
    ):
        """
        Construct a LinearOperator compatible with scipy's sparse linear algebra module.
        Let X be the genotype matrix, as represented by the GRG, then this operator computes the product
        (transpose(X)*X) * v, where v is a vector of length num_mutations.

        :param grg: The GRG the operator will multiply against.
        :type grg: pygrgl.GRG
        :param freqs: A vector of length num_mutations, containing the allele frequency for all mutations.
            Indexed by the mutation ID of the mutation.
        :type freqs: numpy.ndarray
        :param haploid: Set to True to perform haploid computations instead of the ploidy of the individuals
            in the GRG.
        :type haploid: bool
        :param dtype: The numpy.dtype to use for the computation.
        :type dtype: numpy.dtype
        """
        xtx_shape = (grg.num_mutations, grg.num_mutations)
        super().__init__(dtype=dtype, shape=xtx_shape)
        self.std_x_op = SciPyStdXOperator(
            grg, pygrgl.TraversalDirection.UP, freqs, haploid=haploid, dtype=dtype
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
        self.std_x_op = SciPyStdXOperator(
            grg, pygrgl.TraversalDirection.UP, freqs, haploid=haploid, dtype=dtype
        )

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
