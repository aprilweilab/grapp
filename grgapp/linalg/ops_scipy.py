"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
from typing import Tuple
import numpy
import pygrgl


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

    def _matmat(self, other_matrix):
        with numpy.errstate(divide="raise"):
            if self.direction == pygrgl.TraversalDirection.UP:
                vS = other_matrix.T / self.sigma_corrected
                XvS = numpy.transpose(
                    pygrgl.matmul(
                        self.grg,
                        vS,
                        _flip_dir(self.direction),
                        by_individual=not self.haploid,
                    )
                )
                consts = numpy.sum(self.mult_const * self.freqs * vS, axis=1)
                return XvS - consts.T
            else:
                assert self.direction == pygrgl.TraversalDirection.DOWN
                SXv = (
                    pygrgl.matmul(
                        self.grg,
                        other_matrix.T,
                        _flip_dir(self.direction),
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


# Correlation matrix X^T*X operator on the standardized GRG
class SciPyStdXTXOperator(SciPyStandardizedOperator):
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
        super().__init__(grg, freqs, xtx_shape, dtype=dtype, haploid=haploid)

    def _matmat(self, other_matrix):
        with numpy.errstate(divide="raise"):
            out1 = numpy.zeros_like(other_matrix.T, dtype=float)
            vS = numpy.divide(
                numpy.transpose(other_matrix),
                self.original_sigma,
                out=out1,
                where=self.original_sigma != 0,
            )
            XvS = numpy.transpose(
                pygrgl.matmul(
                    self.grg,
                    vS,
                    pygrgl.TraversalDirection.DOWN,
                    by_individual=not self.haploid,
                )
            )
            sub_const = (
                self.mult_const
                * self.freqs
                * numpy.transpose(other_matrix)
                / self.sigma_corrected
            )
            D = XvS - numpy.sum(sub_const, axis=1, keepdims=True).T
            # SXD is a 1xM vector (or KxM, if the input matrix has K columns)
            SXD = (
                pygrgl.matmul(
                    self.grg,
                    numpy.transpose(D),
                    pygrgl.TraversalDirection.UP,
                    by_individual=not self.haploid,
                )
                / self.sigma_corrected
            )
            col_const = numpy.sum(D, axis=0, keepdims=True).T
            sub_const2 = (
                self.mult_const * self.freqs / self.sigma_corrected
            ) * col_const
            result = numpy.transpose(SXD - sub_const2)
            return result
