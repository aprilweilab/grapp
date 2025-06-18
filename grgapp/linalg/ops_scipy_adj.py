"""
Linear operators that are compatible with scipy.
"""

from scipy.sparse.linalg import LinearOperator
import numpy
import pygrgl

numpy.seterr(all="raise")


# Correlation matrix X^T*X operator on the standardized GRG
class SciPyXTXOperator(LinearOperator):
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
        :param sigma: A vector of length num_mutations, containing the standard
        """
        self.haploid = haploid
        self.grg = grg
        self.freqs = freqs
        # TODO: there might be other normalization approachs besides this. For example, FlashPCA2 has different
        # options for what to use (this is the 2-trial binomial).
        self.mult_const = 1 if self.haploid else 2
        # TODO: when haploid is True, double check that this is correct
        # before the sqrt

        ### WEIRD ISSUES WITH FLOATING POINTS CAUSING NEGATIVE VALUES
        raw = self.mult_const * freqs * (1 - freqs)
        self.original_sigma = numpy.sqrt(raw)
        self.sigma_corrected = numpy.where(
            self.original_sigma == 0,
            1,
            self.original_sigma,
        )

        xtx_shape = (grg.num_mutations, grg.num_mutations)

        super().__init__(dtype=dtype, shape=xtx_shape)

    def _matmat(self, other_matrix):
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
        D = XvS - numpy.sum(sub_const)
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
        col_const = numpy.sum(D)
        sub_const2 = (self.mult_const * self.freqs / self.sigma_corrected) * col_const
        result = numpy.transpose(SXD - numpy.transpose(sub_const2))
        return result
