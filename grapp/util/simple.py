"""
Simple utility functions.
"""

from typing import Union, Tuple
import pygrgl
import numpy


class UserInputError(Exception):
    pass


def allele_counts(
    grg: pygrgl.GRG, return_missing: bool = False
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, numpy.typing.NDArray]]:
    """
    Get the allele counts for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param return_missing: Return two arrays: the allele counts, and the missingness counts.
    :type return_missing: bool
    :return: A vector of length grg.num_mutations, containing allele counts
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    kwargs = {}
    if return_missing:
        miss_counts = numpy.zeros((1, grg.num_mutations), dtype=numpy.int32)
        kwargs["miss"] = miss_counts
    else:
        miss_counts = None
    acounts = pygrgl.matmul(
        grg,
        numpy.ones((1, grg.num_samples), dtype=numpy.int32),
        pygrgl.TraversalDirection.UP,
        **kwargs,
    )[0]
    if miss_counts is not None:
        miss_counts = miss_counts[0]
        assert miss_counts is not None
        return acounts, miss_counts
    return acounts


def allele_frequencies(
    grg: pygrgl.GRG, adjust_missing: bool = False
) -> numpy.typing.NDArray:
    """
    Get the allele frequencies for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param adjust_missing: Optional. Set to true to adjust each allele frequncies to be
        :math:`\\frac{count_i}{total - missing_i}` instead of :math:`\\frac{count_i}{total}`.
    :type adjust_missing: bool
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    with numpy.errstate(divide="raise"):
        if adjust_missing:
            acounts, miss_counts = allele_counts(grg, return_missing=True)
        else:
            acounts = allele_counts(grg, return_missing=False)
            miss_counts = 0
        denominator = grg.num_samples - miss_counts
        return numpy.divide(
            acounts,
            denominator,
            out=numpy.zeros(acounts.shape, dtype=numpy.float64),
            where=(denominator != 0),
        )
