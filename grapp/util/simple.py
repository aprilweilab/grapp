"""
Simple utility functions.
"""

from typing import Union, Optional
import pygrgl
import numpy


class UserInputError(Exception):
    pass


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
    kwargs = {}
    miss: Optional[Union[numpy.typing.NDArray, int]] = 0
    if adjust_missing:
        miss = numpy.zeros(grg.num_mutations, dtype=numpy.int32)
        kwargs["miss"] = miss
    else:
        miss = None
    counts = pygrgl.matmul(
        grg,
        numpy.ones((1, grg.num_samples), dtype=numpy.int32),
        pygrgl.TraversalDirection.UP,
        **kwargs,
    )[0]
    if miss is None:
        miss = 0
    return counts / (grg.num_samples - miss)
