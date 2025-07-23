"""
Simple utility functions.
"""

import pygrgl
import numpy


class UserInputError(Exception):
    pass


def allele_frequencies(grg: pygrgl.GRG) -> numpy.typing.NDArray:
    """
    Get the allele frequencies for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    return pygrgl.matmul(
        grg,
        numpy.ones((1, grg.num_samples), dtype=numpy.int32),
        pygrgl.TraversalDirection.UP,
    )[0] / (grg.num_samples)
