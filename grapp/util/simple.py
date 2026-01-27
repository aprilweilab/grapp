"""
Simple utility functions.
"""

from typing import Union, Tuple, List
import pygrgl
import numpy


class UserInputError(Exception):
    pass


def allele_counts(
    grg: pygrgl.GRG,
    return_missing: bool = False,
    mask_samples: Union[List[int], numpy.typing.NDArray] = [],
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, numpy.typing.NDArray]]:
    """
    Get the allele counts for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param return_missing: Return two arrays: the allele counts, and the missingness counts.
    :type return_missing: bool
    :param sample_filter: Optional. Restrict the counts to a subset of samples, listed by sample node ID.
    :type sample_filter: List[int]
    :param mask_samples: Ignore any contribution from the samples listed in this array.
    :type mask_samples: Union[List[int], numpy.typing.NDArray]
    :return: A vector of length grg.num_mutations, containing allele counts
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    assert (
        isinstance(mask_samples, list) or mask_samples.ndim == 1
    ), "mask_samples must be a list or 1D array"
    kwargs = {}
    if return_missing:
        miss_counts = numpy.zeros((1, grg.num_mutations), dtype=numpy.int32)
        kwargs["miss"] = miss_counts
    else:
        miss_counts = None
    input_mat = numpy.ones((1, grg.num_samples), dtype=numpy.int32)
    if mask_samples:
        input_mat[:, mask_samples] = 0
    acounts = pygrgl.matmul(grg, input_mat, pygrgl.TraversalDirection.UP, **kwargs)[0]
    if miss_counts is not None:
        miss_counts = miss_counts[0]
        assert miss_counts is not None
        return acounts, miss_counts
    return acounts


def allele_frequencies(
    grg: pygrgl.GRG,
    adjust_missing: bool = False,
    mask_samples: Union[List[int], numpy.typing.NDArray] = [],
) -> numpy.typing.NDArray:
    """
    Get the allele frequencies for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param adjust_missing: Optional. Set to true to adjust each allele frequncies to be
        :math:`\\frac{count_i}{total - missing_i}` instead of :math:`\\frac{count_i}{total}`.
    :type adjust_missing: bool
    :param mask_samples: Ignore any contribution from the samples listed in this array.
    :type mask_samples: Union[List[int], numpy.typing.NDArray]
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    with numpy.errstate(divide="raise"):
        if adjust_missing:
            acounts, miss_counts = allele_counts(
                grg, return_missing=True, mask_samples=mask_samples
            )
        else:
            acounts = allele_counts(
                grg, return_missing=False, mask_samples=mask_samples
            )
            miss_counts = 0
        denominator = grg.num_samples - miss_counts - len(mask_samples)
        assert numpy.all(denominator >= 0)
        return numpy.divide(
            acounts,
            denominator,
            out=numpy.zeros(acounts.shape, dtype=numpy.float64),
            where=(denominator != 0),
        )
