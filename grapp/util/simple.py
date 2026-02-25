"""
Simple utility functions.
"""

from enum import Enum
from typing import Union, Tuple, List
import pygrgl
import numpy


class UserInputError(Exception):
    pass


# This enum is just a container for string constants used below.
class _GenotypeDist(str, Enum):
    SAMPLE = "sample"
    BINOMIAL = "binomial"

    @classmethod
    def is_valid(cls, str_value: str) -> bool:
        return str_value in set(map(lambda x: x.value, cls))  # type: ignore


def _div_or_default(a, b, d):
    """
    y = a / b, unless b_i is 0, then y_i will be set to 0.

    :param a: Numerator
    :param b: Denominator
    :param d: Default value for when denominator is 0.
    """
    result = numpy.full(a.shape, d)
    return numpy.divide(a, b, out=result, where=(b != 0))


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


def variance(
    grg: pygrgl.GRG,
    dist: str = _GenotypeDist.BINOMIAL.value,
    adjust_missing: bool = False,
    mask_samples: Union[List[int], numpy.typing.NDArray] = [],
    haploid: bool = False,
):
    """
    Compute the variance of the mutations. You can use the ``dist`` parameter to choose
    between the sample variance and the binomial variance.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param dist: Either "sample" or "binomial".
    :type dist: str
    :param adjust_missing: Optional. Set to true to adjust each allele frequncy to be
        :math:`\\frac{count_i}{total - missing_i}` instead of :math:`\\frac{count_i}{total}`.
    :type adjust_missing: bool
    :param mask_samples: Ignore any contribution from the samples listed in this array.
    :type mask_samples: Union[List[int], numpy.typing.NDArray]
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    mult_const = 1 if haploid else grg.ploidy
    acount, miss_count = allele_counts(
        grg, return_missing=True, mask_samples=mask_samples
    )
    n_j = (
        (grg.num_samples - miss_count)
        if adjust_missing
        else numpy.full(grg.num_mutations, grg.num_samples)
    )
    afreq = _div_or_default(acount, n_j, 0.0)
    if dist == _GenotypeDist.SAMPLE.value:
        assert (
            not haploid and grg.ploidy == 2
        ), "The sample-based variance can only be computed for diploids"
        # diag(X^T @ X) / n = Var[X] + E[X]^2
        # --> Var[X] = (diag(X^T @ X) / n) - E[X]^2
        XX = pygrgl.matmul(
            grg,
            numpy.ones((1, grg.num_samples), dtype=numpy.int32),
            pygrgl.TraversalDirection.UP,
            init="xtx",
        )[0]
        return (XX / grg.num_individuals) - ((mult_const * afreq) ** 2)
    else:
        return mult_const * afreq * (1.0 - afreq)
