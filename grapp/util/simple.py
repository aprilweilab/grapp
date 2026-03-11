"""
Simple utility functions.
"""

from enum import Enum
from multiprocessing import Pool
from typing import Union, Tuple, List, Optional
from tqdm import tqdm
import pandas
import pygrgl
import numpy
import sys


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
    sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, numpy.typing.NDArray]]:
    """
    Get the allele counts for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param return_missing: Return two arrays: the allele counts, and the missingness counts.
    :type return_missing: bool
    :param sample_filter: Only consider the samples listed in the filter. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :return: A vector of length grg.num_mutations, containing allele counts
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    if isinstance(sample_filter, numpy.ndarray):
        sample_filter = sample_filter.tolist()
    if sample_filter is not None:
        assert len(set(sample_filter)) == len(
            sample_filter
        ), "Duplicate IDs in sample_filter"
        assert len(sample_filter) <= grg.num_samples
    kwargs = {}
    if return_missing:
        miss_counts = numpy.zeros((1, grg.num_mutations), dtype=numpy.int32)
        kwargs["miss"] = miss_counts
    else:
        miss_counts = None
    if sample_filter is not None:
        input_mat = numpy.zeros((1, grg.num_samples), dtype=numpy.int32)
        input_mat[:, sample_filter] = 1
    else:
        input_mat = numpy.ones((1, grg.num_samples), dtype=numpy.int32)
    acounts = pygrgl.matmul(grg, input_mat, pygrgl.TraversalDirection.UP, **kwargs)[0]
    if miss_counts is not None:
        miss_counts = miss_counts[0]
        assert miss_counts is not None
        return acounts, miss_counts
    return acounts


def allele_frequencies(
    grg: pygrgl.GRG,
    adjust_missing: bool = False,
    sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
) -> numpy.typing.NDArray:
    """
    Get the allele frequencies for the mutations in the given GRG.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param adjust_missing: Optional. Set to true to adjust each allele frequncies to be
        :math:`\\frac{count_i}{total - missing_i}` instead of :math:`\\frac{count_i}{total}`.
    :type adjust_missing: bool
    :param sample_filter: Only consider the samples listed in the filter. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    with numpy.errstate(divide="raise"):
        if adjust_missing:
            acounts, miss_counts = allele_counts(
                grg, return_missing=True, sample_filter=sample_filter
            )
        else:
            acounts = allele_counts(
                grg, return_missing=False, sample_filter=sample_filter
            )
            miss_counts = 0
        num_samples = grg.num_samples if sample_filter is None else len(sample_filter)
        denominator = num_samples - miss_counts
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
    sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
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
    :param sample_filter: Only consider the samples listed in the filter. Default: no filter.
    :type sample_filter: Optional[Union[List[int], numpy.typing.NDArray]]
    :return: A vector of length grg.num_mutations, containing allele frequencies
        indexed by MutationID.
    :rtype: numpy.ndarray
    """
    mult_const = 1 if haploid else grg.ploidy
    acount, miss_count = allele_counts(
        grg, return_missing=True, sample_filter=sample_filter
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


def _star_snphwe_pygrgl(arglist):
    het_A, hom_A, other, mut_ids = arglist
    return (pygrgl.hwe_exact_pv(het_A, hom_A, other), mut_ids)


def hwe(
    grg: pygrgl.GRG,
    pv_only: bool = False,
    jobs: int = 1,
    show_progress: bool = False,
    batch_size: int = 1000,
) -> pandas.DataFrame:
    """
    Compute hardy-weinberg p-values for all variants in the GRG. Missing data is not yet supported.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param pv_only: Only emit the p-values in the dataframe, not other information. Default: False.
    :type pv_only: bool
    :param jobs: Number of parallel jobs to run (threads). Default: 1.
    :type jobs: int
    :param show_progress: Show progress bar on sys.stderr. Default: False.
    :type show_progress: bool
    :return: A DataFrame containing "POS", "ALT", "COUNT", and "P".
    :rtype: pandas.DataFrame
    """
    # TODO: better testing with missing data in general, but also GRGL should be able to detect whether
    # there are partially missing genotypes as well.
    if grg.has_missing_data:
        print(
            "WARNING! HWE implementation is formulated for missingness to be per-individual, not per-haplotype. "
            "Your results may be inaccurate if you have partially missing genotypes",
            file=sys.stderr,
        )
        raise RuntimeError("Missing data not yet supported for HWE")

    if show_progress:
        print(f"Calculating heterozygote and homozygote counts...", file=sys.stderr)
    # Get the het and hom count information for all variants
    inmat = numpy.vstack(
        [
            numpy.zeros(grg.num_samples, dtype=numpy.int32),  # hom only
            numpy.ones(grg.num_samples, dtype=numpy.int32),  # het and hom
        ]
    )
    zyg_info = pygrgl.matmul(
        grg,
        inmat,
        pygrgl.TraversalDirection.UP,
        init="xtx",
    )
    del inmat
    hom_A = zyg_info[0] // 2
    het_A = zyg_info[1] - (zyg_info[0] * 2)
    del zyg_info
    n_A = het_A + 2 * hom_A
    if show_progress:
        print(f"Done.", file=sys.stderr)

    # TODO: subtract missingness from "other"
    other = (grg.num_individuals - (het_A + hom_A)).tolist()

    # Faster access.
    hom_A = hom_A.tolist()
    het_A = het_A.tolist()

    pvalues = numpy.zeros(grg.num_mutations)
    if jobs == 1:
        progress = (lambda x: x) if not show_progress else tqdm
        for mut_id in range(grg.num_mutations):
            pvalues[mut_id] = pygrgl.hwe_exact_pv(
                het_A[mut_id], hom_A[mut_id], other[mut_id]
            )
    else:
        arglist = [
            (het_A[mut_id], hom_A[mut_id], other[mut_id], mut_id)
            for mut_id in range(grg.num_mutations)
        ]
        with Pool(jobs) as pool:
            if show_progress:
                results = list(
                    tqdm(
                        pool.imap_unordered(_star_snphwe_pygrgl, arglist, batch_size),
                        total=grg.num_mutations,
                    )
                )
            else:
                results = list(
                    pool.imap_unordered(_star_snphwe_pygrgl, arglist, batch_size)
                )
            for result, b in results:
                pvalues[b] = result

    if pv_only:
        return pandas.DataFrame({"P": pvalues}, copy=False)

    positions = []
    alts = []
    for mut_id in range(grg.num_mutations):
        mut = grg.get_mutation_by_id(mut_id)
        positions.append(mut.position)
        alts.append(mut.allele)
    # TODO: all these methods should output REF also?
    return pandas.DataFrame(
        {
            "POS": positions,
            "ALT": alts,
            "COUNT": n_A,
            # TODO: output genotype counts also
            "P": pvalues,
        },
        copy=False,
    )


# Get lists of mutations for various properties:
# 1. multi-allelic/bi-allelic
# 2. SNPs/other categories
# 3. ALT is minor (freq < 0.5)
# 4. Has missingness
#
# Collect mutation IDs by site, so List[List[int]]
