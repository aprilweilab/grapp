"""
Simple utility functions.
"""

from enum import Enum
from multiprocessing import Pool
from typing import Union, Tuple, List, Optional, Set
from tqdm import tqdm
import pandas
import pygrgl
import numpy
import sys


class UserInputError(Exception):
    pass


class VariantType(Enum):
    SNPS = "snps"  # Length=1
    INDELS = "indels"  # Length <50
    MNPS = "mnps"  # Length of ALT same as length of REF
    OTHER = "other"  # Anything else

    def __str__(self):
        return self.value


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


def common_mut_dataframe(grg, **kwargs):
    """
    Generate the "common" output format for mutation-based dataframes, which has "POS", "ALT",
    and "REF" in the first three columns, and then whatever extra columns the user provides.

    :param kwargs: Keyword arguments are just passed through to pandas.DataFrame({}).
    :return: The dataframe, with copy=False.
    :rtype: pandas.DataFrame
    """
    positions = []
    alts = []
    refs = []
    for mut_id in range(grg.num_mutations):
        mut = grg.get_mutation_by_id(mut_id)
        positions.append(mut.position)
        alts.append(mut.allele)
        refs.append(mut.ref_allele)
    dict_df = {"POS": positions, "ALT": alts, "REF": refs}
    dict_df.update(kwargs)
    return pandas.DataFrame(dict_df, copy=False)


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
    jobs: int = 1,
    show_progress: bool = False,
    return_counts: bool = False,
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, numpy.typing.NDArray]]:
    """
    Compute hardy-weinberg p-values for all variants in the GRG. Missing data is not yet supported.

    NOTES:

    * Multi-allelic sites only have p-values calculated for the REF/ALT combinations that are present,
      and the calculations are based on hetALT, homALT, and other, where other is the number of genotypes
      that do not contain ALT. We do not "flip" the ALT and REF and test hetREF, homREF, etc.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param jobs: Number of parallel jobs to run (threads). Default: 1.
    :type jobs: int
    :param show_progress: Show progress bar on sys.stderr. Default: False.
    :type show_progress: bool
    :return: A numpy array of length num_mutations, containing a p-value for each mutation.
    :rtype: numpy.array
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
        batch_size = 1000
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
    if return_counts:
        return pvalues, n_A
    return pvalues


def hwe_df(
    grg: pygrgl.GRG,
    jobs: int = 1,
    show_progress: bool = False,
) -> pandas.DataFrame:
    """
    Compute hardy-weinberg p-values for all variants in the GRG. Missing data is not yet supported.

    NOTES:

    * Multi-allelic sites only have p-values calculated for the REF/ALT combinations that are present,
      and the calculations are based on hetALT, homALT, and other, where other is the number of genotypes
      that do not contain ALT. We do not "flip" the ALT and REF and test hetREF, homREF, etc.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param jobs: Number of parallel jobs to run (threads). Default: 1.
    :type jobs: int
    :param show_progress: Show progress bar on sys.stderr. Default: False.
    :type show_progress: bool
    :return: A DataFrame containing "POS", "ALT", "COUNT", and "P".
    :rtype: pandas.DataFrame
    """
    pvalues, n_A = hwe(grg, jobs=jobs, show_progress=show_progress, return_counts=True)
    return common_mut_dataframe(grg, COUNT=n_A, P=pvalues)


def site_alleles(
    grg: pygrgl.GRG,
    alt_only: bool = False,
) -> numpy.typing.NDArray:
    """
    Compute the number of alleles at the site associated with each mutation (variant).
    For example, if there is a site with 3 variants A>T, A>G, A>C, then each of those
    variants (mutations) will have a "4" in their result. Each variant is always bi-allelic,
    but the site it is associated can have an arbitrary number of alleles.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param alt_only: Only count ALT alleles, not REF alleles. Default: False.
    :type alt_only: bool
    :return: A numpy array of length num_mutations, containing a allele count for each mutation.
    :rtype: numpy.array
    """
    result = []
    allele_set = set()
    prev_pos = -1
    to_add = 0
    for mut_id in range(grg.num_mutations):
        mut = grg.get_mutation_by_id(mut_id)
        if prev_pos == mut.position:
            allele_set.add(mut.allele)
            if not alt_only:
                allele_set.add(mut.ref_allele)
            to_add += 1
        else:
            if allele_set:
                result.extend([len(allele_set)] * to_add)
            allele_set = set([mut.allele])
            if not alt_only:
                allele_set.add(mut.ref_allele)
            prev_pos = mut.position
            to_add = 1
    if allele_set:
        result.extend([len(allele_set)] * to_add)
    res_array = numpy.array(result, dtype=numpy.int32)
    assert res_array.shape[0] == grg.num_mutations, (
        res_array.shape[0],
        grg.num_mutations,
    )
    return res_array


def get_variant_type(mut: pygrgl.Mutation) -> VariantType:
    ref_len = len(mut.ref_allele)
    alt_len = len(mut.allele)
    if ref_len == alt_len:
        if ref_len == 1:
            my_type = VariantType.SNPS
        else:
            my_type = VariantType.MNPS
    elif ref_len < 50 and alt_len < 50:
        my_type = VariantType.INDELS
    else:
        my_type = VariantType.OTHER
    return my_type


def variants_of_types(
    grg: pygrgl.GRG,
    types: Set[VariantType],
) -> List[int]:
    """
    Return the list of MutationIDs for variants of the given types. For example, passing
    types={VariantTypes.SNPS, VariantTypes.MNPS} will return every mutation that is either a SNP
    or MNP.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param types: Set of VariantType that is the union of types to return.
    :type types: Set[VariantType]
    :return: A list of MutationIDs.
    :rtype: List[int]
    """
    result = []
    for mut_id in range(grg.num_mutations):
        mut = grg.get_mutation_by_id(mut_id)
        if get_variant_type(mut) in types:
            result.append(mut_id)
    return result
