"""
Simple utility functions.
"""

from grapp.grg_calculator import (
    GRGCalcInterface as _GRGCalcInterface,
    _wrap_grg,
)
from grapp.util.exceptions import UserInputError
from enum import Enum
from multiprocessing import Pool
from typing import Union, Tuple, List, Optional, Set
from tqdm import tqdm
import pandas
import pygrgl
import numpy
import sys


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


def common_mut_dataframe(grg: _GRGCalcInterface, **kwargs):
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
    grg: Union[pygrgl.GRG, _GRGCalcInterface],
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
    grg = _wrap_grg(grg)
    if isinstance(sample_filter, numpy.ndarray):
        sample_filter = sample_filter.tolist()
    if sample_filter is not None:
        assert len(set(sample_filter)) == len(
            sample_filter
        ), "Duplicate IDs in sample_filter"
        assert len(sample_filter) <= grg.num_samples
    kwargs = {}
    with grg.device_context():
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
        acounts = grg.matmul(input_mat, pygrgl.TraversalDirection.UP, **kwargs)[0]  # type: ignore
    if miss_counts is not None:
        miss_counts = miss_counts[0]
        assert miss_counts is not None
        return acounts, miss_counts
    return acounts


def allele_frequencies(
    grg: Union[pygrgl.GRG, _GRGCalcInterface],
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
    grg = _wrap_grg(grg)
    with grg.device_context():
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
    grg: Union[pygrgl.GRG, _GRGCalcInterface],
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
    grg = _wrap_grg(grg)
    with grg.device_context():
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
            XX = grg.matmul(
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


def get_zygosities(grg: pygrgl.GRG) -> numpy.typing.NDArray:
    """
    For a diploid dataset, return information about the homo/heterzygosity of every variant.
    Result is a matrix with 4 rows and grg.num_mutations columns. The rows are:

    * The number of homozygotes for each mutation (ALT of the variant)
    * The number of heterozygotes for each mutation
    * The number of homozygote-missing alleles for each mutation (i.e., corresponding site)
    * the number of heterozygote-missing alleles for each mutation

    :param grg: The GRG.
    :type grg: pygrgl.GRG.
    :return: :math:`4 \\times M` matrix, with rows as described above.
    :rtype: numpy.ndarray
    """
    if grg.ploidy != 2:
        raise UserInputError(
            f"get_zygosities requires for ploidy=2 (dataset ploidy={grg.ploidy})"
        )
    if grg.has_missing_data:
        miss_mat = numpy.zeros((2, grg.num_mutations), dtype=numpy.uint32)
    else:
        miss_mat = None
    inmat = numpy.vstack(
        [
            numpy.zeros(grg.num_samples, dtype=numpy.uint32),  # hom only
            numpy.ones(grg.num_samples, dtype=numpy.uint32),  # het and hom
        ]
    )
    zyg_info = pygrgl.matmul(
        grg,
        inmat,
        pygrgl.TraversalDirection.UP,
        init="xtx",
        miss=miss_mat,
    )
    hom_A = zyg_info[0] // 2
    het_A = zyg_info[1] - (zyg_info[0] * 2)
    if miss_mat is not None:
        hom_miss = miss_mat[0] // 2
        het_miss = miss_mat[1] - (miss_mat[0] * 2)
    else:
        hom_miss = numpy.zeros(hom_A.shape, dtype=numpy.uint32)
        het_miss = numpy.zeros(hom_A.shape, dtype=numpy.uint32)
    return numpy.vstack((hom_A, het_A, hom_miss, het_miss))


def hwe_from_counts(
    het_A: List[int],
    hom_A: List[int],
    other: List[int],
    jobs: int = 1,
    show_progress: bool = False,
) -> List[float]:
    """
    For the given heterozygous, homozygous, and "other" counts, compute the HWE exact p-values.

    :param het_A: List of integer counts for the number of heterozygous individuals (in a focal allele).
    :type het_A: List[int]
    :param hom_A: List of integer counts for the number of homozygous individuals (in a focal allele).
    :type hom_A: List[int]
    :param other: List of integer counts for the number of individuals that do not contain the
        focal allele at all.
    :type other: List[int]
    :param jobs: Number of threads to use.
    :type jobs: int
    :param show_progress: Write progress information to stderr? Default: False.
    :type show_progress: bool
    :return: A list of p-values, one for each focal allele.
    :rtype: List[float]
    """
    assert len(het_A) == len(hom_A), "Input lists must all be the same length."
    assert len(other) == len(het_A), "Input lists must all be the same length."
    if show_progress:
        print(f"Calculating HWE p-values...", file=sys.stderr)
    pvalues = [0.0] * len(het_A)
    if jobs == 1:
        for i in tqdm(range(len(het_A)), disable=not show_progress):
            pvalues[i] = pygrgl.hwe_exact_pv(het_A[i], hom_A[i], other[i])
    else:
        batch_size = 1000
        arglist = [(het_A[i], hom_A[i], other[i], i) for i in range(len(het_A))]
        with Pool(jobs) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_star_snphwe_pygrgl, arglist, batch_size),
                    total=len(het_A),
                    disable=not show_progress,
                )
            )
            for result, b in results:
                pvalues[b] = float(result)
    if show_progress:
        print(f"Done.", file=sys.stderr)
    return pvalues


def site_samples(grg: pygrgl.GRG, multi_list: List[List[int]]) -> numpy.typing.NDArray:
    """
    Given a list of sites (each site being a list of MutationIDs), return a bool numpy matrix that
    represents the samples that have either the ALT or a missing allele at that site.

    :param grg: The GRG.
    :type grg: pygrgl.GRG.
    :param multi_list: A list of "sites", where each site is a list of integer MutationIDs. Those
        mutations all have the same base-pair position, hence are at the same site.
    :type multi_list: List[List[int]]
    :return: Numpy matrix of dimension :math:`K \times N`, where :math:`K` is the number of sites
        that was passed in, and :math:`N` is grg.num_samples.
    :rtype: numpy.ndarray
    """
    k = len(multi_list)
    input_mat = numpy.zeros((k, grg.num_mutations), dtype=bool)
    if grg.has_missing_data:
        miss_mat = numpy.zeros((k, grg.num_mutations), dtype=bool)
    else:
        miss_mat = None
    for j, indices in enumerate(multi_list):
        input_mat[j, indices] = 1
        # For missingness, we only populate the first mutation, because all mutations at a site
        # share the same missingness node.
        if miss_mat is not None:
            miss_mat[j, indices[0]] = 1
    return pygrgl.matmul(grg, input_mat, pygrgl.TraversalDirection.DOWN, miss=miss_mat)


def ref_hwe(
    grg: pygrgl.GRG,
    jobs: int = 1,
    show_progress: bool = False,
    default: Union[numpy.typing.NDArray, float] = numpy.nan,
) -> numpy.typing.NDArray:
    """
    For every mutation, return the HWE p-value comparing REF against not-REF. For bi-allelic
    sites return a defualt value, since the (REF, not REF) p-value is the same as the (ALT, not ALT)
    p-value. For multi-allelic sites, performs multiple graph traversals (slow) to retrieve the REF
    sample list to explicitly compute the homozygous/heterzygous counts.

    WARNING: This is an expensive operation on large datasets.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param jobs: The number of threads to use when computing p-values.
    :type jobs: int
    :param show_progress: When True, show a progress bar.
    :type show_progress: bool
    :param default: Either a scalar value or an array of length grg.num_mutations. When a site is
        bi-allelic, use this default value. Default: NaN.
    :type default: Union[numpy.ndarray, float]
    :return: Array of grg.num_mutations p-values.
    :rtype: numpy.ndarray
    """
    if grg.ploidy != 2:
        raise UserInputError(
            f"HWE only works for ploidy=2 (dataset ploidy={grg.ploidy})"
        )
    if show_progress:
        print(
            f"Calculating multi-allelic heterozygote and homozygote counts...",
            file=sys.stderr,
        )

    miss_counts: Union[int, numpy.typing.NDArray] = 0
    if grg.has_missing_data:
        miss_counts = numpy.zeros((1, grg.num_mutations), dtype=numpy.int32)
        input_mat = numpy.ones((1, grg.num_samples), dtype=numpy.int32)
        acounts = grg.matmul(input_mat, pygrgl.TraversalDirection.UP, miss=miss_counts)[
            0
        ]
        miss_counts = miss_counts[0]

    # Get only the multi-allelic sites.
    multi_list = multi_allelic_muts(grg)
    hom_REF = numpy.zeros(len(multi_list), dtype=numpy.uint32)
    het_REF = numpy.zeros(len(multi_list), dtype=numpy.uint32)

    # For each batch of multi-allelic sites, get the sample list of all ALT and missing alleles,
    # then invert it so we can count hom/het.
    batch_size = 256
    for i in tqdm(range(0, len(multi_list), batch_size), disable=not show_progress):
        batch = multi_list[i : i + batch_size]
        ref_samples = ~site_samples(grg, batch)
        assert ref_samples.shape[1] % 2 == 0, "Internal error: ploidy == 2 expected"
        dosage = ref_samples[:, 0::2].astype(numpy.uint32) + ref_samples[
            :, 1::2
        ].astype(numpy.uint32)
        hom_REF[i : i + len(batch)] = numpy.count_nonzero(dosage == 2, axis=1)
        het_REF[i : i + len(batch)] = numpy.count_nonzero(dosage == 1, axis=1)
    n_REF = het_REF + 2 * hom_REF
    if show_progress:
        print(f"Done.", file=sys.stderr)

    N = (grg.num_samples - miss_counts) // 2
    other = numpy.maximum(0, (N - (het_REF + hom_REF))).tolist()

    # Faster access.
    pvalues_REF = hwe_from_counts(
        het_REF.tolist(),
        hom_REF.tolist(),
        other,
        jobs=jobs,
        show_progress=show_progress,
    )
    if isinstance(default, float):
        pvalues = numpy.full(grg.num_mutations, default)
    else:
        pvalues = default.copy()
    for i, indices in enumerate(multi_list):
        pvalues[list(indices)] = pvalues_REF[i]
    return pvalues


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
    :return: A numpy array of length num_mutations, containing a p-value for each mutation. If the
    :rtype: numpy.array
    """
    if grg.ploidy != 2:
        raise UserInputError(
            f"HWE only works for ploidy=2 (dataset ploidy={grg.ploidy})"
        )
    if show_progress:
        print(f"Calculating heterozygote and homozygote counts...", file=sys.stderr)
    zygosities = get_zygosities(grg)

    het_missing = numpy.sum(zygosities[3])
    if het_missing > 0:
        print(
            "WARNING! Dataset has heterozygous missingness. The HWE implementation is formulated for missingness "
            "by individual (homozygous missingness). Accuracy of HWE p-values may be affected.",
            file=sys.stderr,
        )

    hom_A = zygosities[0, :]
    het_A = zygosities[1, :]
    n_A = het_A + 2 * hom_A
    if show_progress:
        print(f"Done.", file=sys.stderr)

    # We treat every individual that has at least one allele missing as being missing. This breaks down
    # when there is a lot of heterozygous missingness (see WARNING above).
    missing_indivs = zygosities[2] + zygosities[3]
    N = grg.num_individuals - missing_indivs
    other = numpy.maximum(0, (N - (het_A + hom_A))).tolist()

    # Faster access.
    hom_A = hom_A.tolist()
    het_A = het_A.tolist()
    pvalues = numpy.array(
        hwe_from_counts(het_A, hom_A, other, jobs=jobs, show_progress=show_progress)
    )
    if return_counts:
        return pvalues, n_A
    return pvalues


def hwe_df(
    grg: pygrgl.GRG,
    jobs: int = 1,
    show_progress: bool = False,
    all_multi: bool = True,
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
    :param all_multi: Compute p-values for all combinations of multi-allelic sites (e.g., including
        the REF allele). For a bi-allelic site, there is a single p-value that represents the pair
        (REF, ALT). However, for a multi-allelic site, e.g. (REF, A1, A2), there are three combos
        (A1, not A1), (A2, not A2), and (REF, not REF). Setting this parameter to False will only
        compute two p-values: (A1, not A1) and (A2, not A2). Leaving it as True will additionally
        compute (REF, not REF).
    :type all_multi: bool
    :return: A DataFrame containing "POS", "ALT", "COUNT", and "P". If all_multi=True, then also
        includes column "REFP" for the REF allele's p-value.
    :rtype: pandas.DataFrame
    """
    if show_progress and not all_multi:
        num_multi = len(multi_allelic_muts(grg))
        if num_multi > 0:
            print(
                f"WARNING! Your data contains {num_multi} multi-allelic sites, but HWE does not compute REF vs. not-REF "
                "p-values by default. You may want to use --multi-ref to force this (slow) calculation, if you care "
                "about multi-allelic sites.",
                file=sys.stderr,
            )
    pvalues, nA = hwe(grg, jobs=jobs, show_progress=show_progress, return_counts=True)
    df = common_mut_dataframe(grg, COUNT=nA, P=pvalues)
    if all_multi:
        ref_pvs = ref_hwe(
            grg,
            jobs=jobs,
            show_progress=show_progress,
            default=numpy.array(pvalues),
        )
        df["REFP"] = ref_pvs
    return df


def multi_allelic_muts(grg: pygrgl.GRG) -> List[List[int]]:
    """
    Return a list of MutationId lists, where each sublist represents a set of Mutations that exist
    at the same site (base-pair position). An empty list implies the data is bi-allelic.

    :param grg: The GRG containing the mutations.
    :type grg: pygrgl.GRG
    :return: A list of lists [i, i+1, ..., i+k], which are MutationIds that has the same underlying
        base-pair position (site). An empty list implies the data is bi-allelic.
    :return: List[List[int]]
    """
    result: List[List[int]] = []
    prev_pos = -1
    for i in range(grg.num_mutations):
        mut = grg.get_mutation_by_id(i)
        if mut.position == prev_pos:
            result[-1].append(i)
        else:
            if result and len(result[-1]) == 1:
                result.pop()
            result.append([i])
            prev_pos = mut.position
    if result and len(result[-1]) == 1:
        result.pop()
    return result


def site_alleles(
    grg: pygrgl.GRG,
    alt_only: bool = False,
    mut_ids: List[int] = [],
) -> numpy.typing.NDArray:
    """
    Compute the number of alleles at the site associated with each mutation (variant).
    For example, if there is a site with 3 variants A>T, A>G, A>C, then each of those
    variants (mutations) will have a "4" in their result. Each variant is always bi-allelic,
    but the site it is associated can have an arbitrary number of alleles. This function
    counts the number of distinct REF alleles, so the result is count(REF) + count(ALT).

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param alt_only: Only count ALT alleles, not REF alleles. Default: False.
    :type alt_only: bool
    :param mut_ids: Restrict to the MutationIDs given (e.g., for only looks at SNPs, etc.)
    :type mut_ids: List[int]
    :return: A numpy array of length num_mutations, containing a allele count for each mutation.
    :rtype: numpy.array
    """
    result = []
    allele_set = set()
    prev_pos = -1
    to_add = 0
    mut_id_list = range(grg.num_mutations) if not mut_ids else mut_ids
    for mut_id in mut_id_list:
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
    assert res_array.shape[0] == len(mut_id_list)
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
