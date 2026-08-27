"""
Simple counts/statistics based on populations (stored in the GRG) or useful for
population genetics calculations.
"""

import itertools
import numpy
import pygrgl
from grapp.util.simple import (
    _div_or_default,
    multi_allelic_muts,
)
from typing import Tuple, List


def population_pairs(grg: pygrgl.GRG) -> List[Tuple[int, int]]:
    """
    Return the unique pairs of populations from the given GRG. Equivalent to
    ``itertools.combinations(reversed(range(P)), 2))`` where ``P`` is the number of
    populations.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: List of pairs of integer indices for populations.
    :rtype: List[Tuple[int, int]]
    """
    populations = grg.get_populations()
    P = len(populations)
    assert P >= 1, "No populations in the GRG; cannot get population pairs"
    return list(itertools.combinations(reversed(range(P)), 2))


def sample_pop_matrix(grg: pygrgl.GRG) -> numpy.typing.NDArray:
    """
    Given a GRG with :math:`P` populations and :math:`N` samples (haplotypes), return the
    :math:`P \\times N` matrix, where :math:`A_{i, j} = 1` indicates that sample :math:`j`
    is in population :math:`i`.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :return: :math:`P \\times N` matrix
    :rtype: numpy.ndarray
    """
    populations = grg.get_populations()
    P = len(populations)
    assert P >= 1, "No populations in the GRG; cannot get per-population allele counts"

    sample_pop = numpy.zeros((P, grg.num_samples), dtype=numpy.uint32)
    # Assigning values to numpy matrices is much more efficient this way than one-at-a-time.
    samples = list(range(grg.num_samples))
    pop_ids = list(map(lambda s: grg.get_population_id(s), samples))
    indices = (pop_ids, samples)
    sample_pop[indices] = 1
    return sample_pop


def pop_allele_counts(
    grg: pygrgl.GRG,
    impute_missing: bool = False,
    return_ref: bool = False,
) -> numpy.typing.NDArray:
    """
    Get the allele counts by population from a GRG that contains only bi-allelic variants.
    Each column ``j`` in the output matrix corresponds to the mutation with ID ``j``. Each
    row ``i`` in the output matrix corresponds to the allele counts for population ``i``,
    where the order matches ``GRG.get_populations()``.

    .. note::
        Output is directly compatible with `scikit-allel <https://scikit-allel.readthedocs.io/en/stable/index.html>`_
        popgen functions.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param impute_missing: For missing alleles, impute their values as the allele frequency
        in the population. This results in floating-point numbers for allele counts, so we
        round to the nearest integer in the result. Default: False.
    :type impute_missing: bool
    :param return_ref: If True, return a matrix of dimensions :math:`P \\times M \\times 2`
        that is compatible with the folded SFS functions from ``scikit-allel` (i.e., the last
        dimension is ref, alt counts).
    :type return_ref: bool
    :return: A matrix (numpy array) of shape :math:`P \\times M`, where :math:`P` is number
        of populations and :math:`M` is number of mutations in the GRG.
    :rtype: numpy.ndarray
    """
    ma = multi_allelic_muts(grg)
    assert (
        len(ma) == 0
    ), f"GRG must have only bi-allelic sites (found {len(ma)} multi-allelic); try the 'grapp filter' command."

    input_mat = sample_pop_matrix(grg)
    pop_samples = numpy.sum(input_mat, axis=1)
    if impute_missing:
        miss_counts = numpy.zeros(
            (input_mat.shape[0], grg.num_mutations), dtype=numpy.uint32
        )
    else:
        miss_counts = None
    acounts = pygrgl.matmul(
        grg, input_mat, pygrgl.TraversalDirection.UP, miss=miss_counts
    )
    if miss_counts is not None:
        assert miss_counts is not None
        acounts += numpy.round(
            miss_counts * _div_or_default(acounts, pop_samples, 0)
        ).astype(numpy.uint32)
    if return_ref:
        # P x M x 2 matrix
        return numpy.array([(numpy.array([pop_samples]).T - acounts).T, acounts.T]).T
    # P x M matrix
    return acounts


def allele_counts(
    grg: pygrgl.GRG,
    impute_missing: bool = False,
    return_ref: bool = False,
) -> numpy.typing.NDArray:
    """
    Get the allele counts for all samples from a GRG that contains only bi-allelic variants.
    Each column ``j`` in the output vector corresponds to the counts for mutation with ID ``j``.

    .. note::
        Output is directly compatible with `scikit-allel <https://scikit-allel.readthedocs.io/en/stable/index.html>`_
        popgen functions.

    :param grg: The GRG.
    :type grg: pygrgl.GRG
    :param impute_missing: For missing alleles, impute their values as the allele frequency
        in the population. This results in floating-point numbers for allele counts, so we
        round to the nearest integer in the result. Default: False.
    :type impute_missing: bool
    :param return_ref: If True, return a matrix of dimensions :math:`M \\times 2`
        that is compatible with the folded SFS functions from ``scikit-allel` (i.e., the last
        dimension is ref, alt counts).
    :type return_ref: bool
    :return: A vector (numpy array) of length :math:`M`, where :math:`M` is number of mutations
        in the GRG.
    :rtype: numpy.ndarray
    """
    ma = multi_allelic_muts(grg)
    assert (
        len(ma) == 0
    ), f"GRG must have only bi-allelic sites (found {len(ma)} multi-allelic); try the 'grapp filter' command."

    input_mat = numpy.ones((1, grg.num_samples), dtype=numpy.uint32)
    if impute_missing:
        miss_counts = numpy.zeros((1, grg.num_mutations), dtype=numpy.uint32)
    else:
        miss_counts = None
    acounts = pygrgl.matmul(
        grg, input_mat, pygrgl.TraversalDirection.UP, miss=miss_counts
    )[0]
    n_j = grg.num_samples
    if miss_counts is not None:
        assert miss_counts is not None
        n_j -= miss_counts[0]
        acounts += numpy.round(
            miss_counts[0] * _div_or_default(acounts, n_j, 0)
        ).astype(numpy.uint32)
    if return_ref:
        # M x 2 matrix
        return numpy.array([(n_j - acounts).T, acounts.T]).T
    # M-length vector
    return acounts
