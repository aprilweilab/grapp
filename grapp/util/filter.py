"""
Functions for filtering data out of a GRG to create a new, smaller GRG.
"""

from multiprocessing import Pool
from typing import List, Tuple, Optional, Union, Callable
import os
import pygrgl

from grapp.util.simple import UserInputError, allele_frequencies


def grg_save_individuals(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    individual_ids: List[str],
    allow_extra: bool = False,
    verbose: bool = False,
):
    """
    Save a GRG, keeping only the individuals with the IDs given in the list.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The new GRG file to create.
    :type out_filename: str
    :param individual_ids: List of individual identifiers to be kept.
    :type individual_ids: List[str]
    :param allow_extra: When False, throw an exception if individual_ids contains
        any identifier not found in the GRG. Default: False.
    :type allow_extra: bool
    """
    if isinstance(grg_or_filename, str):
        grg = pygrgl.load_immutable_grg(grg_or_filename, load_up_edges=True)
    else:
        grg = grg_or_filename
    sample_nodes = []
    id_set = set(individual_ids)
    for i in range(grg.num_individuals):
        indiv = grg.get_individual_id(i)
        if indiv in id_set:
            base_sample = i * grg.ploidy
            sample_nodes.extend(list(range(base_sample, base_sample + grg.ploidy)))
            id_set.remove(indiv)
    if not allow_extra and id_set:
        raise UserInputError(
            f"Found individuals that were not in the GRG: {','.join(id_set)}"
        )
    if verbose:
        print(f"Keeping {len(sample_nodes)} haplotypes")
    pygrgl.save_subset(
        grg,
        out_filename,
        pygrgl.TraversalDirection.UP,
        sample_nodes,
    )


def grg_save_samples(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    sample_nodes: List[int],
    verbose: bool = False,
):
    """
    Save a GRG, keeping only the haploid samples corresponding to the NodeIDs
    (indexes) given. See grg_save_individuals() for a version that uses
    identifiers to more "safely" down sample a GRG dataset.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The new GRG file to create.
    :type out_filename: str
    :param sample_nodes: List of NodeIDs (indexes) for the haploid samples. If a
        GRG has N samples, then they are numbered 0...(N-1). The ordering matches
        the order of the input file that the GRG was constructed from.
    :type sample_nodes: List[str]
    """
    if isinstance(grg_or_filename, str):
        grg = pygrgl.load_immutable_grg(grg_or_filename, load_up_edges=True)
    else:
        grg = grg_or_filename
    if not all(map(grg.is_sample, sample_nodes)):
        raise UserInputError(
            "One or more input samples were invalid (not present in the GRG)"
        )
    if verbose:
        print(f"Keeping {len(sample_nodes)} haplotypes")
    pygrgl.save_subset(
        grg,
        out_filename,
        pygrgl.TraversalDirection.UP,
        sample_nodes,
    )


def grg_save_populations(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    populations: List[str],
    allow_extra: bool = False,
    verbose: bool = False,
):
    """
    Save a GRG, keeping only the samples with populations matching the given population list.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The new GRG file to create.
    :type out_filename: str
    :param populations: List of population names to be kept.
    :type populations: List[str]
    :param allow_extra: When False, throw an exception if populations contains
        any identifier not found in the GRG. Default: False.
    :type allow_extra: bool
    """
    if isinstance(grg_or_filename, str):
        grg = pygrgl.load_immutable_grg(grg_or_filename, load_up_edges=True)
    else:
        grg = grg_or_filename
    grg_pops = grg.get_populations()
    if not grg_pops:
        raise UserInputError(
            "Cannot filter by population when GRG has no population data"
        )
    pop_indices = set()
    for pop in populations:
        try:
            pop_indices.add(grg_pops.index(pop))
        except ValueError:
            if not allow_extra:
                raise UserInputError(
                    f"Population matching '{pop}' not found; use allow_extra to ignore"
                )
    keep_samples = []
    for i in range(grg.num_samples):
        if grg.get_population_id(i) in pop_indices:
            keep_samples.append(i)
    return grg_save_samples(grg, out_filename, keep_samples, verbose=verbose)


def grg_save_range(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    bp_range: Tuple[int, int],
):
    """
    Given a GRG filename or object, save a new GRG that contains only the Mutations in
    the given basepair range.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The filename of the to-be-created GRG.
    :type out_filename: str
    :param bp_range: A pair (lower, upper), where both are in units basepair, and the
        Mutations will be kept if lower <= Mutation.position < upper. I.e., lower is inclusive
        and upper is exclusive.
    :type bp_range: Tuple[int, int]
    """

    def keep_mut(grg: pygrgl.GRG, mut_id: int):
        position = grg.get_mutation_by_id(mut_id).position
        return position >= bp_range[0] and position < bp_range[1]

    return grg_save_mut_filter(grg_or_filename, out_filename, keep_mut, bp_range)


def grg_save_freq(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    freq_range: Tuple[float, float],
):
    """
    Given a GRG filename or object, save a new GRG that contains only the Mutations in
    the given frequency range.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The filename of the to-be-created GRG.
    :type out_filename: str
    :param freq_range: A pair (lower, upper), where the Mutations will be kept if
        lower <= frequency(Mutation) < upper. I.e., lower is inclusive and upper is exclusive.
    :type freq_range: Tuple[float, float]
    """

    freqs = None

    def keep_mut(grg: pygrgl.GRG, mut_id: int):
        nonlocal freqs
        if freqs is None:
            freqs = allele_frequencies(grg)
        return freqs[mut_id] >= freq_range[0] and freqs[mut_id] < freq_range[1]

    return grg_save_mut_filter(grg_or_filename, out_filename, keep_mut)


def grg_save_mut_filter(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    mut_filter: Callable[[pygrgl.GRG, int], bool],
    bp_range: Tuple[int, int] = (0, 0),
):
    """
    Given a GRG filename or object, save a new GRG that contains only the Mutations selected
    by the given filter function.

    :param grg_or_filename: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grg_or_filename: Union[pygrgl.GRG, str]
    :param out_filename: The filename of the to-be-created GRG.
    :type out_filename: str
    :param mut_filter: Callback (function) that takes a MutationID (int) as input and returns
        true if that mutation should be kept.
    :type mut_filter: Callable[[pygrgl.GRG, int], bool]
    """
    if isinstance(grg_or_filename, str):
        grg = pygrgl.load_immutable_grg(grg_or_filename, load_up_edges=False)
    else:
        grg = grg_or_filename

    seeds = list(filter(lambda m: mut_filter(grg, m), range(grg.num_mutations)))
    if not seeds:
        raise UserInputError(
            "No Mutations found matching range; cannot filter to an empty GRG."
        )
    pygrgl.save_subset(
        grg,
        out_filename,
        pygrgl.TraversalDirection.DOWN,
        seeds,
        bp_range=bp_range,
    )


def multi_grg_save_mut_filter(
    grgs_or_filenames: Union[List[pygrgl.GRG], List[str]],
    out_filenames: List[str],
    mut_filter: Callable[[pygrgl.GRG, int, int], bool],
):
    """
    Given a list of GRG filenames or GRG objects, save a new GRG for each that contains only the
    Mutations selected by the given filter function. The callback takes the GRG, the MutationID
    within that GRG, and the "cumulative MutationID" when considering all GRGs sequentially (e.g.
    the second GRG's mutations start counting right after the last MutationID of the first GRG).

    :param grgs_or_filenames: Either a pygrgl.GRG object, or the filename of a GRG.
    :type grgs_or_filenames: Union[List[pygrgl.GRG], List[str]]
    :param out_filenames: The list of filenames of the to-be-created GRGs.
    :type out_filename: List[str]
    :param mut_filter: Callback (function) that takes a MutationID (int) as input and returns
        true if that mutation should be kept.
    :type mut_filter: Callable[[pygrgl.GRG, int, int], bool]
    """
    assert len(grgs_or_filenames) > 0
    assert len(grgs_or_filenames) == len(
        out_filenames
    ), "Input and output lists must be equal length"

    if isinstance(grgs_or_filenames[0], str):
        grgs = [
            pygrgl.load_immutable_grg(f, load_up_edges=False) for f in grgs_or_filenames
        ]
    else:
        assert isinstance(grgs_or_filenames[0], pygrgl.GRG)
        grgs = grgs_or_filenames

    def filter_one(grg, seeds, out_filename):
        pygrgl.save_subset(
            grg,
            out_filename,
            pygrgl.TraversalDirection.DOWN,
            seeds,
        )

    prev_mut_id = 0
    for out, grg in zip(out_filenames, grgs):
        seeds = list(
            filter(
                lambda m: mut_filter(grg, m, m + prev_mut_id), range(grg.num_mutations)
            )
        )
        if not seeds:
            raise UserInputError(
                "No Mutations found matching range; cannot filter to an empty GRG."
            )
        filter_one(grg, seeds, out)
        prev_mut_id += grg.num_mutations


def split_by_ranges(
    grg_filename: str,
    ranges: List[Tuple[int, int]],
    jobs: int = 1,
    out_dir: Optional[str] = None,
) -> List[str]:
    """
    Split a GRG into multiple parts, spanning the list of basepair ranges given.

    :param grg_filename: The input GRG filename.
    :type grg_filename: str
    :param ranges: A list of (lower, upper) pairs, where lower and upper are in units
        basepair, and lower is inclusive while upper is exclusive.
    :type ranges: List[Tuple[int, int]]
    :param jobs: Number of processes/threads to use. Default: 1.
    :type jobs: int
    :param out_dir: Output directory to put the split pieces into. If None, then use the
        current working directory. Default: None.
    :type out_dir: Optional[str]
    :return: List of filenames for the resulting GRG files.
    :rtype: List[str]
    """
    basename = os.path.basename(grg_filename)
    arguments = []
    for r in ranges:
        out_filename = f"{basename}.range_{r[0]}_{r[1]}.grg"
        if out_dir:
            out_filename = os.path.join(out_dir, out_filename)
        arguments.append((grg_filename, out_filename, r))
    with Pool(jobs) as pool:
        pool.starmap(grg_save_range, arguments)
    return [t[1] for t in arguments]
