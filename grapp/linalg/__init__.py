"""
Linear algebra-related operations on GRG. These are typically "generic" operations that
could apply to many different types of analyses.
"""

import numpy
import pandas as pd
import pygrgl
import random
import sys
import concurrent.futures
from enum import Enum
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Any, Union, Optional

from grapp.util import allele_frequencies as _allele_frequencies

# Everything below is imported so that users can import them via grapp.linalg
from grapp.linalg.ops_scipy import (  # noqa: F401
    SciPyXOperator,
    SciPyXTXOperator,
    SciPyXXTOperator,
    SciPyStdXOperator,
    SciPyStdXTXOperator,
    SciPyStdXXTOperator,
    MultiSciPyStdXOperator,
    MultiSciPyStdXTXOperator,
    MultiSciPyStdXXTOperator,
)
from grapp.linalg.proPCA import get_pcs_propca


class MatrixSelection(Enum):
    X = 1  # The NxM genotype matrix
    XT = 2  # The MxN genotype matrix
    XTX = 3  # The MxM covariance or correlation matrix


def _random_window_filter(grg: pygrgl.GRG, sample_window: int):
    # Create a mask for random mutation subset based on input argument.
    mutation_filter = []
    if sample_window > 1:
        mutation_window_bp = [
            grg.get_mutation_by_id(0).position,
        ]
        mutation_window_ids: List[List[int]] = [
            [],
        ]
        for mut_id in range(grg.num_mutations):
            mut = grg.get_mutation_by_id(mut_id)
            while mut.position >= (mutation_window_bp[-1] + sample_window):
                mutation_window_bp.append(mutation_window_bp[-1] + sample_window)
                mutation_window_ids.append([])
            mutation_window_ids[-1].append(mut_id)
        # Randomly choose a mutation ID from each window.
        for window_list in mutation_window_ids:
            if window_list:
                random.shuffle(window_list)
                mutation_filter.append(window_list[0])
    return mutation_filter


def sort_by_eigvalues(
    eigen_values: numpy.typing.NDArray, eigen_vectors: numpy.typing.NDArray
):
    """
    Reorder the eigen value and vector arrays so that they are in descending order of the corresponding
    eigen value.

    :param eigen_values: The vector of eigen values, of length k.
    :type eigen_values: numpy.typing.NDArray
    :param eigen_vectors: The matrix of eigen vectors, with k columns.
    :type eigen_vectors: numpy.typing.NDArray
    """
    ordered = numpy.flip(numpy.argsort(eigen_values))
    eigen_values[...] = eigen_values[ordered]
    eigen_vectors[...] = eigen_vectors[:, ordered]


def eigs(
    matrix: MatrixSelection,
    grg: pygrgl.GRG,
    k: int,
    standardized: bool = True,
    haploid: bool = False,
    op_kwargs: Dict[str, Any] = {},
) -> Tuple[NDArray, NDArray]:
    """
    Get the first K eigen values and vectors from a GRG.

    :param matrix: Which matrix derived from the GRG should be used: the genotype matrix (MatrixSelection.X),
        the transposed genotype matrix (MatrixSelection.XT), or the covariance/correlation matrix (MatrixSelection.XTX).
    :type matrix: MatrixSelection
    :param grg: The GRG to operate on.
    :type grg: pygrgl.GRG
    :param k: The number of (largest) eigen values/vectors to retrieve.
    :type k: int
    :param standardized: Set to False to use the non-standardized matrix. Default: True.
    :type standardized: bool
    :param haploid: Set to True to use the haploid values (0,1) instead of diploid values (0,1,2).
    :type haploid: bool
    :return: (eigen_value, eigen_vectors) as defined by scipy.sparse.linalg.eigs
    """
    k = min(k, grg.num_mutations)
    freqs = _allele_frequencies(grg)

    if matrix == MatrixSelection.X:
        if standardized:
            operator = SciPyStdXOperator(
                grg,
                pygrgl.TraversalDirection.UP,
                freqs,
                haploid=haploid,
                **op_kwargs,
            )
        else:
            operator = SciPyXOperator(
                grg,
                pygrgl.TraversalDirection.UP,
                freqs,
                haploid=haploid,
                **op_kwargs,
            )
    elif matrix == MatrixSelection.XT:
        if standardized:
            operator = SciPyStdXOperator(
                grg,
                pygrgl.TraversalDirection.DOWN,
                freqs,
                haploid=haploid,
                **op_kwargs,
            )
        else:
            operator = SciPyXOperator(
                grg,
                pygrgl.TraversalDirection.DOWN,
                freqs,
                haploid=haploid,
                **op_kwargs,
            )
    elif matrix == MatrixSelection.XTX:
        if standardized:
            operator = SciPyStdXTXOperator(grg, freqs, haploid=haploid, **op_kwargs)
        else:
            operator = SciPyXTXOperator(grg, freqs, haploid=haploid, **op_kwargs)
    eigen_values, eigen_vectors = eigsh(operator, k=k, which="LM")
    sort_by_eigvalues(eigen_values, eigen_vectors)
    return eigen_values, eigen_vectors


def get_eig_pcs(
    grgs: Union[pygrgl.GRG, List[pygrgl.GRG]],
    k: int,
    op_kwargs: Dict[str, Any] = {},
    threads: int = 1,
    verbose: bool = True,
) -> Tuple[NDArray, NDArray]:
    """
    Get the principal components for each sample corresponding to the first :math:`k` eigenvectors from a GRG,
    using an iterative eigenvector decomposition method.

    :param grgs: The GRG or list of GRGs to perform PCA on.
    :type grgs: Union[pygrgl.GRG, List[pygrgl.GRG]]
    :param k: The number of eigenvectors/values to use. These correspond to the `k` largest
        eigenvalues.
    :type k: int
    :param op_kwargs: A dictionary of keyword arguments to pass to the underlying SciPyStdXTXOperator.
    :type op_kwargs: Dict[str, Any]
    :param threads: Maximum number of threads to use. At most len(grgs) tasks can be done in parallel.
    :type threads: int
    :param verbose: Emit information on stderr.
    :type verboose: bool
    :return: A pair (PC_scores, eigen_values) where each is a numpy array.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    freqs: Union[List[numpy.typing.NDArray], numpy.typing.NDArray]
    if isinstance(grgs, list):
        executor = concurrent.futures.ThreadPoolExecutor(threads)
        futures = [executor.submit(_allele_frequencies, grg) for grg in grgs]
        freqs = [f.result() for f in futures]
        op = MultiSciPyStdXXTOperator(
            grgs, freqs, haploid=False, threads=threads, **op_kwargs
        )
    else:
        freqs = _allele_frequencies(grgs, adjust_missing=True)
        op = SciPyStdXXTOperator(grgs, freqs, haploid=False, **op_kwargs)
    if verbose:
        print(f"Running eigen decomposition on {op.shape[0]} variants")

    eigen_values, eigen_vectors = eigsh(op, k=k, which="LM")
    sort_by_eigvalues(eigen_values, eigen_vectors)
    assert eigen_vectors.real.dtype == numpy.float64
    return eigen_vectors, eigen_values


def PCs(
    grgs: Union[pygrgl.GRG, List[pygrgl.GRG]],
    k: int,
    include_eig: bool = False,
    use_pro_pca: bool = False,
    sample_window: int = 1,
    threads: int = 1,
):
    """
    Get the principal components for each sample corresponding to the first :math:`k` eigenvectors from a GRG.

    :param grgs: The GRG or list of GRGs to perform PCA on.
    :type grgs: Union[pygrgl.GRG, List[pygrgl.GRG]]
    :param k: The number of eigenvectors/values to use. These correspond to the `k` largest
        eigenvalues.
    :type k: int
    :param include_eig: When True, the return value is a triple of (DataFrame, EigenValues, EigenVectors),
        where the eigen values are as returned by scipy.sparse.linalg.eigsh(), and the eigenvalues are only provided when use_pro_pca=True. Default: False.
    :type include_eig: bool
    :param sample_window: If provided, defines a window width in base-pair. Within each window (starting at
        the Mutation with the lowest coordinate) randomly choose a single SNP and use that for performing
        PCA. Default: 1 (use every SNP).
    :type sample_window: Optional[int]
    :param threads: Number of threads to use. Will never use more than the number of input GRGs.
        Default: 1.
    :type threads: int
    :return: A pandas.DataFrame with a row per individual and a column per principal component. Or, if include_eig
        then a triple (dataframe, eigen values, eigen vectors), where eigen vectors are None unless use_pro_pca
        was True.
    :rtype: Union[pandas.DataFrame, Tuple[pandas.DataFrame, numpy.array, Optional[numpy.array]]]
    """
    grg_list = grgs if isinstance(grgs, list) else [grgs]
    grgs = None

    # Create a mask for random mutation subset based on input argument.
    total_muts = 0
    mutation_filter: Optional[List[int]] = []
    for grg in grg_list:
        assert mutation_filter is not None  # mypy is really dumb sometimes
        mutation_filter.extend(_random_window_filter(grg, sample_window))
        total_muts += grg.num_mutations
    if mutation_filter:
        print(
            f"Using {len(mutation_filter)} / {total_muts} mutations",
            file=sys.stderr,
        )
        k = min(k, len(mutation_filter))
    else:
        mutation_filter = None
        k = min(k, total_muts)

    if use_pro_pca:
        PC_scores, eigen_values, eigen_vectors = get_pcs_propca(
            grg_list, k, threads=threads, op_kwargs={"mutation_filter": mutation_filter}
        )
    else:
        PC_scores, eigen_values = get_eig_pcs(
            grg_list, k, threads=threads, op_kwargs={"mutation_filter": mutation_filter}
        )
        eigen_vectors = None

    colnames = [f"PC{i+1}" for i in range(PC_scores.shape[1])]
    df = pd.DataFrame(PC_scores, columns=colnames, copy=False)
    df.index.name = "Individual"
    if include_eig:
        return df, eigen_values, eigen_vectors
    return df
