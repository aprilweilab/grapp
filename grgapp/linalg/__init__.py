"""
Linear algebra-related operations on GRG. These are typically "generic" operations that
could apply to many different types of analyses.
"""

from .ops_scipy import (
    SciPyXOperator as _SciPyXOperator,
    SciPyXTXOperator as _SciPyXTXOperator,
    SciPyStdXOperator as _SciPyStdXOperator,
    SciPyStdXTXOperator as _SciPyStdXTXOperator,
)
import pygrgl
import numpy
from scipy.sparse.linalg import eigs as _scipy_eigs
from enum import Enum
from grgapp.util import allele_frequencies
import pandas as pd


class MatrixSelection(Enum):
    X = 1  # The NxM genotype matrix
    XT = 2  # The MxN genotype matrix
    XTX = 3  # The MxM covariance or correlation matrix


def eigs(
    matrix: MatrixSelection,
    grg: pygrgl.GRG,
    first_k: int,
    standardized: bool = True,
    haploid: bool = False,
):
    """
    Get the first K eigen values and vectors from a GRG.

    :param matrix: Which matrix derived from the GRG should be used: the genotype matrix (MatrixSelection.X),
        the transposed genotype matrix (MatrixSelection.XT), or the covariance/correlation matrix (MatrixSelection.XTX).
    :type matrix: MatrixSelection
    :param grg: The GRG to operate on.
    :type grg: pygrgl.GRG
    :param first_k: The number of (largest) eigen values/vectors to retrieve.
    :type first_k: int
    :param standardized: Set to False to use the non-standardized matrix. Default: True.
    :type standardized: bool
    :param haploid: Set to True to use the haploid values (0,1) instead of diploid values (0,1,2).
    :type haploid: bool
    :return: (eigen_value, eigen_vectors) as defined by scipy.sparse.linalg.eigs
    """
    first_k = min(first_k, grg.num_mutations)
    freqs = allele_frequencies(grg)

    if matrix == MatrixSelection.X:
        if standardized:
            operator = _SciPyStdXOperator(
                grg, pygrgl.TraversalDirection.UP, freqs, haploid=haploid
            )
        else:
            operator = _SciPyXOperator(
                grg, pygrgl.TraversalDirection.UP, freqs, haploid=haploid
            )
    elif matrix == MatrixSelection.XT:
        if standardized:
            operator = _SciPyStdXOperator(
                grg, pygrgl.TraversalDirection.DOWN, freqs, haploid=haploid
            )
        else:
            operator = _SciPyXOperator(
                grg, pygrgl.TraversalDirection.DOWN, freqs, haploid=haploid
            )
    elif matrix == MatrixSelection.XTX:
        if standardized:
            operator = _SciPyStdXTXOperator(grg, freqs, haploid=haploid)
        else:
            operator = _SciPyXTXOperator(grg, freqs, haploid=haploid)
    eigen_values, eigen_vectors = _scipy_eigs(operator, k=first_k)
    return eigen_values, eigen_vectors


def PCs(grg: pygrgl.GRG, first_k: int, unitvar: bool = True, include_eig: bool = False):
    """
    Get the PCs for each sample corresponding to kth eigenvector from  A GRG
    """
    first_k = min(first_k, grg.num_mutations)
    freqs = allele_frequencies(grg)

    op = _SciPyStdXTXOperator(grg, freqs, haploid=False)

    eigen_values, eigen_vectors = _scipy_eigs(op, k=first_k)

    # Standardize all k eigenvectors at once: for later
    eigvects_f64 = eigen_vectors.real.astype(numpy.float64)
    PC_scores = _SciPyStdXOperator(
        grg, pygrgl.TraversalDirection.UP, freqs, haploid=False
    )._matmat(eigvects_f64)
    if unitvar:
        PC_scores = PC_scores / numpy.sqrt(eigen_values.real)[None, :]

    colnames = [f"PC{i+1}" for i in range(PC_scores.shape[1])]
    df = pd.DataFrame(PC_scores, columns=colnames)
    if include_eig:
        return df, eigen_values, eigen_vectors
    return df
