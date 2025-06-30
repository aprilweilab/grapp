"""
Linear algebra-related operations on GRG. These are typically "generic" operations that
could apply to many different types of analyses.
"""

from .ops_scipy import SciPyStdXTXOperator as _SciPyStdXTXOperator
import pygrgl
import numpy
from scipy.sparse.linalg import eigs as _scipy_eigs


def eigs(grg: pygrgl.GRG, first_k: int):
    """
    Get the first K eigen values and vectors from a GRG.
    """
    first_k = min(first_k, grg.num_mutations)

    freqs = pygrgl.matmul(
        grg,
        numpy.ones((1, grg.num_samples)),
        pygrgl.TraversalDirection.UP,
    )[0] / (grg.num_samples)

    eigen_values, eigen_vectors = _scipy_eigs(
        _SciPyStdXTXOperator(grg, freqs, haploid=False), k=first_k
    )
    return eigen_values, eigen_vectors


def PCs(grg: pygrgl.GRG, first_k: int):
    """
    Get the PCs for each sample corresponding to kth eigenvector from  A GRG
    """
    first_k = min(first_k, grg.num_mutations)

    freqs = pygrgl.matmul(
        grg,
        numpy.ones((1, grg.num_samples)),
        pygrgl.TraversalDirection.UP,
    )[0] / (grg.num_samples)

    op = _SciPyStdXTXOperator(grg, freqs, haploid=False)

    eigen_values, eigen_vectors = _scipy_eigs(op, k=first_k)

    # Standardize all k eigenvectors at once: for later
    eigvects_f32 = eigen_vectors.real.astype(numpy.float32)
    V_std = eigvects_f32 / op.sigma_corrected[:, None]

    raw_pcs = pygrgl.matmul(
        grg, V_std.T, pygrgl.TraversalDirection.DOWN, by_individual=True
    ).T

    # Compute the “mean‐adjustment” constant for each PC:
    consts = numpy.sum(grg.ploidy * freqs[:, None] * V_std, axis=0)
    PC_scores = raw_pcs - consts[None, :]

    PC_unitvar = PC_scores / numpy.sqrt(eigen_values.real)[None, :]

    return PC_unitvar
