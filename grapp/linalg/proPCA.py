from grapp.linalg.ops_scipy import SciPyStdXOperator as _SciPyStdXOperator
from grapp.util.simple import allele_frequencies
from numpy.typing import NDArray
from typing import Tuple
import numpy as np
import pygrgl


def _EM(grg: pygrgl.GRG, C: NDArray, freqs: NDArray) -> NDArray:
    ###Compute E
    E = np.linalg.inv(C.T @ C) @ C.T
    ###Find stuff for standardization
    X = (
        _SciPyStdXOperator(grg, pygrgl.TraversalDirection.UP, freqs, haploid=False)
        ._matmat(E.T)
        .T
    )
    ###Compute M
    M = X.T @ np.linalg.inv(X @ X.T)
    ###Compute C
    C = _SciPyStdXOperator(
        grg, pygrgl.TraversalDirection.DOWN, freqs, haploid=False
    )._matmat(M)
    ###Repeat
    return C


def _compute_pcs_from_C(
    C: np.ndarray, grg: pygrgl.GRG, freqs: np.typing.NDArray, k_orig: int
) -> Tuple[NDArray, NDArray, NDArray]:
    # 1 Orthonormalize columns of C
    Q, R = np.linalg.qr(C, mode="reduced")

    # 2 Project into that basis
    B = _SciPyStdXOperator(grg, pygrgl.TraversalDirection.UP, freqs, False)._matmat(Q).T

    # 3 SVD of matrix B
    U, S, Vt = np.linalg.svd(B, full_matrices=False)

    # 4 k components
    U_k = U[:, :k_orig]
    evals = S[:k_orig]

    # 5 Eigenvectors in original space
    evecs = Q @ U_k

    # 5 Sample scores
    scores = Vt[:k_orig, :].T

    return scores, evals, evecs


def _get_change(C_new: np.ndarray, C_old: np.ndarray) -> float:
    """
    Compute the relative Frobenius‐norm change between C_new and C_old.
    """
    diff = C_new - C_old
    return float(np.linalg.norm(diff, "fro")) / (
        float(np.linalg.norm(C_old, "fro")) + 1e-12
    )


def get_pcs_propca(
    grg: pygrgl.GRG,
    k: int = 10,
    l: float = -1,
    g: float = 3,
    max_iterations: int = -1,
    convergence_lim: float = 0.005,
    verbose: bool = False,
):
    if verbose:

        def vlog(msg):
            print(msg)

    else:

        def vlog(msg):
            pass

    if max_iterations == -1:
        max_iterations = k + 2
    if l == -1:
        l = k

    freqs = allele_frequencies(grg)

    C0 = np.random.normal(loc=0, scale=1, size=(grg.num_mutations, 2 * k))
    for i in range(max_iterations):
        C = _EM(grg, C0, freqs)
        if convergence_lim != -1 and i % g == 0:
            difference = _get_change(C, C0)
            if difference <= convergence_lim:
                vlog(f"Converged after {i} iterations with delta {difference}")
                break
        C0 = C

    return _compute_pcs_from_C(C0, grg, freqs, k)
