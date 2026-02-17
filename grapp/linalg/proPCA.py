from scipy.sparse.linalg import LinearOperator
from grapp.linalg.ops_scipy import (
    SciPyStdXOperator as _SciPyStdXOperator,
    MultiSciPyStdXOperator as _MultiSciPyStdXOperator,
)
from grapp.util.simple import allele_frequencies
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Union, List
import numpy as np
import pygrgl
import concurrent.futures


def _EM(C: NDArray, operator: LinearOperator) -> Tuple[NDArray, NDArray]:
    ###Compute E
    E = np.linalg.inv(C.T @ C) @ C.T
    ###Find stuff for standardization
    X = (operator @ E.T).T
    ###Compute M
    M = X.T @ np.linalg.inv(X @ X.T)
    ###Compute C
    C = operator.T @ M
    ###Repeat
    return C, X


def _compute_pcs_from_C(
    C: np.ndarray, operator: LinearOperator, k_orig: int
) -> Tuple[NDArray, NDArray, NDArray]:
    # 1 Orthonormalize columns of C
    Q, R = np.linalg.qr(C, mode="reduced")

    # 2 Project into that basis
    B = (operator @ Q).T

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
    grgs: Union[pygrgl.GRG, List[pygrgl.GRG]],
    k: int = 10,
    l: float = -1,
    g: float = 3,
    max_iterations: int = -1,
    convergence_lim: float = 0.005,
    verbose: bool = False,
    threads: int = 1,
    op_kwargs: Dict[str, Any] = {},
    return_approx: bool = False,
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

    grg_list = grgs if isinstance(grgs, list) else [grgs]
    if len(grg_list) == 1:
        grg_freqs = allele_frequencies(grg_list[0])
        std_operator = _SciPyStdXOperator(
            grg_list[0], pygrgl.TraversalDirection.UP, grg_freqs
        )
    else:
        executor = concurrent.futures.ThreadPoolExecutor(threads)
        futures = [executor.submit(allele_frequencies, grg) for grg in grgs]
        all_freqs = [f.result() for f in futures]
        std_operator = _MultiSciPyStdXOperator(
            grg_list,
            pygrgl.TraversalDirection.UP,
            all_freqs,
            threads=threads,
            **op_kwargs,
        )

    X = np.ones((2 * k, std_operator.shape[1]))
    C0 = np.random.normal(loc=0, scale=1, size=(std_operator.shape[1], 2 * k))
    for i in range(max_iterations):
        C, X = _EM(C0, std_operator)
        if convergence_lim != -1 and i % g == 0:
            difference = _get_change(C, C0)
            if difference <= convergence_lim:
                vlog(f"Converged after {i} iterations with delta {difference}")
                break
        C0 = C

    scores, evals, evecs = _compute_pcs_from_C(C0, std_operator, k)
    if return_approx:
        return scores, evals, evecs, C0, X
    return scores, evals, evecs
