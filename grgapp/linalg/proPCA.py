from ops_scipy import SciPyStdXOperator as _SciPyStdXOperator
import numpy as np
import pygrgl
from scipy.sparse.linalg import svds
import time
import pandas as pd
import cProfile, pstats, io
from pstats import SortKey


def grg2X(grg: pygrgl.GRG, diploid: bool = False):
    samples = grg.num_individuals if diploid else grg.num_samples
    ploidy = grg.ploidy
    result = np.zeros((samples, grg.num_mutations))
    muts_above = {}
    for node_id in reversed(range(grg.num_nodes)):
        muts = grg.get_mutations_for_node(node_id)
        ma = []
        if muts:
            ma.extend(muts)
        for parent_id in grg.get_up_edges(node_id):
            ma.extend(muts_above[parent_id])
        muts_above[node_id] = ma
        if grg.is_sample(node_id):
            indiv = node_id // ploidy
            for mut_id in muts_above[node_id]:
                if diploid:
                    result[indiv][mut_id] += 1
                else:
                    result[node_id][mut_id] = 1
    return result


def standardize_X(X: np.ndarray):
    """
    X: N×M diploid genotype matrix with entries in {0,1,2}.
    Returns:
      Xstd: N×M standardized matrix,
      freqs: length-M allele freqs f_i,
      sigma: length-M stddev sqrt(2 f_i (1-f_i))
    """
    N, M = X.shape
    # allele frequency per variant
    freqs = X.sum(axis=0) / (2 * N)
    # U is N×M each column = 2 f_i
    U = 2 * freqs
    # center
    Xc = X - U[None, :]
    # s_i = sqrt(2 f_i (1-f_i))
    sigma = np.sqrt(2 * freqs * (1 - freqs))
    # avoid division by zero
    zero_sigma = sigma == 0
    if np.any(zero_sigma):
        print(
            f"Warning: {zero_sigma.sum()} sites are monomorphic → will stay zero after std."
        )
        sigma[zero_sigma] = 1.0
    # standardize
    Xstd = Xc / sigma[None, :]
    # re-zero freq 1 cols
    Xstd[:, zero_sigma] = 0
    return Xstd


def EM_Matrix(Y, C):
    ###Compute E
    E = np.linalg.inv(C.T @ C) @ C.T
    print(E.shape)
    ###Find stuff for standardization
    X = E @ Y.T
    ###Compute M
    M = X.T @ np.linalg.inv(X @ X.T)
    ###Compute C
    C = Y.T @ M
    ###Repeat
    return C


def EM(grg: pygrgl.GRG, C: np.typing.NDArray, freqs: np.typing.NDArray):
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


def compute_pcs_from_C(
    C: np.ndarray, grg: pygrgl.GRG, freqs: np.typing.NDArray, k_orig: int
):
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

    return evecs, evals, scores


def get_change(C_new: np.ndarray, C_old: np.ndarray) -> float:
    """
    Compute the relative Frobenius‐norm change between C_new and C_old.
    """
    diff = C_new - C_old
    return np.linalg.norm(diff, "fro") / (np.linalg.norm(C_old, "fro") + 1e-12)


def main(
    grg,
    k: float = 10,
    l: float = -1,
    g: float = 3,
    max_iterations: float = -1,
    convergence_lim: float = -1,
):
    if max_iterations == -1:
        max_iterations = k + 2
    if l == -1:
        l = k

    freqs = pygrgl.matmul(
        grg,
        np.ones((1, grg.num_samples)),
        pygrgl.TraversalDirection.UP,
    )[0] / (grg.num_samples)

    C0 = np.random.normal(loc=0, scale=1, size=(grg.num_mutations, 2 * k))
    for i in range(max_iterations):
        C = EM(grg, C0, freqs)
        if convergence_lim != -1 and i % g == 0:
            difference = get_change(C, C0)
            print(difference)
            if difference <= convergence_lim:
                print(f"Converged after {i} iterations with delta {difference}")
                break
        C0 = C

    evec, eval, scores = compute_pcs_from_C(C0, grg, freqs, 10)

    print(evec[0])
    for i in range(10):
        print(scores[i])


if __name__ == "__main__":
    grg = pygrgl.load_immutable_grg(
        "/home/chris/GRGWAS-Cov/ALL.chr10.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.igd.final.grg"
    )
    main(grg, convergence_lim=0.005)
