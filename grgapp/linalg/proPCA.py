from ops_scipy import SciPyStdXOperator as _SciPyStdXOperator
from util.simple import allele_frequencies
import numpy as np
import pygrgl


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
    return float(np.linalg.norm(diff, "fro")) / (
        float(np.linalg.norm(C_old, "fro")) + 1e-12
    )


def main(
    grg,
    k: int = 10,
    l: float = -1,
    g: float = 3,
    max_iterations: int = -1,
    convergence_lim: float = -1,
):
    if max_iterations == -1:
        max_iterations = k + 2
    if l == -1:
        l = k

    freqs = allele_frequencies(grg)

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
