import numpy as np
import pandas as pd
from grgapp.linalg import eigs, PCs
import pygrgl


def compute_mev(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute MEV between two PC sets U and V.

    Parameters
    ----------
    U : array_like, shape (d, K)
        Columns are the estimated PCs.
    V : array_like, shape (d, K)
        Columns are the reference PCs.

    Returns
    -------
    mev : float
        Mean of explained variance.
    """

    if U.shape != V.shape:
        raise ValueError(
            f"U and V must have the same shape, got {U.shape} vs {V.shape}"
        )
    overlap = U.T @ V
    norms = np.linalg.norm(overlap, axis=0)
    return norms.mean()


plink = pd.read_csv(
    "/home/chris/GRGWAS-Cov/PCAExp/plink2.eigenvec", delim_whitespace=True, header=None
)
plink_pcs = plink.iloc[:, 2:].to_numpy()
U = np.loadtxt("/home/chris/GRGWAS-Cov/PCAExp/fastppca_projections.txt")
V = plink_pcs

grg = pygrgl.load_immutable_grg(
    "/home/chris/GRGWAS-Cov/simulation-source-10000-100000000.igd.final.grg"
)
PCs = PCs(grg, 10, True).to_numpy()
mev_value = compute_mev(V, PCs)
print(mev_value)
