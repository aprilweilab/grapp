from pygrgl import get_topo_order, TraversalDirection
import pygrgl
import numpy as np
import pandas as pd
import time
import math
import re
import cProfile, pstats, io

from pstats import SortKey
from scipy.stats import t as t_distribution


def read_covariates_matrix(
    covar_path: str, add_intercept: bool = True
) -> np.typing.NDArray:
    """
    Reads a PLINK-style covariate file (no headers) and returns a NumPy matrix of covariate values.
    The first two columns (FID/IID) are ignored. Optionally adds an intercept column of 1s.

    :param path: Path to the covariate file.
    :type path: str
    :param add_intercept: If True, adds a column of 1s to the left of the matrix.
    :type add_intercept: bool
    :return: A NumPy array of shape (n_samples, n_covariates [+1 if intercept]).
    :rtype: numpy.ndarray
    """
    rows = []
    with open(covar_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # skip FID/IID, keep covariates
            cov_vals = [float(v) for v in parts[2:]]
            rows.append(cov_vals)

    # stack into (n_samples × K)
    X = np.vstack(rows) if rows else np.empty((0, 0))

    if add_intercept and X.size:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack([intercept, X])

    return X


def read_pheno(filename: str):
    """
    Reads a PLINK/GCTA/GRG-style phenotype file and returns the phenotype vector.

    :param path: Path to the phenotype file.
    :type path: str
    :return: A one-dimensional NumPy array of phenotype values.
    :rtype: numpy.ndarray
    """
    header_line = None
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find header line if it's there - this is helpful since PLINK allows for comments before the header
    for i, line in enumerate(lines):
        if re.match(r"^(#?FID\s+#?IID|#?IID)", line.strip(), re.IGNORECASE):
            header_line = i
            break

    # Read data starting from the header (if present)
    if header_line is not None:
        df = pd.read_csv(filename, sep=r"\s+", skiprows=header_line, engine="python")
    else:
        df = pd.read_csv(filename, sep=r"\s+", header=None, engine="python")

    # Check column count
    if df.shape[1] not in (2, 3):
        raise ValueError(f"Expected 2 or 3 columns, but found {df.shape[1]}.")

    # Extract last column and make sure it's a number
    try:
        last_col = df.iloc[:, -1].astype(float).to_numpy()
    except ValueError:
        raise ValueError("Last column contains non-numeric values.")

    return last_col


def compute_XtX(g: pygrgl.GRG) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Computes X^T X and allele frequencies from the GRG via graph traversal

    :return: A tuple containing:
        - mut_XX (numpy.ndarray): Vector of mutation-specific X^T X values.
        - freq (numpy.ndarray): Vector of mutation-specific allele counts.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    # do by node
    num_nodes = g.num_nodes
    freq_map = np.zeros(num_nodes)
    node_XX_count = np.zeros(num_nodes)
    node_mean = np.zeros(num_nodes)
    num_samples = g.num_samples

    topo_nodes = list(get_topo_order(g, TraversalDirection.UP, g.get_sample_nodes()))

    for node_id in topo_nodes:
        curr_coals = g.get_num_individual_coals(node_id)
        assert (
            curr_coals != pygrgl.COAL_COUNT_NOT_SET
        ), "XtX only computable on diploid datasets that contain coalescence counts"
        assert (
            curr_coals <= num_samples / 2
        ), "Coalescence counts less than the number of diploid samples"

        # check if sample node
        if g.is_sample(node_id):
            node_XX_count[node_id] = 1
            freq_map[node_id] = 1

        else:
            count = 0
            mean_count = 0
            # check children node to accumulate count
            frequency = 0
            for child_id in g.get_down_edges(node_id):
                count += node_XX_count[child_id]
                mean_count += node_mean[child_id]
                frequency += freq_map[child_id]

            node_XX_count[node_id] = count + 2 * curr_coals
            node_mean[node_id] = mean_count
            freq_map[node_id] = frequency

    mut_XX_count = np.zeros(g.num_mutations)
    freq_count = np.zeros(g.num_mutations)

    # extend to mutations
    mut_pairs = g.get_mutation_node_pairs()
    for pair in mut_pairs:
        freq = 0
        node_id = pair[1]
        mut = pair[0]
        if node_id > num_nodes:
            mut_XX_count[mut] = 0
            freq_count[mut] = 0
        else:
            XX = node_XX_count[node_id]
            mut_XX_count[mut] = XX
            freq_count[mut] = freq_map[node_id]

    return mut_XX_count, freq_count


def linear_assoc_no_covar(
    grg: pygrgl.GRG, Y: np.typing.NDArray, only_beta: bool = False
) -> pd.DataFrame:
    """
    Performs regression for each mutation without adjusting for covariates.

    :param Y: Phenotype vector of shape (n_samples,).
    :type Y: numpy.ndarray
    :param only_beta: If True, returns a DataFrame with only the BETA column.
    :type only_beta: bool
    :return: A DataFrame containing statistics for each mutation:
        - POS, FREQ, BETA, B0, SE, R2, T, and P.
    :rtype: pandas.DataFrame
    """
    assert grg.ploidy == 2, "GWAS is only supported on diploid individuals"

    mut_XX_count, freq_count = compute_XtX(grg)
    n = grg.num_individuals

    y = np.repeat(Y, grg.ploidy)
    total_pheno = Y.sum()
    yy = np.dot(Y, Y)

    freq_count_norm = freq_count / n
    mut_XY_count = pygrgl.dot_product(grg, y, TraversalDirection.UP)

    # Vectorized regression components
    nodeXY = mut_XY_count - freq_count_norm * total_pheno
    nodeXX = mut_XX_count - freq_count * freq_count_norm
    beta = nodeXY / nodeXX

    if only_beta:
        return beta

    b0 = total_pheno / n - beta * freq_count_norm

    sse = (
        yy
        - 2 * b0 * total_pheno
        - 2 * beta * mut_XY_count
        + n * b0**2
        + 2 * b0 * beta * freq_count
        + beta**2 * mut_XX_count
    )

    se = np.sqrt(np.abs(sse / ((n - 2) * nodeXX)))
    t_stat = beta / se

    s_tot = yy - (total_pheno**2) / n
    r2 = 1 - sse / s_tot

    cdf_vals = t_distribution.cdf(t_stat, df=n - 2)
    p_val = 2 * np.where(t_stat > 0, 1 - cdf_vals, cdf_vals)

    positions = list(
        map(lambda i: grg.get_mutation_by_id(i).position, range(grg.num_mutations))
    )

    # Build DataFrame
    df = pd.DataFrame(
        {
            "POS": positions,
            "FREQ": freq_count,
            "BETA": beta,
            "B0": b0,
            "SE": se,
            "R2": r2,
            "T": t_stat,
            "P": p_val,
        }
    )

    return df


def linear_assoc_covar(
    grg: pygrgl.GRG,
    Y: np.typing.NDArray,
    C: np.typing.NDArray,
    only_beta: bool = False,
    hide_covars: bool = True,
) -> pd.DataFrame:
    """
    Performs regression for each mutation with covariate adjustment.
    Uses QR decomposition to project out covariate effects from the phenotype and genotype vectors.

    :param Y: Phenotype vector of shape (n_samples,).
    :type Y: numpy.ndarray
    :param C: Covariate matrix of shape (n_samples, n_covariates).
              Should include intercept.
    :type C: numpy.ndarray
    :param only_beta: If True, returns only the BETA column in the output.
    :type only_beta: bool
    :param hide_covars: If False, includes estimated covariate effects (GAMMA_i) in the output.
    :type hide_covars: bool
    :return: A DataFrame containing at least BETA, SE, T, and P columns.
             If hide_covars is False, also includes GAMMA columns.
    :rtype: pandas.DataFrame
    """
    assert grg.ploidy == 2, "GWAS is only supported on diploid individuals"

    Q, R = np.linalg.qr(C)

    # Compute Y adj
    Yadj = Y - Q @ (Q.T @ Y)
    # For haploid matrix
    Yadj2 = np.repeat(Yadj, grg.ploidy)

    # X^TX
    t0 = time.perf_counter()
    mut_XX_count, freq_count = compute_XtX(grg)
    t1 = time.perf_counter()

    print("Time: ", t1 - t0)

    beta = np.zeros(mut_XX_count.size)

    # G^TQ
    Q_hap = np.repeat(Q, grg.ploidy, axis=0)
    ###Computes G^TQ where Q's rows are duplicated so we can get X^TQ
    XtQ = pygrgl.matmul(grg, Q_hap.T, TraversalDirection.UP).T

    # Diagonal of (X^TQ)(X^TQ)^T
    diagonal = (XtQ * XtQ).sum(axis=1)

    # Xadj^TXadj
    xadjTxadj = mut_XX_count - diagonal
    # Compute (Xadj^TYadj)
    xadjTyadj = pygrgl.dot_product(grg, Yadj2, TraversalDirection.UP)

    if only_beta:
        for i in range(mut_XX_count.size):
            beta[i] = xadjTyadj[i] / xadjTxadj[i]
        return pd.DataFrame({"BETA": beta})

    if not hide_covars:
        QtY = Q.T @ Y
        gamma0 = np.linalg.solve(R, QtY)
        gammas = np.zeros((mut_XX_count.size, Q.shape[1]))

    df = Yadj.shape[0] - Q.shape[1] - 1
    YY = Yadj.T @ Yadj

    beta = xadjTyadj / xadjTxadj
    SSE = YY - (xadjTyadj**2) / xadjTxadj
    se = np.sqrt(np.abs(SSE / (df * xadjTxadj)))
    t_vals = beta / se

    cdf_vals = t_distribution.cdf(t_vals, df)
    p = 2 * np.where(t_vals > 0, 1 - cdf_vals, cdf_vals)

    # Optional GAMMA calculation
    gamma_cols = {}
    if not hide_covars:
        QtY = Q.T @ Y
        gamma0 = np.linalg.solve(R, QtY)
        corrections = np.linalg.solve(R, XtQ.T).T  # (num_snps, num_covars)
        gammas = gamma0 - beta[:, None] * corrections
        for j in range(Q.shape[1]):
            gamma_cols[f"GAMMA_{j}"] = gammas[:, j]

    # Build output DataFrame
    df_data = {
        "BETA": beta,
        "SE": se,
        "T": t_vals,
        "P": p,
    }
    df_data.update(gamma_cols)
    return pd.DataFrame(df_data)


# testing_covar = True
# if testing_covar:
#     y = read_pheno("/home/chris/GRGWAS-Cov/PLINK_Assitance/p100000.txt")
#     g = pygrgl.load_immutable_grg("/home/chris/GRGWAS-Cov/Test_GRG's/simulation-source-100000-100000000.igd.final.grg")
#     C = read_covariates_matrix("/home/chris/GRGWAS-Cov/PLINK_Assitance/c100000-5.txt", True)

#     pr = cProfile.Profile()
#     pr.enable()
#     df = linear_assoc_covar(g,y,C)
#     pr.disable()
#     s = io.StringIO()
#     sortby = SortKey.CUMULATIVE
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print(s.getvalue())
#     print(df.head(100))

# else:
#     y = read_pheno("/home/chris/GRGWAS-Cov/original_files/phenotypes.txt")
#     g = pygrgl.load_immutable_grg("/home/chris/GRGWAS-Cov/test-200-samples.vcf.gz.final.grg")
#     df = linear_assoc_no_covar(g,y)
#     print(df.head(100))
