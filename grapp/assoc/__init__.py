from scipy.stats import t as t_distribution
import numpy
import pandas
import pygrgl
import re
from grapp.util.simple import allele_counts, allele_frequencies
from grapp.linalg.ops_scipy import SciPyStdXOperator, SciPyXOperator


def read_covariates_matrix(
    covar_path: str, add_intercept: bool = True
) -> numpy.typing.NDArray:
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
    X = numpy.vstack(rows) if rows else numpy.empty((0, 0))

    if add_intercept and X.size:
        intercept = numpy.ones((X.shape[0], 1))
        X = numpy.hstack([intercept, X])

    return X


def read_pheno(filename: str) -> numpy.typing.NDArray:
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
        df = pandas.read_csv(
            filename, sep=r"\s+", skiprows=header_line, engine="python"
        )
    else:
        df = pandas.read_csv(filename, sep=r"\s+", header=None, engine="python")

    # Check column count
    if df.shape[1] not in (2, 3):
        raise ValueError(f"Expected 2 or 3 columns, but found {df.shape[1]}.")

    # Extract last column and make sure it's a number
    try:
        last_col = df.iloc[:, -1].astype(float).to_numpy()
    except ValueError:
        raise ValueError("Last column contains non-numeric values.")

    return last_col


def linear_assoc_no_covar(
    grg: pygrgl.GRG, Y: numpy.typing.NDArray, only_beta: bool = False
) -> pandas.DataFrame:
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

    with numpy.errstate(divide="ignore", invalid="ignore"):
        freq_count = allele_counts(grg)
        XX = pygrgl.matmul(
            grg,
            numpy.ones((1, grg.num_samples), dtype=numpy.int32),
            pygrgl.TraversalDirection.UP,
            init="xtx",
        ).squeeze()
        n = grg.num_individuals

        y = numpy.repeat(Y, grg.ploidy)
        total_pheno = Y.sum()
        yy = numpy.dot(Y, Y)

        freq_count_norm = freq_count / n
        mut_XY_count = pygrgl.dot_product(grg, y, pygrgl.TraversalDirection.UP)

        # Vectorized regression components
        nodeXY = mut_XY_count - freq_count_norm * total_pheno
        nodeXX = XX - freq_count * freq_count_norm
        beta = nodeXY / nodeXX
        if only_beta:
            return pandas.DataFrame({"BETA": beta})

        b0 = total_pheno / n - beta * freq_count_norm

        sse = (
            yy
            - 2 * b0 * total_pheno
            - 2 * beta * mut_XY_count
            + n * b0**2
            + 2 * b0 * beta * freq_count
            + beta**2 * XX
        )

        se = numpy.sqrt(numpy.abs(sse / ((n - 2) * nodeXX)))
        t_stat = beta / se

        s_tot = yy - (total_pheno**2) / n
        r2 = 1 - sse / s_tot

        cdf_vals = t_distribution.cdf(t_stat, df=n - 2)
        p_val = 2 * numpy.where(t_stat > 0, 1 - cdf_vals, cdf_vals)

        positions = list(
            map(lambda i: grg.get_mutation_by_id(i).position, range(grg.num_mutations))
        )

        # Build DataFrame
        df = pandas.DataFrame(
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
    Y: numpy.typing.NDArray,
    C: numpy.typing.NDArray,
    only_beta: bool = False,
    hide_covars: bool = True,
    standardize: bool = False,
) -> pandas.DataFrame:
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
    :param standardize: If True, standardize X and Y (after adjusting for covariates).
    :type standardize: bool
    :return: A DataFrame containing at least BETA, SE, T, and P columns.
            If hide_covars is False, also includes GAMMA columns.
    :rtype: pandas.DataFrame
    """
    assert grg.ploidy == 2, "GWAS is only supported on diploid individuals"

    with numpy.errstate(divide="ignore", invalid="ignore"):
        Q, R = numpy.linalg.qr(C)

        # Compute Y adj
        Yadj = Y - Q @ (Q.T @ Y)
        if standardize:
            Yadj = (Yadj - numpy.mean(Yadj, axis=0)) / numpy.std(Yadj, axis=0)

        # We perform the multiplications in the same way, but with different operators depending
        # on whether we are using the standardized genotype matrix.
        if standardize:
            # diag(X^T @ X): for any standardized matrix with N rows, the diagonal will just be N
            XX = numpy.full(grg.num_mutations, grg.num_individuals)

            freqs = allele_frequencies(grg)
            X_op = SciPyStdXOperator(
                grg, pygrgl.TraversalDirection.UP, freqs, haploid=False
            )
        else:
            # diag(X^T @ X): for the non-standardized genotype matrix, GRG has a special initialization
            # method "xtx" which uses coalescence information to compute the diagonal in a single pass
            XX = pygrgl.matmul(
                grg,
                numpy.ones((1, grg.num_samples), dtype=numpy.int32),
                pygrgl.TraversalDirection.UP,
                init="xtx",
            ).squeeze()

            X_op = SciPyXOperator(grg, pygrgl.TraversalDirection.UP, haploid=False)

        beta = numpy.zeros(XX.size)

        # G^TQ
        ###Computes G^TQ where Q's rows are duplicated so we can get X^TQ
        XtQ = X_op.T @ Q

        # Diagonal of (X^TQ)(X^TQ)^T
        diagonal = (XtQ * XtQ).sum(axis=1)

        # Xadj^TXadj
        xadjTxadj = XX - diagonal

        # Compute (Xadj^TYadj)
        xadjTyadj = Yadj @ X_op

        beta = xadjTyadj / xadjTxadj
        if only_beta:
            return pandas.DataFrame({"BETA": beta})

        df = Yadj.shape[0] - Q.shape[1] - 1
        YY = Yadj.T @ Yadj

        SSE = YY - (xadjTyadj**2) / xadjTxadj
        se = numpy.sqrt(numpy.abs(SSE / (df * xadjTxadj)))
        t_vals = beta / se

        cdf_vals = t_distribution.cdf(t_vals, df)
        p = 2 * numpy.where(t_vals > 0, 1 - cdf_vals, cdf_vals)

        # Optional GAMMA calculation
        gamma_cols = {}
        if not hide_covars:
            QtY = Q.T @ Y
            gamma0 = numpy.linalg.solve(R, QtY)
            corrections = numpy.linalg.solve(R, XtQ.T).T  # (num_snps, num_covars)
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
        return pandas.DataFrame(df_data)
