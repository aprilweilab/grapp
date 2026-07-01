from grapp.linalg.ops_scipy import SciPyStdXOperator, SciPyXOperator
from grapp.util.simple import allele_counts, _GenotypeDist, common_mut_dataframe
from grapp.util.exceptions import UserInputError
from grapp.grg_calculator import (
    GRGCalcInterface as _GRGCalcInterface,
    _wrap_grg,
)
from scipy.stats import t as t_distribution
from typing import List, Optional, Union, Tuple
import itertools
import math
import numpy
import pandas
import pygrgl
import re
import sklearn.linear_model
import sys


def _div_or_default(a, b, d):
    """
    y = a / b, unless b_i is 0, then y_i will be set to 0.

    :param a: Numerator
    :param b: Denominator
    :param d: Default value for when denominator is 0.
    """
    result = numpy.full(a.shape, d)
    return numpy.divide(a, b, out=result, where=(b != 0))


def read_plink_covariates(
    covar_path: str, return_indivs: bool = False, verbose: bool = False
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, List[str]]]:
    """
    Reads a PLINK-style covariate file: Optional header line, FID and IID in first two columns,
    covariates in remaining columns. The first two columns (FID/IID) are ignored. Does not allow

    :param path: Path to the covariate file.
    :type path: str
    :param return_indivs: If True, return a list of individual identifiers (IIDs from the plink file).
    :type return_indivs: bool
    :param verbose: Emit warnings/information about the file if True. Default: True.
    :type verbose: bool
    :return: The covariates as a numpy array of shape (n_samples, n_covariates). If return_indivs is True, then also
        return a list of individual IDs (strings).
    :rtype: Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, List[str]]]
    """
    rows = []
    num_covar = 0
    covar_names = []
    iids = []
    with open(covar_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            if i == 0 and parts[0] == "FID":
                assert (
                    len(parts) > 1 and parts[1] == "IID"
                ), "Header line must include FID followed by IID."
                covar_names = parts[2:]
                num_covar = len(covar_names)
            else:
                cov_vals = [float(v) for v in parts[2:]]
                if num_covar == 0:
                    num_covar = len(cov_vals)
                assert num_covar == len(
                    cov_vals
                ), f"Inconsistent number of columns in covariates file (line: {i}, cols: {len(cov_vals)})"
                rows.append(cov_vals)
                iids.append(parts[1])
    assert num_covar > 0, "No covariates are present in the file (only two columns)."

    # stack into (n_samples × K)
    X = numpy.vstack(rows) if rows else numpy.empty((0, 0))

    if verbose:
        print(f"Loaded {num_covar} covariates for {len(rows)} individuals")
        if covar_names:
            print(f"Covariate names: {covar_names}")
    if return_indivs:
        return (X, iids)
    return X


def read_pheno(
    filename: str, return_indivs: bool = False, verbose: bool = True
) -> Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, List[str]]]:
    """
    Reads a PLINK/GCTA/GRG-style phenotype file and returns the phenotype vector. In all cases, the
    row order of the file must match the individual indexing order of the GRG.

    PLINK-style: Optional header "FID IID PHEN". The first column is the family ID (ignored by grapp),
    the second column is the IID (used for validation by grapp), and the third column in the numerical
    phenotype value. The IID order is checked against the GRG individual ID order, when IDs are present
    in the GRG (and also not "NA" in the plink input file).

    GRG-style: Optional header "person_id phenotypes". The first column is checked against the GRG individual
    ID order, when IDs are present in GRG (and also not "NA" in the phenotype file). The second column is
    the numerical phenotype value.

    If there is no header, the number of columns determines the type (3 = plink, 2 = GRG).

    In all cases, columns can be tab-separated or space separated.

    :param path: Path to the phenotype file.
    :type path: str
    :param return_indivs: If True, return a list of individual identifiers (IIDs from the plink file).
    :type return_indivs: bool
    :param verbose: Emit warnings/information about the file if True. Default: True.
    :type verbose: bool
    :return: A one-dimensional NumPy array of phenotype values.
    :rtype: Union[numpy.typing.NDArray, Tuple[numpy.typing.NDArray, List[str]]]
    """
    header_line = None
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find header line if it's there - this is helpful since PLINK allows for comments before the header
    for i, line in enumerate(lines):
        # Plink header
        if re.match(r"^(#?FID\s+#?IID|#?IID)", line.strip(), re.IGNORECASE):
            header_line = i
            break
        # grg_pheno_sim header
        if re.match(r"^((person_id|individual_id)\s+(phenotypes|phenotype))", line.strip(), re.IGNORECASE):
            header_line = i
            break

    # Read data starting from the header (if present)
    if header_line is not None:
        df = pandas.read_csv(
            filename,
            sep=r"\s+",
            skiprows=header_line,
            engine="python",
            skipinitialspace=True,
            na_filter=False,
        )
    else:
        df = pandas.read_csv(
            filename,
            sep=r"\s+",
            header=None,
            engine="python",
            skipinitialspace=True,
            na_filter=False,
        )

    # Check column count
    if df.shape[1] > 3:
        raise ValueError(f"Expected 1, 2, or 3 columns, but found {df.shape[1]}.")

    # Extract last column and make sure it's a number
    try:
        last_col = df.iloc[:, -1].astype(float).to_numpy()
    except ValueError:
        raise ValueError(
            "Last column should be phenotype, but it contains non-numeric values."
        )
    if verbose:
        print(
            f"Using the {df.shape[1]}th column (the last one) for phenotype ({df.columns[-1]}).",
            file=sys.stderr,
        )

    if return_indivs:
        if len(df.columns) >= 2:
            iids = df.iloc[:, -2].to_list()
        else:
            iids = ["NA"] * len(df)
        return (last_col, iids)

    return last_col


def _computeDiagXTX(
    grg: _GRGCalcInterface,
    dist: str,
    acount: numpy.typing.NDArray,
    n_j: numpy.typing.NDArray,
    afreq_haploid: numpy.typing.NDArray,
    standardize: bool,
    sample_filter: Optional[Union[List[int], numpy.typing.NDArray]] = None,
):
    # diag(X^T @ X): for any standardized matrix with N rows, the diagonal will just be N
    if standardize:
        XX = n_j
    else:
        # diag(X^T @ X): for the non-standardized genotype matrix, GRG has a special initialization
        # method "xtx" which uses coalescence information to compute the diagonal in a single pass
        if dist == _GenotypeDist.SAMPLE.value:
            if not grg.has_individual_coals:
                raise UserInputError(
                    "GRG does not have individual coalescences; try using the binomial approximation for variance"
                )

            XX = grg.matmul(
                numpy.ones((1, grg.num_samples), dtype=numpy.int32),
                pygrgl.TraversalDirection.UP,
                init="xtx",
            ).squeeze()
            # The xtx calculation above does not know about our missing phenotypes, so we need
            # to perform an adjustment. It was calculated for true_n samples, when we only have n_j
            # samples after masking. This is equivalent to using X^T*X to compute the sample variance
            # and then computing n_j*variance.
            if sample_filter is not None:
                true_n = n_j + (grg.num_individuals - len(sample_filter))
                XX = (XX * n_j) / true_n
        else:
            assert dist == _GenotypeDist.BINOMIAL.value
            XX = acount * (1 - afreq_haploid) + 2 * acount * afreq_haploid
    assert XX.shape == (grg.num_mutations,)
    return XX


def linear_assoc_no_covar(
    grg: Union[pygrgl.GRG, _GRGCalcInterface],
    Y: numpy.typing.NDArray,
    only_beta: bool = False,
    standardize: bool = False,
    dist: str = _GenotypeDist.SAMPLE.value,
) -> pandas.DataFrame:
    """
    Performs regression for each mutation without adjusting for covariates. Missing data is treated as the
    mean genotype value (allele frequency for the relevant variant).

    :param Y: Phenotype vector of shape (n_samples,), with missing values specified as NaN.
    :type Y: numpy.ndarray
    :param only_beta: If True, returns a DataFrame with only the BETA column.
    :type only_beta: bool
    :param standardize: If True, standardize X and Y (after adjusting for covariates).
    :type standardize: bool
    :param dist: How to compute the :math:`diag(X^T X)` term. Options are: "sample" (use individual coalescence
        information to compute sample mean and variance), "binomial" (assume the diploid data follows a binomial
        distribution, for mean and variance).  Default: "sample".
    :type dist: str
    :return: A DataFrame containing statistics for each mutation:
        - POS, ALT, COUNT, BETA, B0, SE, R2, T, and P.
    :rtype: pandas.DataFrame
    """
    PLOIDY = 2
    assert grg.ploidy == PLOIDY, "GWAS is only supported on diploid individuals"
    assert _GenotypeDist.is_valid(dist), "Invalid dist= value provided"
    y_miss = numpy.isnan(Y)
    assert not numpy.all(y_miss), "Error: phenotype is all NaN (missing)"
    non_missing_indivs = numpy.flatnonzero(~y_miss).tolist()
    non_missing_samples = list(
        itertools.chain.from_iterable(
            map(lambda i: (2 * i, 2 * i + 1), non_missing_indivs)
        )
    )
    # Remove the missing individuals from the Y matrix.
    if any(y_miss):
        Y = Y[non_missing_indivs]
    grg = _wrap_grg(grg)

    acount, miss_count = allele_counts(
        grg,
        return_missing=True,
        sample_filter=non_missing_samples,
    )
    n = len(non_missing_indivs)
    n_j = n - (miss_count / PLOIDY)
    assert numpy.all(n_j >= 0.0)
    afreq_diploid = _div_or_default(acount, n_j, 0.0)
    afreq_haploid = afreq_diploid / PLOIDY
    if standardize:
        X_op = SciPyStdXOperator(
            grg,
            pygrgl.TraversalDirection.UP,
            afreq_haploid,
            haploid=False,
            sample_filter=non_missing_indivs,
        )
        x_mean = numpy.zeros(afreq_diploid.shape)
        nx_mean = numpy.zeros(acount.shape)
    else:
        X_op = SciPyXOperator(
            grg,
            pygrgl.TraversalDirection.UP,
            haploid=False,
            miss_values=afreq_haploid,
            sample_filter=non_missing_indivs,
        )
        x_mean = afreq_diploid  # 2*f_i
        nx_mean = acount  # 2*n*f_i

    mut_XY_count = Y @ X_op

    total_pheno = Y.sum()
    nodeXY = mut_XY_count - n_j * x_mean * (total_pheno / n)
    XX = _computeDiagXTX(
        grg,
        dist,
        acount,
        n_j,
        afreq_haploid,
        standardize,
        sample_filter=non_missing_indivs,
    )
    nodeXX = XX - nx_mean * x_mean
    beta = _div_or_default(nodeXY, nodeXX, math.nan)
    if only_beta:
        return pandas.DataFrame({"BETA": beta})

    b0 = total_pheno / n - beta * afreq_diploid

    yy = numpy.dot(Y, Y)
    sse = (
        yy
        - 2 * b0 * total_pheno
        - 2 * beta * mut_XY_count
        + n * b0**2
        + 2 * b0 * beta * acount
        + beta**2 * XX
    )

    se = numpy.sqrt(numpy.abs(sse / ((n - 2) * nodeXX)))
    t_stat = beta / se

    s_tot = yy - (total_pheno**2) / n
    r2 = 1 - sse / s_tot

    cdf_vals = t_distribution.cdf(t_stat, df=n - 2)
    p_val = 2 * numpy.where(t_stat > 0, 1 - cdf_vals, cdf_vals)

    return common_mut_dataframe(
        grg, COUNT=acount, BETA=beta, B0=b0, SE=se, R2=r2, T=t_stat, P=p_val
    )


def linear_assoc_covar(
    grg: Union[pygrgl.GRG, _GRGCalcInterface],
    Y: numpy.typing.NDArray,
    C: numpy.typing.NDArray,
    only_beta: bool = False,
    hide_covars: bool = True,
    standardize: bool = False,
    method: str = "QR",
    dist: str = _GenotypeDist.SAMPLE.value,
) -> pandas.DataFrame:
    """
    Performs regression for each mutation with covariate adjustment. Missing data is treated as the
    mean genotype value (allele frequency for the relevant variant).
    Uses QR decomposition to project out covariate effects from the phenotype and genotype vectors.

    :param Y: Phenotype vector of shape (n_samples,), with missing values specified as NaN.
    :type Y: numpy.ndarray
    :param C: Covariate matrix of shape (n_samples, n_covariates). Should include intercept.
    :type C: numpy.ndarray
    :param only_beta: If True, returns only the BETA column in the output.
    :type only_beta: bool
    :param hide_covars: If False, includes estimated covariate effects (GAMMA_i) in the output.
    :type hide_covars: bool
    :param standardize: If True, standardize X and Y (after adjusting for covariates).
    :type standardize: bool
    :param method: Either "QR" (default) or "regress". "QR" uses QR decomposition to adjust both :math:`X` and
        :math:`Y` for covariates (:math:`C`), but if standardize=True then it assumes that :math:`X` and :math:`C`
        are independent (the more correlated they are, the less "standardized" the result will be). "regress" uses
        linear regression between :math:`Y` and :math:`C` (to get :math:`B_c`), and then performs GWAS against
        :math:`Y'` (:math:`Y' = Y - C \\times B_c`).
    :type method: str
    :param dist: How to compute the :math:`diag(X^T X)` term. Options are: "sample" (use individual coalescence
        information to compute sample mean and variance), "binomial" (assume the diploid data follows a binomial
        distribution, for mean and variance).  Default: "sample".
    :type dist: str
    :return: A DataFrame containing at least BETA, SE, T, and P columns. If hide_covars is False, also includes
        GAMMA columns.
    :rtype: pandas.DataFrame
    """
    PLOIDY = 2
    assert grg.ploidy == PLOIDY, "GWAS is only supported on diploid individuals"
    assert method in ("QR", "regress"), 'Invalid "method" parameter'
    assert _GenotypeDist.is_valid(dist), "Invalid dist= value provided"

    y_miss = numpy.isnan(Y)
    assert not numpy.all(y_miss), "Error: phenotype is all NaN (missing)"
    non_missing_indivs = numpy.flatnonzero(~y_miss).tolist()
    non_missing_samples = list(
        itertools.chain.from_iterable(
            map(lambda i: (2 * i, 2 * i + 1), non_missing_indivs)
        )
    )
    # Remove the missing individuals from the Y and C matrices.
    if any(y_miss):
        Y = Y[non_missing_indivs]
        C = C[non_missing_indivs, :]

    if method == "regress":
        model = sklearn.linear_model.LinearRegression()
        regression = model.fit(C, Y)
        return linear_assoc_no_covar(
            grg,
            Y - C @ regression.coef_,
            only_beta=only_beta,
            dist=dist,
            standardize=standardize,
        )

    grg = _wrap_grg(grg)

    # QR decompose the covariate matrix after adding an intercept column.
    C_intercept = numpy.hstack([numpy.ones((C.shape[0], 1)), C])
    Q, R = numpy.linalg.qr(C_intercept)

    # Compute Y adj
    Yadj = Y - Q @ (Q.T @ Y)
    if standardize:
        Yadj = (Yadj - numpy.mean(Yadj, axis=0)) / numpy.std(Yadj, axis=0)

    acount, miss_count = allele_counts(
        grg, return_missing=True, sample_filter=non_missing_samples
    )
    n = len(non_missing_indivs)
    n_j = n - (miss_count / PLOIDY)
    assert numpy.all(n_j >= 0.0)
    afreq_diploid = _div_or_default(acount, n_j, 0.0)
    afreq_haploid = afreq_diploid / PLOIDY

    # We perform the multiplications in the same way, but with different operators depending
    # on whether we are using the standardized genotype matrix.
    if standardize:
        X_op = SciPyStdXOperator(
            grg,
            pygrgl.TraversalDirection.UP,
            afreq_haploid,
            haploid=False,
            sample_filter=non_missing_indivs,
        )
    else:
        X_op = SciPyXOperator(
            grg,
            pygrgl.TraversalDirection.UP,
            haploid=False,
            miss_values=afreq_haploid,
            sample_filter=non_missing_indivs,
        )

    # G^TQ
    XtQ = X_op.T @ Q

    # Diagonal of (X^TQ)(X^TQ)^T
    diagonal = (XtQ * XtQ).sum(axis=1)

    # Xadj^TXadj
    XX = _computeDiagXTX(
        grg,
        dist,
        acount,
        n_j,
        afreq_haploid,
        standardize,
        sample_filter=non_missing_indivs,
    )
    xadjTxadj = XX - diagonal

    # Compute (Xadj^TYadj)
    xadjTyadj = Yadj @ X_op

    beta = numpy.zeros(XX.size)
    beta = _div_or_default(xadjTyadj, xadjTxadj, math.nan)
    if only_beta:
        return pandas.DataFrame({"BETA": beta})

    df = Yadj.shape[0] - Q.shape[1] - 1
    YY = Yadj.T @ Yadj

    SSE = YY - _div_or_default(xadjTyadj**2, xadjTxadj, math.nan)
    se = numpy.sqrt(numpy.abs(_div_or_default(SSE, (df * xadjTxadj), math.nan)))
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

    df_data = {
        "COUNT": acount,
        "BETA": beta,
        "SE": se,
        "T": t_vals,
        "P": p,
    }
    df_data.update(gamma_cols)
    return common_mut_dataframe(grg, **df_data)
