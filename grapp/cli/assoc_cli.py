from grapp.assoc import (
    read_plink_covariates,
    read_pheno,
    linear_assoc_covar,
    linear_assoc_no_covar,
)
from grapp.cli.util import pandas_to_tsv
from grapp.grg_calculator import load_grg_calculator, GRGCalcInterface
from grapp.util.exceptions import UserInputError
from typing import Optional, List
import argparse
import concurrent.futures
import numpy
import os
import pandas
import sys


def add_options(subparser):
    subparser.add_argument(
        "grg_input",
        nargs="+",
        help="One or more input GRG file. If multiple are provided, the GWAS can be run in parallel (see --jobs)",
    )
    subparser.add_argument(
        "-p",
        "--phenotypes",
        help="The file containing the phenotypes. If no file is provided, random phenotype values are used.",
    )
    subparser.add_argument(
        "-b",
        "--binomial",
        action="store_true",
        help="Use the binomial approximation for SNP variance instead of sample variance.",
    )
    subparser.add_argument(
        "-c",
        "--covariates",
        help="Covariates file to load: plink format (.txt) or pandas dataframe tab-separated (.tsv)",
    )
    subparser.add_argument(
        "-o",
        "--out-file",
        default=None,
        help="Tab-separated output file (with header); exported Pandas DataFrame. Default: <grg_input>.assoc.tsv",
    )
    subparser.add_argument(
        "-r",
        "--regress-y",
        action="store_true",
        help="Use linear regression to regress the covariates out of Y only. By default, QR decomposition is used to adjust both X and Y for covariates.",
    )
    subparser.add_argument(
        "-s",
        "--standardize",
        action="store_true",
        help="Standardize the X and Y matrices prior to performing linear regression.",
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="Use this many threads when processing more than one GRG.",
    )


def do_single_gwas(
    grg: GRGCalcInterface,
    y: numpy.typing.NDArray,
    C: Optional[numpy.typing.NDArray],
    method: Optional[str],
    standardize: bool,
    binomial: bool,
):
    if not binomial and not grg.has_individual_coals:
        raise UserInputError(
            "GRG does not have individual coalescences; try --binomial to approximate the variance"
        )

    dist = "binomial" if binomial else "sample"
    if C is not None:
        assert method is not None
        gwas_df = linear_assoc_covar(
            grg, y, C, method=method, standardize=standardize, dist=dist
        )
    else:
        gwas_df = linear_assoc_no_covar(grg, y, standardize=standardize, dist=dist)
    return gwas_df


def run(args):
    grgs = [load_grg_calculator(g) for g in args.grg_input]
    num_indivs = grgs[0].num_individuals

    def notna(value):
        if value == "NA":
            return False
        try:
            return not numpy.isnan(value)
        except TypeError:
            return True

    def check_iid_order(iids: List[str], filetype: str):
        if not grgs[0].has_individual_ids:
            nonna = list(filter(notna, iids))
            if len(nonna) > 0:
                print(
                    f"WARNING! IIDs were provided in {filetype} file (e.g., {nonna[0]}), but GRG has no individual IDs to cross-check against.",
                    file=sys.stderr,
                )
        else:
            for i in range(grgs[0].num_individuals):
                assert iids[i] == "NA" or iids[i] == grgs[0].get_individual_id(
                    i
                ), f"The {i}th row of {filetype} file has an IID mismatch. Input file={iids[i]}, GRG={grgs[0].get_individual_id(0)}"

    if args.phenotypes is None:
        print("No phenotype provided; randomly generating phenotype values")
        y = numpy.random.standard_normal(num_indivs)
    else:
        y, iids = read_pheno(args.phenotypes, return_indivs=True, verbose=True)
        assert (
            len(y) == num_indivs
        ), f"Phenotype file had {len(y)} rows, expected {num_indivs}"
        check_iid_order(iids, "phenotype")

    if args.covariates is not None:
        if args.covariates.endswith(".txt"):
            C, iids = read_plink_covariates(
                args.covariates, return_indivs=True, verbose=True
            )
            assert (
                len(iids) == num_indivs
            ), "Read {len(iids)} covariates, but GRG has {num_indivs} individuals"
            check_iid_order(iids, "covariate")
        else:
            assert args.covariates.endswith(
                ".tsv"
            ), "Covariates filename must end in .txt (plink format) or .tsv (pandas dataframe)"
            C = pandas.read_csv(args.covariates, delimiter="\t").to_numpy()
        method = "regress" if args.regress_y else "QR"
    else:
        assert not args.regress_y, "--regress-y doesn't apply to non-covariate GWAS"
        method = None
        C = None

    if len(grgs) > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
        futures = [
            executor.submit(
                do_single_gwas, g, y, C, method, args.standardize, args.binomial
            )
            for g in grgs
        ]
        dataframes = [f.result() for f in futures]
        if args.out_file is None:
            for gf, df in zip(args.grg_input, dataframes):
                out_file = f"{os.path.basename(gf)}.assoc.tsv"
                pandas_to_tsv(out_file, df)
                print(f"Wrote results to {out_file}")
        else:
            # Add a column that indicates which file each result came from, and concat
            for gf, df in zip(args.grg_input, dataframes):
                df["GRG"] = [os.path.basename(gf)] * len(df)
            df = pandas.concat(dataframes)
            pandas_to_tsv(args.out_file, df)
            print(f"Wrote results to {args.out_file}")
    else:
        gwas_df = do_single_gwas(grgs[0], y, C, method, args.standardize, args.binomial)
        if args.out_file is None:
            args.out_file = f"{os.path.basename(args.grg_input[0])}.assoc.tsv"
        pandas_to_tsv(args.out_file, gwas_df)
        print(f"Wrote results to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
