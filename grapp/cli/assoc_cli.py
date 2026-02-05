from grapp.assoc import (
    read_plink_covariates,
    read_pheno,
    linear_assoc_covar,
    linear_assoc_no_covar,
)
from grapp.cli.util import pandas_to_tsv
import argparse
import numpy
import os
import pandas
import pygrgl


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "-p",
        "--phenotypes",
        help="The file containing the phenotypes. If no file is provided, random phenotype values are used.",
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


def run(args):
    g = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
    if args.phenotypes is None:
        print("No phenotype provided; randomly generating phenotype values")
        y = numpy.random.standard_normal(g.num_individuals)
    else:
        y = read_pheno(args.phenotypes)
        assert (
            len(y) == g.num_individuals
        ), f"Phenotype file had {len(y)} rows, expected {g.num_individuals}"

    if args.covariates is not None:
        if args.covariates.endswith(".txt"):
            C = read_plink_covariates(args.covariates)
        else:
            assert args.covariates.endswith(
                ".tsv"
            ), "Covariates filename must end in .txt (plink format) or .tsv (pandas dataframe)"
            C = pandas.read_csv(args.covariates, delimiter="\t").to_numpy()
        method = "regress" if args.regress_y else "QR"
        gwas_df = linear_assoc_covar(
            g, y, C, method=method, standardize=args.standardize
        )
    else:
        assert not args.regress_y, "--regress-y doesn't apply to non-covariate GWAS"
        gwas_df = linear_assoc_no_covar(g, y, standardize=args.standardize)

    if args.out_file is None:
        args.out_file = f"{os.path.basename(args.grg_input)}.assoc.tsv"

    pandas_to_tsv(args.out_file, gwas_df)
    print(f"Wrote results to {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
