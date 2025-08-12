from grapp.assoc import (
    read_covariates_matrix,
    read_pheno,
    linear_assoc_covar,
    linear_assoc_no_covar,
)
import argparse
import numpy
import os
import pygrgl


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "-p",
        "--phenotypes",
        help="The file containing the phenotypes. If no file is provided, random phenotype values are used.",
    )
    subparser.add_argument("-c", "--covariates", help="Covariates text file to load")
    subparser.add_argument(
        "-o",
        "--out-file",
        default=None,
        help="Tab-separated output file (with header); exported Pandas DataFrame. Default: <grg_input>.assoc.tsv",
    )


def run(args):
    g = pygrgl.load_immutable_grg(args.grg_input)
    if args.phenotypes is None:
        y = numpy.random.standard_normal(g.num_individuals)
    else:
        y = read_pheno(args.phenotypes)
        assert (
            len(y) == g.num_individuals
        ), f"Phenotype file had {len(y)} rows, expected {g.num_individuals}"

    if args.covariates is not None:
        C = read_covariates_matrix(args.covariates, True)
        df = linear_assoc_covar(g, y, C)
    else:
        df = linear_assoc_no_covar(g, y)

    if args.out_file is None:
        args.out_file = f"{os.path.basename(args.grg_input)}.assoc.tsv"

    with open(args.out_file, "w") as fout:
        df.to_csv(fout, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
