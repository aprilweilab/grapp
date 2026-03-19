from grapp.linalg import (
    PCs,
)
from grapp.cli.util import pandas_to_tsv
import argparse
import numpy
import os
import pygrgl


def add_options(subparser):
    subparser.add_argument("grg_input", nargs="+", help="One or more input GRG files")
    subparser.add_argument(
        "-d",
        "--dimensions",
        default=10,
        type=int,
        help="The number of PCs to extract. Default: 10.",
    )
    subparser.add_argument(
        "-o",
        "--pcs-out",
        default=None,
        help='Output filename to write the PCs to. Default: "<grg_input>.pcs.tsv"',
    )
    subparser.add_argument(
        "--pro-pca",
        action="store_true",
        help="Use the ProPCA algorithm to compute principal components.",
    )
    subparser.add_argument(
        "--sample-window",
        type=int,
        default=1,
        help="If provided, defines a window width in base-pair. Within each window (starting at the"
        "Mutation with the lowest coordinate) randomly choose a single SNP. This SNP set is used for PCA.",
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs (threads) to use. Will only use up to the number of GRG files. Default: 1",
    )
    subparser.add_argument(
        "-e",
        "--save-eigs",
        default=None,
        help="Save the eigenvectors and eigenvalues to a file with the given prefix.",
    )


def run(args):
    grg_list = [
        pygrgl.load_immutable_grg(gf, load_up_edges=False) for gf in args.grg_input
    ]
    scores = PCs(
        grg_list,
        k=args.dimensions,
        use_pro_pca=args.pro_pca,
        sample_window=args.sample_window,
        threads=args.jobs,
        include_eig=(args.save_eigs is not None),
    )
    if args.save_eigs is not None:
        scores, eigen_values, eigen_vectors = scores
        val_out = args.save_eigs + ".vals.txt"
        numpy.savetxt(val_out, eigen_values)
        print(f"Wrote eigenvalues to {val_out}")
        vec_out = args.save_eigs + ".vecs.txt"
        numpy.savetxt(vec_out, eigen_vectors)
        print(f"Wrote eigenvectors to {vec_out}")

    if args.pcs_out is None:
        base = os.path.basename(args.grg_input[0])
        suffix = ".and_others" if len(args.grg_input) > 1 else ""
        args.pcs_out = f"{base}{suffix}.pcs.tsv"
    pandas_to_tsv(args.pcs_out, scores)
    print(f"Wrote PCs to {args.pcs_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
