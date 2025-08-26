from grapp.linalg import (
    PCs,
)
import argparse
import os
import pygrgl


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
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
        "--normalize",
        action="store_true",
        help="Normalize the PCs according to sqrt(eigenvalue) for each.",
    )
    subparser.add_argument(
        "--pro-pca",
        action="store_true",
        help="Use the ProPCA algorithm to compute principle components.",
    )


def run(args):
    grg = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
    scores = PCs(grg, args.dimensions, args.normalize, use_pro_pca=args.pro_pca)

    if args.pcs_out is None:
        args.pcs_out = f"{os.path.basename(args.grg_input)}.pcs.tsv"

    scores.to_csv(args.pcs_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
