from grgapp.linalg import (
    PCs,
)
from .util import numpy_to_tsv
import argparse
import os
import pygrgl


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "-d",
        "--dimensions",
        default=10,
        help="The number of PCs to extract. Default: 10.",
    )
    subparser.add_argument(
        "-o",
        "--pcs-out",
        default=None,
        help='Output filename to write the PCs to. Default: "<grg_input>.pcs.tsv"',
    )


def run(args):
    grg = pygrgl.load_immutable_grg(args.grg_input)
    # TODO: options for normalization, etc.?
    scores = PCs(grg, args.dimensions)

    if args.pcs_out is None:
        args.pcs_out = f"{os.path.basename(args.grg_input)}.pcs.tsv"

    with open(args.pcs_out, "w") as fout:
        cols = [f"PC_{i}" for i in range(args.dimensions)]
        numpy_to_tsv(fout, scores, cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
