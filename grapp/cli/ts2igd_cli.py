import argparse
import pygrgl
import os

from grapp.util.igd import export_igd
from grapp.util.parallel import _get_temp_dir_context


def add_options(subparser):
    subparser.add_argument(
        "ts_input", help="The input tskit TreeSequence (.trees) file"
    )
    subparser.add_argument("igd_output", help="The output filename (.igd)")
    subparser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of the output file, if it exists.",
    )
    subparser.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of processes/threads to use, if possible. Default: 1.",
    )
    subparser.add_argument(
        "--temp-dir",
        help="Put all temporary files in the given directory, instead of creating a directory in "
        "the system temporary location. WARNING: Intermediate/temporary files will not be cleaned "
        "up when this is specified.",
    )


VERBOSE = True


def run(args):
    assert args.force or not os.path.exists(
        args.igd_output
    ), f"{args.igd_output} already exists; remove it or use --force"
    assert os.path.isfile(args.ts_input), f"{args.ts_input} does not exist"

    # Convert TS->GRG
    grg = pygrgl.grg_from_trees(args.ts_input)

    # Simplifying the GRG by writing it to disk and reloading it is worthwhile! It speeds up
    # the IGD conversion by quite a lot.
    with _get_temp_dir_context(args.temp_dir)() as tmpdirname:
        pygrgl.save_grg(grg, "simplify.grg")
        grg = pygrgl.load_immutable_grg("simplify.grg", load_up_edges=False)

    # Convert GRG->IGD
    export_igd(grg, args.igd_output, args.jobs, verbose=VERBOSE, temp_dir=args.temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
