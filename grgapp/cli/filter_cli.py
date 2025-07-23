import argparse
import os

from grgapp.util.filter import (
    grg_save_individuals,
    grg_save_range,
    grg_save_samples,
)
from grgapp.util.simple import UserInputError


def list_or_filename(arg_value):
    parts = arg_value.split(",")
    if len(parts) > 1:
        return list(map(str.strip, parts))
    if not os.path.isfile(arg_value):
        raise FileNotFoundError(f"File {arg_value} cannot be read")
    with open(arg_value) as f:
        return list(map(str.strip, f))


def int_list_or_filename(arg_value):
    try:
        return list(map(int, list_or_filename(arg_value)))
    except ValueError:
        raise UserInputError("One or more argument values were not integers")


def genome_range(arg_value):
    parts = arg_value.split("-")
    if len(parts) != 2:
        raise UserInputError(f"Invalid range specification: {arg_value}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise UserInputError(f"Range specification must use integers: {arg_value}")


def add_options(subparser: argparse.ArgumentParser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument("grg_output", help="The output GRG file")
    filters = subparser.add_mutually_exclusive_group(required=True)
    # TODO: update the pygrgl APIs to take a filename _or_ a file object, so that we can do stdout piping
    filters.add_argument(
        "-S",
        "--individuals",
        type=list_or_filename,
        help="Keep only the individuals with the IDs given as a comma-separated list or in the given filename.",
    )
    filters.add_argument(
        "--hap-samples",
        type=int_list_or_filename,
        help="Keep only the haploid samples with the NodeIDs (indexes) given as a comma-separated list or in the given filename.",
    )
    filters.add_argument(
        "-r",
        "--range",
        type=genome_range,
        help='Keep only the variants within the given range, in base pairs. Example: "lower-upper", where both are integers '
        "and lower is inclusive, upper is exclusive.",
    )


def run(args):
    if args.individuals:
        grg_save_individuals(args.grg_input, args.grg_output, args.individuals)
    elif args.hap_samples:
        grg_save_samples(args.grg_input, args.grg_output, args.hap_samples)
    elif args.range:
        grg_save_range(args.grg_input, args.grg_output, args.range)
    else:
        assert False, "Unreachable"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
