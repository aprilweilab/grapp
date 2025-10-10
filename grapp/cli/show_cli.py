import argparse
import pygrgl


def add_options(subparser: argparse.ArgumentParser):
    subparser.add_argument("grg_input", help="The input GRG file")
    whattoshow = subparser.add_mutually_exclusive_group()
    whattoshow.add_argument(
        "-S",
        "--individuals",
        action="store_true",
        help="Show the individual IDs.",
    )


def run(args):
    if args.individuals:
        grg = pygrgl.load_immutable_grg(args.grg_input, load_up_edges=False)
        if grg.has_individual_ids:
            print("\n".join(map(grg.get_individual_id, range(grg.num_individuals))))
