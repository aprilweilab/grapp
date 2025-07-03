import argparse


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    subparser.add_argument(
        "--todo",
        help="(TODO) Add filtering options here (TODO)",
    )


def run(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
