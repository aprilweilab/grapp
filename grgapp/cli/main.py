from . import assoc_cli
from . import pca_cli

import argparse
import sys

CMD_ASSOC = "assoc"
CMD_PCA = "pca"


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    assoc_parser = subparsers.add_parser(
        CMD_ASSOC,
        help="Perform an association between genotypes and phenotypes.",
    )
    assoc_cli.add_options(assoc_parser)
    pca_parser = subparsers.add_parser(
        CMD_PCA,
        help="Extract principal components from the GRG dataset.",
    )
    pca_cli.add_options(pca_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit(1)
    elif args.command == CMD_ASSOC:
        assoc_cli.run(args)
    elif args.command == CMD_PCA:
        pca_cli.run(args)
    else:
        print(f"Invalid command {args.command}", file=sys.stderr)
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
