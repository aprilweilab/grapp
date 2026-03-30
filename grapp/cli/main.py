from grapp.cli import assoc_cli
from grapp.cli import export_cli
from grapp.cli import filter_cli
from grapp.cli import pca_cli
from grapp.cli import show_cli
from grapp.cli import pheno_cli
from grapp.util.exceptions import UserInputError

import argparse
import sys

CMD_ASSOC = "assoc"
CMD_PCA = "pca"
CMD_EXPORT = "export"
CMD_FILTER = "filter"
CMD_SHOW = "show"
CMD_PHENO = "pheno"


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
    export_parser = subparsers.add_parser(
        CMD_EXPORT,
        help="Export data from a GRG.",
    )
    export_cli.add_options(export_parser)
    filter_parser = subparsers.add_parser(
        CMD_FILTER,
        help="Export data from a GRG.",
    )
    filter_cli.add_options(filter_parser)
    show_parser = subparsers.add_parser(
        CMD_SHOW,
        help="Show data from a GRG.",
    )
    show_cli.add_options(show_parser)
    pheno_parser = subparsers.add_parser(
        CMD_PHENO,
        help="Simulate phenotypes with a GRG.",
    )
    pheno_cli.add_options(pheno_parser)

    try:
        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            exit(1)
        elif args.command == CMD_ASSOC:
            assoc_cli.run(args)
        elif args.command == CMD_PCA:
            pca_cli.run(args)
        elif args.command == CMD_EXPORT:
            export_cli.run(args)
        elif args.command == CMD_FILTER:
            filter_cli.run(args)
        elif args.command == CMD_SHOW:
            show_cli.run(args)
        elif args.command == CMD_PHENO:
            pheno_cli.run(args)
        else:
            print(f"Invalid command {args.command}", file=sys.stderr)
            parser.print_help()
            exit(1)
    except (FileNotFoundError, UserInputError) as e:
        print(f"Command failed with error: {str(e)}", file=sys.stderr)
        exit(2)


if __name__ == "__main__":
    main()
