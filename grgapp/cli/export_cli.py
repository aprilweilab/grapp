import argparse
import gzip
import os
import sys

from grgapp.util.igd import (
    export_igd,
    export_vcf,
)


def add_options(subparser):
    subparser.add_argument("grg_input", help="The input GRG file")
    fileout_group = subparser.add_mutually_exclusive_group()
    fileout_group.add_argument(
        "--igd",
        help="Export the entire dataset to the given IGD filename.",
    )
    fileout_group.add_argument(
        "--vcf",
        help="Export the entire dataset to the given VCF filename. "
        "Use '-' to write to stdout (and, e.g., pipe through bgzip). If the filename ends with .gz then"
        " the Python GZIP codec will be used (not bgzip). Otherwise, a plaintext VCF file will be created.",
    )
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
    subparser.add_argument(
        "--contig",
        default="unknown",
        help='Use the given contig name when exporting to VCF. Default: "unknown".',
    )


VERBOSE = True


def run(args):
    if args.igd is not None:
        assert args.force or not os.path.exists(
            args.igd
        ), f"{args.igd} already exists; remove it or use --force"
        export_igd(
            args.grg_input, args.igd, args.jobs, verbose=VERBOSE, temp_dir=args.temp_dir
        )
    if args.vcf is not None:
        assert (
            args.force or args.vcf == "-" or not os.path.exists(args.vcf)
        ), f"{args.vcf} already exists; remove it or use --force"
        if args.vcf == "-":
            export_vcf(
                args.grg_input,
                sys.stdout,
                jobs=args.jobs,
                verbose=VERBOSE,
                temp_dir=args.temp_dir,
                contig=args.contig,
            )
        elif args.vcf.endswith(".gz"):
            with gzip.open(args.vcf, "wt") as fgz:
                export_vcf(
                    args.grg_input,
                    fgz,
                    jobs=args.jobs,
                    verbose=VERBOSE,
                    temp_dir=args.temp_dir,
                    contig=args.contig,
                )
        else:
            with open(args.vcf, "w") as ftext:
                export_vcf(
                    args.grg_input,
                    ftext,
                    jobs=args.jobs,
                    verbose=VERBOSE,
                    temp_dir=args.temp_dir,
                    contig=args.contig,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
