from grapp.popgen.polarize import (
    polarize_grg,
    PolarizationStats,
    DEFAULT_BATCH_SIZE,
)
from typing import Tuple, Any
import argparse
import os
import pygrgl
import sys

try:
    import pyfaidx
except ImportError:
    pyfaidx = None


def load_fasta(path: str) -> Tuple[Any, str]:
    assert (
        pyfaidx is not None
    ), "Requires 'pyfaidx' module: 'pip install pyfaidx' or 'pip install grapp[popgen]'"
    fasta = pyfaidx.Fasta(path, as_raw=False, sequence_always_upper=True)
    contigs = list(fasta.keys())
    if len(contigs) != 1:
        raise ValueError(
            "FASTA must contain exactly one contig for grg polarize. "
            f"Found: {', '.join(contigs)}"
        )
    return fasta, contigs[0]


def polarize_grg_from_fasta(
    grg: pygrgl.MutableGRG,
    fasta_file: str,
    drop_if_no_match: bool = True,
    map_batch_size: int = DEFAULT_BATCH_SIZE,
) -> PolarizationStats:
    fasta, contig = load_fasta(fasta_file)
    ancestral_sequence = str(
        fasta[contig][:]
    ).upper()  # TODO: shouldn't this always be upper, because you passed sequence_always_upper=True?
    return polarize_grg(grg, ancestral_sequence, drop_if_no_match, map_batch_size)


def add_options(subparser):
    subparser.add_argument("grg_input", help="Input GRG file to polarize")
    subparser.add_argument(
        "fasta_file", help="FASTA containing (only) the ancestral sequence"
    )
    subparser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        required=True,
        help="Output GRG file",
    )
    subparser.add_argument(
        "--keep-no-match",
        action="store_true",
        help="Keep mutations with no matching allele in the FASTA (instead of dropping them)",
    )
    subparser.add_argument(
        "--map-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of flipped mutations to process per graph traversal; larger values use more RAM",
    )


def run(args):
    if not os.path.isfile(args.grg_input):
        print(f"Input GRG file does not exist: {args.grg_input}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(args.fasta_file):
        print(f"FASTA file does not exist: {args.fasta_file}", file=sys.stderr)
        sys.exit(2)

    grg = pygrgl.load_mutable_grg(args.grg_input, load_up_edges=True)
    if grg is None:
        print(f"Failed to load GRG: {args.grg_input}", file=sys.stderr)
        sys.exit(2)

    try:
        stats = polarize_grg_from_fasta(
            grg,
            args.fasta_file,
            drop_if_no_match=not args.keep_no_match,
            map_batch_size=args.map_batch_size,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)

    pygrgl.save_grg(grg, args.output_file)

    print("Polarization complete")
    print(f"  Total seen:           {stats.total_seen}")
    print(f"  Emitted:              {stats.emitted}")
    print(f"  Already polarized:    {stats.already_polarized}")
    print(f"  Swapped:              {stats.swapped}")
    print(f"  Inconsistent:         {stats.inconsistent}")
    print(f"  After alignment end:  {stats.after_alignment}")
    print(f"  Alignment mismatch:   {stats.no_alignment}")
    print(f"  Non-SNVs skipped:     {stats.non_snv_skipped}")
    print(f"  Missingness remapped: {stats.missing_remapped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()
    run(args)
