from typing import Union, Optional, List, Callable, TextIO
from multiprocessing import Pool
from pyigd.extra import (
    igd_merge,
    collect_next_site,
)
from contextlib import contextmanager
from grgapp.util.filter import split_by_ranges

import numpy
import os
import pygrgl
import pyigd
import shutil
import sys
import tempfile


# Helper that converts a single GRG into a single IGD
def _grg2igd(
    grg_or_file: Union[str, pygrgl.GRG], igd_prefix: str, batch_size: int
) -> str:
    if isinstance(grg_or_file, str):
        grg = pygrgl.load_immutable_grg(grg_or_file)
    else:
        grg = grg_or_file
    if igd_prefix.endswith(".igd"):
        igd_prefix = igd_prefix[:-4]
    igd_filename = (
        f"{igd_prefix}.{grg.specified_bp_range[0]}-{grg.specified_bp_range[1]}.igd"
    )

    with open(igd_filename, "wb") as fout:
        igd_writer = pyigd.IGDWriter(fout, grg.num_individuals, source="grg2igd")
        igd_writer.write_header()
        for start in range(0, grg.num_mutations, batch_size):
            end = min(start + batch_size, grg.num_mutations)
            rows = end - start
            # Using bool in GRG matrix multiplication is a little trick that causes the
            # GRG calculations to be performed bitwise, saving space and CPU time.
            DTYPE = bool
            # Construct the input matrix for the given batch, that looks like:
            #   0 0 0 0 0 1 0 0 0 0 0 0 0
            #   0 0 0 0 0 0 1 0 0 0 0 0 0
            #   0 0 0 0 0 0 0 1 0 0 0 0 0
            # where the square diagonal is the current batch that we are querying.
            x = numpy.zeros((rows, start), dtype=DTYPE)
            y = numpy.identity(rows, dtype=DTYPE)
            z = numpy.zeros((rows, grg.num_mutations - end), dtype=DTYPE)

            sample_matrix = pygrgl.matmul(
                grg,
                numpy.concatenate((x, y, z), axis=1),
                pygrgl.TraversalDirection.DOWN,
            )

            # Each row of the sample_matrix is the dense vector for the IGD variant.
            # So the sparse lists can be obtained by numpy.nonzero:
            for i, row in enumerate(sample_matrix):
                grg_mut = grg.get_mutation_by_id(start + i)
                sample_list = numpy.flatnonzero(row).tolist()
                igd_writer.write_variant(
                    grg_mut.position,
                    grg_mut.ref_allele,
                    grg_mut.allele,
                    sample_list,
                    False,
                    0,
                )

        igd_writer.write_index()
        igd_writer.write_variant_info()
        if grg.has_individual_ids:
            indiv_ids = [grg.get_individual_id(i) for i in range(grg.num_individuals)]
            igd_writer.write_individual_ids(indiv_ids)
        igd_writer.out.seek(0)
        igd_writer.write_header()
    return igd_filename


# Helper to make the usage of a user-specified directory and a temporary directory seemless.
def _get_temp_dir_context(temp_dir: Optional[str] = None) -> Callable:
    if temp_dir is None:
        return tempfile.TemporaryDirectory

    @contextmanager
    def existing_dir_context_mgr(*args, **kwargs):
        yield temp_dir

    return existing_dir_context_mgr


def export_igd(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_filename: str,
    jobs: int = 1,
    batch_size: Union[str, int] = "auto",
    temp_dir: Optional[str] = None,
    no_merge: bool = False,
    split_threshold: int = 5_000_000,
    verbose: bool = False,
):
    """
    Export a GRG to a phased IGD file, which is a sparse matrix representation of
    the same data. An IGD will almost always be larger than a GRG, but it can
    be useful because:
    1. The rows are variants, giving fast access to specific variants and their
       list of samples. Instead of having traverse many graph edges to get the
       sample list for a variant, you can just read the row from the IGD.
    2. Conversion to other standard formats is very fast, for example .vcf.gz

    :param grg_or_filename: The GRG to convert, either as a pygrgl.GRG or the
        filename of a GRG.
    :type grg: Union[pygrgl.GRG, str]
    :param out_filename: The IGD file to create. The path up to the filename must
        already exist, and the file itself must not exist.
    :type out_filename: str
    :param jobs: The number of parallel processes to use to do the conversion. The
        speed-up is essentially linear. Default: 1.
    :type jobs: int
    :param batch_size: The number of Mutations to process simultaneously, or the
        string "auto" if you want a reasonable value to be chosen for you.
    :type batch_size: Union[str, int]
    :param temp_dir: The directory to use for intermediate IGD files. The GRG
        is split into multiple pieces and placed in this directory, and then each
        piece gets converted to an IGD file, and then those IGD files are merged
        into the final result. If temp_dir is None, these files are placed in a
        temporary directory which is then deleted upon completion.
    :type temp_dir: Optional[str]
    :param no_merge: Set to True to get all the intermediate files, but not merge
        them into a final IGD. In this case, `out_filename` will not be created.
    :type no_merge: bool
    :param split_threshold: Basepair threshold for splitting the GRG into chunks
        for processing. A split GRG is much faster to operate on than a full sized
        GRG, plus this is how we parallelize the conversion. Default: 5MB.
    :type split_threshold: int
    """
    if verbose:

        def logv(msg):
            print(msg, file=sys.stderr)

    if batch_size == "auto":
        batch_size = 100  # TODO: improve this by measuring available RAM?
    assert not isinstance(batch_size, str)
    assert temp_dir is None or os.path.isdir(
        temp_dir
    ), f"Provided temp_dir {temp_dir} does not exist."

    with _get_temp_dir_context(temp_dir)() as tmpdirname:
        if isinstance(grg_or_filename, str):
            grg = pygrgl.load_immutable_grg(grg_or_filename)
            grg_filename = grg_or_filename
        else:
            grg = grg_or_filename
            grg_filename = os.path.join(tmpdirname, "input.grg")
            pygrgl.save_grg(grg, grg_filename)

        split_ranges = []
        for start in range(grg.bp_range[0], grg.bp_range[1], split_threshold):
            split_ranges.append((start, start + split_threshold))
        if len(split_ranges) == 1:
            logv(f"Converting single GRG part to {out_filename}...")
            igd_filename = _grg2igd(grg, out_filename, batch_size)
            shutil.move(igd_filename, out_filename)
        else:

            def merge(filename_list: List[str]):
                if no_merge:
                    return
                in_files = [open(fn, "rb") for fn in filename_list]
                try:
                    in_readers = [pyigd.IGDReader(f) for f in in_files]
                    igd_merge(out_filename, in_readers, True)
                finally:
                    list(map(lambda f: f.close(), in_files))

            logv(f"Using temporary directory {tmpdirname}.")
            logv(f"Splitting GRG into {len(split_ranges)} parts..")
            grg_parts = split_by_ranges(
                grg_filename, split_ranges, jobs, out_dir=tmpdirname
            )
            arguments = [
                (part, os.path.join(tmpdirname, "part_"), batch_size)
                for part in grg_parts
            ]
            logv("Converting GRG parts to IGD files...")
            with Pool(jobs) as pool:
                igd_parts = pool.starmap(_grg2igd, arguments)
            logv(f"Merging {len(igd_parts)} parts into single IGD {out_filename}...")
            merge(igd_parts)


def igd_to_vcf(
    igd_filename: str,
    out_file_obj: TextIO,
    contig: str,
    buffer_lines: int = 1000,
):
    """
    Convert and IGD file to VCF. Can be slow for huge datasets! General
    usage should to either use a Gzip file object for the output, or stdout and
    then pipe the results to bgzip.

    This method produces a VCF file with the variants expanded just like the IGD file.
    To "unexpand" the VCF file, use `bcftools norm -m +any input.vcf -o output.vcf`.

    :param igd_filename: The input IGD filename.
    :type igd_filename: str
    :param out_file_obj: The file handle to write VCF data to.
    :type out_filename: TextIO
    :param contig: The contig name to use in the VCF.
    :type contig: str
    :param buffer_lines: The number of lines to buffer before flushing to disk. Default: 1000.
    :type buffer_lines: int
    """
    separator = "\t"

    with open(igd_filename, "rb") as f:
        igd_file = pyigd.IGDReader(f)

        assert igd_file.ploidy <= 2, "Only haploid or diploid data supported"
        assert igd_file.is_phased, "Unphased data not yet supported"

        last_pos, _ = igd_file.get_position_and_flags(igd_file.num_variants - 1)

        print("##fileformat=VCFv4.2", file=out_file_obj)
        print(
            f"##source=igd_to_vcf({os.path.basename(igd_filename)}) (IGD format v{igd_file.version})",
            file=out_file_obj,
        )
        print(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            file=out_file_obj,
        )
        print(f"##contig=<ID={contig},length={last_pos+1}>", file=out_file_obj)

        individual_ids = igd_file.get_individual_ids()
        if not individual_ids:
            individual_ids = [f"idv_{i}" for i in range(igd_file.num_individuals)]
        header = [
            "CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT",
        ]
        out_file_obj.write("#" + separator.join(header + individual_ids))

        variant_ids = igd_file.get_variant_ids()

        line_buffer = []
        buffered = 0
        next_index = 0
        while next_index < igd_file.num_variants:
            variant_indices = collect_next_site(igd_file, next_index)
            next_index = variant_indices[-1] + 1

            position, _, _ = igd_file.get_position_flags_copies(variant_indices[0])
            ref = igd_file.get_ref_allele(variant_indices[0])
            var_id = (
                variant_ids[variant_indices[0]] if variant_ids else f"var{position}"
            )
            alts = []
            alt_pos = 4
            line_meta = [
                contig,
                str(int(position)),
                var_id,
                ref,
                None,
                ".",
                ".",
                ".",
                "GT",
            ]
            assert line_meta[alt_pos] is None

            row_genotypes = ["0"] * igd_file.num_samples

            alt_count = 1
            for index in variant_indices:
                position, is_missing, samples = igd_file.get_samples(index)

                if not is_missing:
                    assert ref == igd_file.get_ref_allele(
                        index
                    ), f"Multiple REF alleles at position {position}"
                    alts.append(igd_file.get_alt_allele(index))
                    allele_value = str(alt_count)
                    alt_count += 1
                else:
                    allele_value = "."
                for s in samples:
                    row_genotypes[s] = allele_value
            line_meta[alt_pos] = ",".join(alts)

            line_buffer.append("\n")
            line_buffer.append(separator.join(line_meta))
            line_buffer.append(separator)
            if igd_file.ploidy == 1:
                line_buffer.append(separator.join(row_genotypes))
            else:
                line_buffer.append(
                    separator.join(
                        map(
                            lambda alleles: "|".join(alleles),
                            zip(*([iter(row_genotypes)] * igd_file.ploidy)),
                        )
                    )
                )
            buffered += 1

            if buffered >= buffer_lines:
                out_file_obj.write("".join(line_buffer))
                line_buffer = []
                buffered = 0
        out_file_obj.write("".join(line_buffer))


def export_vcf(
    grg_or_filename: Union[pygrgl.GRG, str],
    out_file_obj: TextIO,
    contig: str = "unknown",
    jobs: int = 1,
    batch_size: Union[str, int] = "auto",
    temp_dir: Optional[str] = None,
    split_threshold: int = 5_000_000,
    verbose: bool = False,
):
    """
    Export a GRG to a phased VCF file. Can be slow for huge datasets! General
    usage should to either use a Gzip file object for the output, or stdout and
    then pipe the results to bgzip.

    :param grg_or_filename: The GRG to convert, either as a pygrgl.GRG or the
        filename of a GRG.
    :type grg: Union[pygrgl.GRG, str]
    :param out_file_obj: The file handle to write VCF data to.
    :type out_filename: TextIO
    :param contig: The contig name to use in the VCF. Default: "unknown".
    :type contig: str
    :param jobs: The number of parallel processes to use to do the conversion. The
        speed-up is essentially linear. Default: 1.
    :type jobs: int
    :param batch_size: The number of Mutations to process simultaneously, or the
        string "auto" if you want a reasonable value to be chosen for you.
    :type batch_size: Union[str, int]
    :param temp_dir: The directory to use for intermediate IGD files. The GRG
        is split into multiple pieces and placed in this directory, and then each
        piece gets converted to an IGD file, and then those IGD files are merged
        into the final result. If temp_dir is None, these files are placed in a
        temporary directory which is then deleted upon completion.
    :type temp_dir: Optional[str]
    :param split_threshold: Basepair threshold for splitting the GRG into chunks
        for processing. A split GRG is much faster to operate on than a full sized
        GRG, plus this is how we parallelize the conversion. Default: 5MB.
    :type split_threshold: int
    """
    with _get_temp_dir_context(temp_dir)() as tmpdirname:
        igd_filename = os.path.join(
            tmpdirname, f"{os.path.basename(grg_or_filename)}.igd"
        )
        export_igd(
            grg_or_filename,
            igd_filename,
            jobs=jobs,
            batch_size=batch_size,
            temp_dir=tmpdirname,
            no_merge=False,
            split_threshold=split_threshold,
            verbose=verbose,
        )
        if verbose:
            print("Converting IGD to VCF", file=sys.stderr)
        igd_to_vcf(igd_filename, out_file_obj, contig)
