from grapp import nn as grgnn
import numpy
import numpy.typing
import os
import pyigd
import pygrgl
import subprocess
import unittest
from typing import Optional

JOBS = 1


# Matrix is NxM (N = haplotypes, M = mutations)
def igd_from_matrix(genotype_matrix: numpy.typing.NDArray, filename: str):
    with open(filename, "wb") as fout:
        ploidy = 1
        writer = pyigd.IGDWriter(fout, genotype_matrix.shape[0], ploidy)
        writer.write_header()
        for col in range(genotype_matrix.shape[1]):
            sample_list = numpy.flatnonzero(genotype_matrix[:, col])
            writer.write_variant(col, "0", "1", sample_list)
        writer.write_index()
        writer.write_variant_info()
        writer.out.seek(0)
        writer.write_header()


def construct_grg(input_file: str, output_file: Optional[str] = None) -> str:
    cmd = [
        "grg",
        "construct",
        "-p",
        "1",
        "-t",
        "1",
        "-j",
        str(JOBS),
        input_file,
    ]
    if output_file is not None:
        cmd.extend(["-o", output_file])
    else:
        output_file = os.path.basename(input_file) + ".final.grg"
    subprocess.check_call(cmd)
    return output_file


class TestExactDistance(unittest.TestCase):
    def test_identical_samples(self):
        matrix = numpy.array(
            [
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
            ]
        )
        igd_filename = "test_identical_samples.igd"
        igd_from_matrix(matrix, igd_filename)
        grg_file = construct_grg(igd_filename)

        grg = pygrgl.load_immutable_grg(grg_file)
        self.assertEqual(grg.num_samples, 5)
        self.assertEqual(grg.num_mutations, 12)

        nn = grgnn.NearestNeighborContext(grg)
        # All samples are identical.
        expected = numpy.zeros((grg.num_samples, grg.num_samples))
        actual = nn.exact_hamming_dists_by_sample(list(range(grg.num_samples)))
        self.assertTrue(numpy.array_equal(actual, expected))

    def test_identical_mutations(self):
        matrix = numpy.transpose(
            numpy.array(
                [
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 1, 1],
                ]
            )
        )
        igd_filename = "test_identical_samples.igd"
        igd_from_matrix(matrix, igd_filename)
        grg_file = construct_grg(igd_filename)

        grg = pygrgl.load_immutable_grg(grg_file)
        self.assertEqual(grg.num_samples, 9)
        self.assertEqual(grg.num_mutations, 11)

        nn = grgnn.NearestNeighborContext(grg)
        # All mutations are identical
        expected = numpy.zeros((grg.num_mutations, grg.num_mutations))
        actual = nn.exact_hamming_dists_by_mutation(list(range(grg.num_mutations)))
        self.assertTrue(numpy.array_equal(actual, expected))

    def test_simple(self):
        matrix = numpy.array(
            [
                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            ]
        )
        igd_filename = "test_simple.igd"
        igd_from_matrix(matrix, igd_filename)
        grg_file = construct_grg(igd_filename)

        grg = pygrgl.load_immutable_grg(grg_file)
        self.assertEqual(grg.num_samples, 5)
        self.assertEqual(grg.num_mutations, 12)

        nn = grgnn.NearestNeighborContext(grg)
        # We are testing all samples against all samples, so diagonal is 0 and the matrix
        # is symmetric.
        expected = numpy.array(
            [
                [0, 2, 7, 7, 4],
                [2, 0, 5, 7, 2],
                [7, 5, 0, 4, 5],
                [7, 7, 4, 0, 7],
                [4, 2, 5, 7, 0],
            ]
        )
        actual = nn.exact_hamming_dists_by_sample(list(range(grg.num_samples)))
        self.assertTrue(numpy.array_equal(actual, expected))

        # Check all sample pairs from above using the pairwise function
        for i in range(grg.num_samples):
            for j in range(i + 1, grg.num_samples):
                self.assertEqual(
                    nn.fast_pairwise_hamming(i, j, pygrgl.TraversalDirection.UP),
                    expected[i, j],
                )

        # Same for mutations
        expected = numpy.array(
            [
                [0, 4, 3, 3, 3, 1, 3, 3, 4, 3, 2, 3],
                [4, 0, 1, 1, 3, 3, 1, 3, 2, 3, 2, 3],
                [3, 1, 0, 2, 4, 4, 0, 2, 1, 4, 3, 2],
                [3, 1, 2, 0, 2, 2, 2, 4, 3, 2, 1, 4],
                [3, 3, 4, 2, 0, 2, 4, 2, 3, 2, 3, 2],
                [1, 3, 4, 2, 2, 0, 4, 4, 5, 2, 1, 4],
                [3, 1, 0, 2, 4, 4, 0, 2, 1, 4, 3, 2],
                [3, 3, 2, 4, 2, 4, 2, 0, 1, 4, 5, 0],
                [4, 2, 1, 3, 3, 5, 1, 1, 0, 3, 4, 1],
                [3, 3, 4, 2, 2, 2, 4, 4, 3, 0, 1, 4],
                [2, 2, 3, 1, 3, 1, 3, 5, 4, 1, 0, 5],
                [3, 3, 2, 4, 2, 4, 2, 0, 1, 4, 5, 0],
            ]
        )
        actual = nn.exact_hamming_dists_by_mutation(list(range(grg.num_mutations)))
        # Should be symmetric
        self.assertTrue(numpy.array_equal(actual, numpy.transpose(actual)))
        self.assertTrue(numpy.array_equal(actual, expected))
