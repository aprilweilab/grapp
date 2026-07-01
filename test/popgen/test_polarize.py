import os
import sys
import tempfile
import unittest
import numpy as np
import pygrgl
from grapp.popgen import polarize_grg

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg

INPUT_DIR = os.path.join(THIS_DIR, "..", "input")


class TestPolarize(unittest.TestCase):
    def test_multiallelic_snv_site_from_vcf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_vcf = os.path.join(INPUT_DIR, "multi.vcf")
            input_grg = os.path.join(tmpdir, "multi.grg")
            output_grg = os.path.join(tmpdir, "multi.polar.grg")

            seq = ["N"] * 1110700
            seq[14370 - 1] = "G"
            seq[17330 - 1] = "A"
            seq[1110696 - 1] = "T"
            seq = "".join(seq)

            grg_file = construct_grg(input_vcf, input_grg, ignore_missing=True)
            grg = pygrgl.load_mutable_grg(grg_file)
            stats = polarize_grg(grg, seq, map_batch_size=2)
            print(stats)
            pygrgl.save_grg(grg, output_grg)
            grg = pygrgl.load_immutable_grg(output_grg)
            self.assertGreater(grg.num_mutations, 0)

            values = np.eye(grg.num_mutations, dtype=np.int32)
            sample_matrix = pygrgl.matmul(grg, values, pygrgl.TraversalDirection.DOWN)

            observed = {}
            for mut_id in range(grg.num_mutations):
                mutation = grg.get_mutation_by_id(mut_id)
                observed[
                    (int(mutation.position), mutation.ref_allele, mutation.allele)
                ] = tuple(
                    int(sample_id)
                    for sample_id in np.flatnonzero(sample_matrix[mut_id] > 0)
                )

            self.assertEqual(
                observed,
                {
                    (17330, "A", "T"): (0, 1, 2, 4, 5),
                    (1110696, "T", "A"): (3,),
                    (1110696, "T", "G"): (0,),
                },
            )


if __name__ == "__main__":
    unittest.main()
