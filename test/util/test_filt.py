from grapp.util.filter import (
    grg_save_freq,
    grg_save_mut_filter,
    multi_grg_save_mut_filter,
)
from grapp.util.simple import allele_counts, allele_frequencies
import pygrgl
import os
import unittest
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, ".."))
from testing_utils import construct_grg, split_and_load

JOBS = 4
CLEANUP = True

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(THIS_DIR, "input")


class TestFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grg_filename = construct_grg("test-200-samples.vcf.gz", "test.filt.grg")
        cls.grg = pygrgl.load_immutable_grg(cls.grg_filename, load_up_edges=False)

    def test_filt_by_freq(self):
        test_filename = "test.filt.freq.grg"
        grg_save_freq(self.grg, test_filename, (0.2, 0.8))

        # Read the filtered GRG
        filt_grg = pygrgl.load_immutable_grg(test_filename, load_up_edges=False)
        counts = allele_counts(filt_grg)
        seen = set()
        for mut_id in range(filt_grg.num_mutations):
            freq = counts[mut_id] / filt_grg.num_samples
            self.assertGreaterEqual(freq, 0.2)
            self.assertLess(freq, 0.8)
            mut = filt_grg.get_mutation_by_id(mut_id)
            seen.add((mut.position, mut.allele))

        # Now compare to the unfiltered
        freqs = allele_frequencies(self.grg)
        for mut_id in range(self.grg.num_mutations):
            mut = self.grg.get_mutation_by_id(mut_id)
            if freqs[mut_id] < 0.2 or freqs[mut_id] >= 0.8:
                self.assertFalse((mut.position, mut.allele) in seen)
            else:
                self.assertTrue((mut.position, mut.allele) in seen)

    def test_multi_filt(self):
        full_freqs = allele_frequencies(self.grg)
        freq_range = (0.2, 0.8)

        test_filename = "test.filt.multi.grg"

        def keep_mut(grg: pygrgl.GRG, mut_id: int):
            return (
                full_freqs[mut_id] >= freq_range[0]
                and full_freqs[mut_id] < freq_range[1]
            )

        grg_save_mut_filter(self.grg, test_filename, keep_mut)

        test_dir = "test.filt.multi.split"
        partial_grgs = split_and_load(
            self.grg_filename, test_dir, 5_000_000, JOBS, CLEANUP
        )
        partial_filenames = [
            f"test.filt.multi.{i}.grg" for i in range(len(partial_grgs))
        ]

        def keep_mut_multi(grg: pygrgl.GRG, mut_id: int, cumu_id: int):
            return (
                full_freqs[cumu_id] >= freq_range[0]
                and full_freqs[cumu_id] < freq_range[1]
            )

        multi_grg_save_mut_filter(partial_grgs, partial_filenames, keep_mut_multi)

        merged_file = "test.filt.multi.merged.grg"
        assert len(partial_filenames) > 0
        subprocess.check_call(["grg", "merge", merged_file] + partial_filenames)

        single_result = pygrgl.load_immutable_grg(test_filename)
        merged_result = pygrgl.load_immutable_grg(merged_file)
        self.assertEqual(single_result.num_mutations, merged_result.num_mutations)

    @classmethod
    def tearDownClass(cls):
        if CLEANUP:
            os.remove(cls.grg_filename)
