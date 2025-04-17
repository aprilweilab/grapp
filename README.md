# grgnn

Nearest neighbors search over GRG. A [Genotype Representation Graph](https://github.com/aprilweilab/grgl) is a file format that losslessly
represents a genetic dataset. It has the advantage of compressing large datasets significantly, while also making calculations over that
dataset extremely fast (see [the paper](https://www.nature.com/articles/s43588-024-00739-9)).

The `grgnn` library lets you search a dataset stored as a GRG for nearest neighbors in a variety of ways:
* Similarity is either between samples (haplotypes) or mutations (variants). I.e., you can ask to find
  samples that are similar to a given sample, or mutations that are similar to a given mutation.
* Similarity is defined as the Hamming distance between items. The Hamming distance is just the number
  of differences, so for two samples their distance is defined as the number of mutations that either of
  them has, but both of them do not. I.e., `Hamming(A, B) = |Muts(A)| + |Muts(B)| - 2*|Muts(A) intersect Muts(B)|`.
* There are APIs that let you query using a sample/mutation already in the dataset (GRG) or more generally
  you can query using an external sample/mutation that is not in the dataset, though your options may
  be slightly more limited in the latter case.

