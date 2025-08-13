# grapp

A library and command-line tool for tackling problems in statistical and population genetics, implemented
on top of the Genotype Representation Graph (GRG) format. [GRG](https://github.com/aprilweilab/grgl) is
a file format and data structure that losslessly represents a genetic dataset. It has the advantage of
compressing large datasets significantly, while also making calculations over that dataset extremely fast
(see [the paper](https://www.nature.com/articles/s43588-024-00739-9) and the
[core library](https://github.com/aprilweilab/grgl)).

## Installation

Clone the repository, then run:
```
pip install /path/to/grapp/
```

## Modules

### assoc

Perform association tests between phenotypes and genotypes.

#### Command Line

```
usage: grapp assoc [-h] [-p PHENOTYPES] [-c COVARIATES] [-o OUT_FILE] grg_input

positional arguments:
  grg_input             The input GRG file

options:
  -h, --help            show this help message and exit
  -p PHENOTYPES, --phenotypes PHENOTYPES
                        The file containing the phenotypes. If no file is provided, random phenotype values are used.
  -c COVARIATES, --covariates COVARIATES
                        Covariates text file to load
  -o OUT_FILE, --out-file OUT_FILE
                        Tab-separated output file (with header); exported Pandas DataFrame. Default: <grg_input>.assoc.tsv
```

#### Library

There are methods for GWAS with (`grapp.assoc.linear_assoc_covar`) and without covariates (`grapp.assoc.linear_assoc_no_covar`).

### linalg

Linear algebra functionality that integrates GRG with [numpy](https://numpy.org/) and [scipy](https://scipy.org/). The main workhorses behind this module are
the operators compatible with [scipy.sparse.linalg.LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.LinearOperator).

#### Command Line

The Principle Component Analysis (PCA) is available via the command line:
```
usage: grapp pca [-h] [-d DIMENSIONS] [-o PCS_OUT] [--normalize] [--pro-pca] grg_input

positional arguments:
  grg_input             The input GRG file

options:
  -h, --help            show this help message and exit
  -d DIMENSIONS, --dimensions DIMENSIONS
                        The number of PCs to extract. Default: 10.
  -o PCS_OUT, --pcs-out PCS_OUT
                        Output filename to write the PCs to. Default: "<grg_input>.pcs.tsv"
  --normalize           Normalize the PCs according to sqrt(eigenvalue) for each.
  --pro-pca             Use the ProPCA algorithm to compute principle components.
```

#### Library

The core of the library are the `LinearOperator`s that operate on GRGs:
* `grapp.linalg.ops_scipy.SciPyXOperator`: An operator that performs matrix multiplication against the genotype matrix `X` (`NxM`) or its transpose (`MxN`).
* `grapp.linalg.ops_scipy.SciPyXTXOperator`: An operator that performs matrix multiplication against the `MxM` product `transpose(X) * X`.
* `grapp.linalg.ops_scipy.SciPyStdXOperator`: The same as `SciPyXOperator`, except the genotype matrix is standardized by using the allele frequencies (standard deviation via binomial distribution).
* `grapp.linalg.ops_scipy.SciPyStdXTXOperator`: The same as `SciPyXTXOperator`, except the genotype matrix is standardized by using the allele frequencies (standard deviation via binomial distribution).

Additionally, there is a helpful method for eigen decomposition (`grapp.linalg.eigs`) and PCA (`grapp.linalg.PCs`).

### util

Common utility functions for working with the GRG format.

#### Command Line

GRG can be exported to tabular data formats. `.vcf.gz` is supported but slow. It is recommended to use [IGD](https://github.com/aprilweilab/picovcf?tab=readme-ov-file#indexable-genotype-data-igd) which is dramatically faster and more compact, while being similar to VCF in how it is structured. Export command:
```
usage: grapp export [-h] [--igd IGD | --vcf VCF] [-f] [-j JOBS] [--temp-dir TEMP_DIR] [--contig CONTIG] grg_input

positional arguments:
  grg_input             The input GRG file

options:
  -h, --help            show this help message and exit
  --igd IGD             Export the entire dataset to the given IGD filename.
  --vcf VCF             Export the entire dataset to the given VCF filename. Use '-' to write to stdout (and, e.g., pipe through bgzip). If the filename ends with .gz then the Python GZIP codec will be used
                        (not bgzip). Otherwise, a plaintext VCF file will be created.
  -f, --force           Force overwrite of the output file, if it exists.
  -j JOBS, --jobs JOBS  Number of processes/threads to use, if possible. Default: 1.
  --temp-dir TEMP_DIR   Put all temporary files in the given directory, instead of creating a directory in the system temporary location. WARNING: Intermediate/temporary files will not be cleaned up when
                        this is specified.
  --contig CONTIG       Use the given contig name when exporting to VCF. Default: "unknown".
```

GRG files can be filtered, prior to performing analysis on them. See the filter command:
```
usage: grapp filter [-h] (-S INDIVIDUALS | --hap-samples HAP_SAMPLES | -r RANGE) grg_input grg_output

positional arguments:
  grg_input             The input GRG file
  grg_output            The output GRG file

options:
  -h, --help            show this help message and exit
  -S INDIVIDUALS, --individuals INDIVIDUALS
                        Keep only the individuals with the IDs given as a comma-separated list or in the given filename.
  --hap-samples HAP_SAMPLES
                        Keep only the haploid samples with the NodeIDs (indexes) given as a comma-separated list or in the given filename.
  -r RANGE, --range RANGE
                        Keep only the variants within the given range, in base pairs. Example: "lower-upper", where both are integers and lower is inclusive, upper is exclusive.
```

#### Library

Library documentation will be generated soon.

### nn

Experimental library for nearest neighbors search over GRG.

The `nn` module lets you search a dataset stored as a GRG for nearest neighbors in a variety of ways:
* Similarity is either between samples (haplotypes) or mutations (variants). I.e., you can ask to find
  samples that are similar to a given sample, or mutations that are similar to a given mutation.
* Similarity is defined as the Hamming distance between items. The Hamming distance is just the number
  of differences, so for two samples their distance is defined as the number of mutations that either of
  them has, but both of them do not. I.e., `Hamming(A, B) = |Muts(A)| + |Muts(B)| - 2*|Muts(A) intersect Muts(B)|`.
* There are APIs that let you query using a sample/mutation already in the dataset (GRG) or more generally
  you can query using an external sample/mutation that is not in the dataset, though your options may
  be slightly more limited in the latter case.


