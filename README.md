# grapp

`grapp` provides statistical and population genetics analyses that natively use the [GRG](https://github.com/aprilweilab/grgl)
format. GRG is an order of magnitude smaller than other representations, and can be multiple orders of
magnitude faster for calculations, such as PCA.

**Statgen features**: GWAS, PCA, phenotype simulation (via [grg\_pheno\_sim](https://github.com/aprilweilab/grg_pheno_sim)), linear mixed models, filtering samples and variants.

**Popgen features**: Polarization, integration with [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/index.html) for various SFS and population statistics, conversions between [tskit.TreeSequence](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence) files and various other formats.

`grapp` provides a command-line interface (ala plink or GCTA), and also a Python API that integrates with [numpy](https://numpy.org/), [scipy](https://scipy.org/),
and [pandas](https://pandas.pydata.org/). Of particular interest to method developers, `grapp` provides [linear operators](https://grapp.readthedocs.io/en/latest/reference/grapp.html#linear-algebra)
that are compatible with [scipy.sparse.linalg](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) and [other compatible libraries](https://grapp.readthedocs.io/en/latest/overview/compatible.html),
which enables iterative numerical algorithms to scale to datasets with a million samples and hundreds of millions of variants.
Some notes on usage are below, and you can also browse [the Python documentation for grapp](https://grapp.readthedocs.io/en/latest/).

Check out the [command-line cheatsheet](https://github.com/aprilweilab/grapp/blob/main/CHEATSHEET.md) for examples on how to perform many tasks.
There are also some tutorials in the [GRG documentation](https://grgl.readthedocs.io/en/latest/tutorials/index.html) (and correspond [Jupyter notebooks](https://github.com/aprilweilab/grgl/tree/main/doc/tutorials/notebooks)).

See [the GRG paper](https://www.nature.com/articles/s43588-024-00739-9) and the [main GRG documentation](https://grgl.readthedocs.io/en/latest/index.html)
for more details about Genotype Representation Graphs.

## Installation

```
pip install grapp
```

## Functionality

### GWAS

Perform association tests between phenotypes and genotypes.

* Command: `grapp assoc [-h] [-p PHENOTYPES] [-b] [-c COVARIATES] [-o OUT_FILE] [-r] [-s] [-j JOBS] grg_input [grg_input ...]`
* Python: [grapp.assoc.linear\_assoc\_covar()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.assoc.linear_assoc_covar), [grapp.assoc.linear\_assoc\_no\_covar()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.assoc.linear_assoc_no_covar)

### PCA

Principal Component Analysis (PCA) using iterative methods.

* Command: `grapp pca [-h] [-d DIMENSIONS] [-o PCS_OUT] [--normalize] [--pro-pca] grg_input`
* Python: [grapp.linalg.PCs()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.linalg.PCs), [grapp.linalg.eigs()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.linalg.eigs)

### BOLT-LMM

[BOLT-LMM](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html) reimplemented on GRG.

* Command: `grapp bolt-lmm [-h] -p PHENOTYPES [-c COVARIATES] [--covar-cols [COVAR_COLS ...]] [--q-covar-cols [Q_COVAR_COLS ...]] [--covar-max-levels COVAR_MAX_LEVELS] [--num-calib-snps NUM_CALIB_SNPS] [--mc-trials MC_TRIALS] [--cg-tol CG_TOL] [--max-iter MAX_ITER] [--seed SEED] [-j JOBS] [-o OUT_FILE] grg_input [grg_input ...]`
* Python: [grapp.assoc.bolt\_lmm\_inf()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#module-grapp.assoc)

### Filtering

Filtering variants and samples, with syntax similar to that of `bcftools`.

* Command: `grapp filter [-h] [-j JOBS] [-S INDIVIDUALS | --hap-samples HAP_SAMPLES | -P POPULATIONS] [-r RANGE] [-c MIN_AC] [-C MAX_AC] [-q MIN_AF] [-Q MAX_AF] [-v TYPES] [-A] [-m MIN_ALLELES] [-M MAX_ALLELES] [-H HWE] [--multi-ref] grg_input grg_output`
* Python: [grapp.util.filter.grg\_save\_freq()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.util.filter.grg_save_freq), [grapp.util.filter.grg\_save\_individuals()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.util.filter.grg_save_individuals), [etc.](https://grapp.readthedocs.io/en/latest/reference/grapp.html#module-grapp.util.filter).

### Phenotype Simulation

Simulate phenotypes, for testing GWAS and association study scenarios.

* Command: `grapp pheno [-h] [-e HERITABILITY] [-n NUM_CAUSAL] [--seed SEED] [--no-normalize] [--save-effects SAVE_EFFECTS] [-o OUT_FILE] grg_input`
* Python: See [grg\_pheno\_sim](https://github.com/aprilweilab/grg_pheno_sim/)

### Data Conversion

GRG can be converted back to a tabular (sparse matrix) format, [IGD](https://picovcf.readthedocs.io/en/latest/igd_overview.html):

* Command: `grapp export [-h] [--igd IGD | --vcf VCF] [-f] [-j JOBS] [--temp-dir TEMP_DIR] [--contig CONTIG] grg_input`
* Python: [grapp.util.igd.export\_igd()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.util.igd.export_igd)

`grapp` also supports converting [tskit.TreeSequence](https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence) to IGD, since for really large
simulated `TreeSequence` files (e.g., from [msprime](https://tskit.dev/msprime/docs/stable/intro.html)) converting them to VCF can take days (or weeks):

* Command: `grapp ts2igd [-h] [-f] [-j JOBS] [--temp-dir TEMP_DIR] ts_input igd_output`

You can also convert `TreeSequence` to _GRG_ using the core GRG library, via `grg convert ts_input igd_output`. In situations where converting a `TreeSequence`
to VCF takes days, converting to IGD takes hours, and converting to GRG takes minutes.

### Simple Computations

The following information can be easily displayed on the command line: GRG summary, HWE p-values, allele counts,
allele frequencies, sample identifiers, mapping between samples and populations.

* Command: `grapp show [-h] [-j JOBS] [-S | -P | -i | -f | -c | -H | --multi-sites] [--multi-ref] grg_input`
* Python: [grapp.util.simple.hwe()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.util.simple.hwe), [grapp.util.simple.get\_zygosities()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.util.simple.get_zygosities), [etc.](https://grapp.readthedocs.io/en/latest/reference/grapp.html#module-grapp.util.simple).

### LinearOperators

* `grapp.linalg.ops_scipy.SciPyXOperator`: An operator that performs matrix multiplication against the genotype matrix `X` (`NxM`) or its transpose (`MxN`).
* `grapp.linalg.ops_scipy.SciPyXTXOperator`: An operator that performs matrix multiplication against the `MxM` product `transpose(X) * X`.
* `grapp.linalg.ops_scipy.SciPyStdXOperator`: The same as `SciPyXOperator`, except the genotype matrix is standardized by using the allele frequencies (standard deviation via binomial distribution).
* `grapp.linalg.ops_scipy.SciPyStdXTXOperator`: The same as `SciPyXTXOperator`, except the genotype matrix is standardized by using the allele frequencies (standard deviation via binomial distribution).

Additionally, there is a helpful method for eigen decomposition (`grapp.linalg.eigs`) and PCA (`grapp.linalg.PCs`).

### scikit-allel Integration

The primary integration with [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/index.html) is via the 
[grapp.popgen.allele\_counts()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.popgen.allele_counts) and
[grapp.popgen.pop\_allele\_counts()](https://grapp.readthedocs.io/en/latest/reference/grapp.html#grapp.popgen.pop_allele_counts) methods,
which return output compatible with many `scikit-allel` functions, such as [allel.sequence\_diversity()](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.sequence_diversity),
[allel.watterson\_theta()](https://scikit-allel.readthedocs.io/en/stable/stats/diversity.html#allel.watterson_theta), [allel.hudson\_fst()](https://scikit-allel.readthedocs.io/en/stable/stats/fst.html#allel.hudson_fst),
and [many](https://scikit-allel.readthedocs.io/en/stable/stats/fst.html) [others](https://scikit-allel.readthedocs.io/en/stable/stats/sf.html).

See the [grapp.popgen](https://grapp.readthedocs.io/en/latest/reference/grapp.html#population-genetics) documentation for more details.

### Nearest Neighbors

Experimental library for nearest neighbors search over GRG.

The [grapp.nn](https://grapp.readthedocs.io/en/latest/reference/grapp.html#module-grapp.nn) module lets you search a dataset
stored as a GRG for nearest neighbors in a variety of ways:
* Similarity is either between samples (haplotypes) or mutations (variants). I.e., you can ask to find
  samples that are similar to a given sample, or mutations that are similar to a given mutation.
* Similarity is defined as the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) between items. The Hamming 
  distance is just the number of differences, i.e., `Hamming(A, B) = |Muts(A)| + |Muts(B)| - 2*|Muts(A) intersect Muts(B)|`.
* There are APIs that let you query using a sample/mutation already in the dataset (GRG) or more generally
  you can query using an external sample/mutation that is not in the dataset, though your options may
  be slightly more limited in the latter case.

For changes to the Mikado integration, follow the shared [contribution guide](https://github.com/CornellHPC/GRG-SpMV/blob/rocm-support/CONTRIBUTING.md).
