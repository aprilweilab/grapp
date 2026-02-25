# grapp command-line cheat sheet

A cheat sheet for commands that manipulate GRGs.

### Filtering

All variant-based filters can be combined together in a single command. A lot of the syntax matches
that of `bcftools`.

#### Extract a region of the genome

```
# Get a 1MB region in the "middle"  (20MBP to 21MBP)
grapp filter -r 20000000-21000000 input.grg output.grg
```

#### Minor allele frequency >= 0.01

```
# Only keep variants with frequency >= 0.01 and <= 0.99 (minor allele may be reference)
grapp filter -q 0.01 -Q 0.99 input.grg output.grg
```

#### Variants only

```
# Only keep variants if they have at least one sample (allele count >= 1)
# Assume we have a dataset with 200 haplotypes in it
grapp filter -c 1 -C 199 input.grg output.grg
```

#### Bi-allelic SNPs only

```
grapp filter -v snps -m 2 -M 2 input.grg output.grg
```

#### SNP/MNP _sites_ only

```
# This drops the entire site if any variant is not a SNP or MNP
grapp filter -v snps,mnps -A input.grg output.grg
```

#### INDELs only

```
grapp filter -v indels input.grg output.grg
```

#### Keep individuals by population label

```
# The GRG file must have been created with population labels
grapp filter -P EUR,YRI,CHB input.grg output.grg
```

#### Keep individuals by identifier

```
grapp filter -S indiv1,indiv5 input.grg output.grg

# Or via a file containing one ID per line
grapp filter -S keep_list.txt input.grg output.grg
```

#### Keep individuals by haplotype indices

```
# Keep the first and fifth diploid individual (both haplotypes for each)
# You can also pass a file, like above
grapp filter --hap-samples 0,1,10,11 input.grg output.grg
```

### Viewing and exporting information

#### Show allele frequencies and counts

```
grapp show -f input.grg > frequencies.tsv
grapp show -c input.grg > counts.tsv
```

#### Show GRG info and population labels

```
grapp show -i -P input.grg
```

#### Get individual identifiers

```
grapp show -S input.grg > individuals.txt
```

#### Export GRG to IGD format (tabular)

```
# The more threads (-j) the better!
grapp export --igd output.igd -j 6 input.grg
```

### GWAS, PCA, and phenotypes

#### Phenotype simulation

```
# Simulate a random phenotype with h^2=0.5 and 1000 causal SNPs
grapp pheno -e 0.5 -n 1000 -o simulated.phenotypes.par input.grg
```

#### GWAS

```
# Perform GWAS between the GRG and the simulated phenotype
grapp assoc -p simulated.phenotypes.par -o gwas_results.tsv input.grg
```

#### PCA

```
# Get the 10 largest PCs
grapp pca -d 10 -o output.pcs.tsv input.grg
```