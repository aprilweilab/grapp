File Formats
============

The command line interface to grapp primarily reads and writes tab-separated text files.

Phenotype Inputs
----------------

Phenotypes can be input to ``grapp`` in three different formats:

 * `plink-style text file <https://www.cog-genomics.org/plink/1.9/input#pheno>`_
 * Tab-separate value file (TSV), compatible with ``pandas.read_csv(filename, delimiter="\t")``
   * With a single column. The phenotype values only, and the order is assumed to be the individual order.
   * With two columns. The individual ID and phenotype values; the order is still assumed to be the individual order, but the individual ID is used to verify correctness of the order.

plink-style
~~~~~~~~~~~

Text file with optional header line, FID and IID in first two columns, phenotypes in the third column.
The first column (FID) is ignored. The IID column is ignored if it has the value NA, otherwise it is cross-checked against
the order of individual IDs in the GRG file. That is, the *order* of the covariates file must correspond to the order of the individuals
as indexed in the GRG, and if IIDs are provided then ``grapp`` will confirm that this order is correct and reject the input otherwise.

Example:

::

  FID IID PHENOTYPE_A
  0 tsk_9 3.14
  IGNORED tsk_2 54.123

pandas-style
~~~~~~~~~~~~

Style1 has two columns, the individual ID and the phenotype:

::

   individual_id phenotypes
   tsk_1 1.19
   tsk_2 99

The header is optional, so this is also valid:

::

   tsk_1 1.19
   tsk_2 99

The individual ID is also optional, so this is also valid:

::

   1.19
   99

The disadvantage of this last, single column, format is that you don't get any sanity checking on the
individual IDs against the order of samples within the GRG.

Covariates Inputs
-----------------

Covariates can be input to ``grapp`` in two different formats:

 * `plink-style text file <https://www.cog-genomics.org/plink/1.9/input#covar>`_
 * Tab-separate value file (TSV), compatible with ``pandas.read_csv(filename, delimiter="\t")``

You must provide a file with the correct file extension so that ``grapp`` knows which file type to use: ``.txt`` for plink-style,
and ``.tsv`` for pandas-style.

plink-style
~~~~~~~~~~~

Text file with optional header line, FID and IID in first two columns, covariates in remaining columns.
The first column (FID) is ignored. The IID column is ignored if it has the value NA, otherwise it is cross-checked against
the order of individual IDs in the GRG file. That is, the *order* of the covariates file must correspond to the order of the individuals
as indexed in the GRG, and if IIDs are provided then ``grapp`` will confirm that this order is correct and reject the input otherwise.

Example:

::

   FID IID SITE AGE DOB BMI SMOKE
   0 tsk_0 11.1 2.1 3.0 100 99.999999
   0 tsk_1 12.1 22.1 23.0 120 19.191919

pandas-style
~~~~~~~~~~~~

You can also just use a tab-separate file, where each column is a covariate. There is no individual ID, the rows are ordered
according to the order of the individuals (e.g., see ``grapp show -S`` to see the individual order in a GRG).

Example:

::

   PC1     PC2     PC3     PC4     PC5
   -0.02962050343353934    -0.1454314700126317     0.09400946258618631     0.049484006249728846    -0.047684090813491696
   0.08143511260364325     -0.08589047179019427    0.02806784261156361     -0.03490721182219861    -0.008695950854523016
   0.009043842123764047    0.05004380910616621     0.058859397426153766    -0.041742678498370434   0.021168181948134204
   0.07220254903672255     0.006810145300172974    -0.19663115373847032    -0.022568831773233364   0.06676064188763692
   -0.011695803933424008   0.0493589235771859      0.0364513132579304      0.06629422839731464     0.015111070046121165
   0.05903374410746705     0.0643695785650203      -0.02606823869488902    0.040894123687997726    0.019502422077301032
   0.03031031254694341     0.009922066597477694    -0.03350773443115891    0.05706756991552372     -0.19362734163613687


Sample Filtering Inputs
-----------------------

By individual
~~~~~~~~~~~~~

Filter option:

::

  -S INDIVIDUALS, --individuals INDIVIDUALS
                        Keep only the individuals with the IDs given as a comma-separated list or in the given filename.

When providing a list of individuals in a file, there should be one individual ID per line with no extra whitespace. For example,
here is a short input file for the `1,000 Genomes Project <https://www.internationalgenome.org/>`_ dataset that filters to a few individuals:

::

  HG00112
  HG00121
  HG00125
  NA21090
  NA21123
  NA21107


For this command to work, the GRG must have individual identifiers (e.g., the ``VCF`` or ``IGD`` file it was constructed from must have had them).

The order of the individuals as provided will be their new order in the resulting GRG. E.g., individual ``0`` will be ``HG00112``, regardless of its
order in the original GRG.

By haplotype index
~~~~~~~~~~~~~~~~~~

Filter option:

::

  --hap-samples HAP_SAMPLES
                        Keep only the haploid samples with the NodeIDs (indexes) given as a comma-separated list or in the given filename.

When individual identifiers are unavailable, it may be desirable to filter by haplotype index (order of samples in the GRG) instead. The order of
samples in the GRG always follows the original order from the dataset (e.g., the ``VCF`` or ``IGD`` file). For diploid data, each individual's
two haplotype samples are sequential in order. E.g., diploid individual ``0`` maps to haplotype indices ``0, 1``, individual ``1`` maps to
indices ``2, 3``, etc.

When providing a list of haplotype indices by file, there should be one index per line. Each index must be an integer. For example, this input
file selects 4 diploid individuals by their haplotype indices:

::

  50
  51
  0
  1
  8
  9
  32
  33

The order of the haplotypes as provided will be their new order in the resulting GRG. E.g., sample ``0`` in the new GRG will be ``50`` from the old GRG, etc.


Polarization Inputs
-------------------

Polarization requires an ancestral sequence as input, in `FASTA format <https://en.wikipedia.org/wiki/FASTA_format>`_. grapp requires that
the FASTA file contains a single sequence, which represents the ancestral state of each nucleotide. The FASTA file is assumed to be 1-based
indexing, which means that the first nucleotide in the sequence is for base-pair position ``1`` (not ``0``).  There is no case expectation
for the sequence: lower and upper should both work.

We use the excellent `pyfaidx <https://pypi.org/project/pyfaidx/>`_ library for FASTA parsing.