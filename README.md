Parascopy
=========

Parascopy is designed for robust and accurate estimation of paralog-specific copy number for duplicated genes using whole-genome sequencing.

Created by Timofey Prodanov `timofey.prodanov[at]gmail.com` and Vikas Bansal `vibansal[at]health.ucsd.edu` at the University of California San Diego.

Table of contents
=================
* [Citing Parascopy](#citing-parascopy)
* [Installation](#installation)
* [General usage](#general-usage)
* [Issues](#issues)
* [See also](#see-also)

Citing Parascopy
================

Currently, the paper is in progress, please check later.

Installation
============

<!-- To install the Parascopy you can use `conda`:
```
conda install -c bioconda parascopy
``` -->

Parascopy is written in Python, to manually install it you can run the following commands:
```bash
git clone https://github.com/tprodanov/parascopy.git
cd parascopy
python3 setup.py install
```

To skip dependency installation, you can run
```bash
python3 setup.py develop --no-deps
```

Additionally, you can specify installation path using `--prefix <path>`.

Some parascopy commands require installed
* [samtools](http://samtools.sourceforge.net),
* [tabix](http://www.htslib.org/doc/tabix.html),
* [bgzip](http://www.htslib.org/doc/bgzip.html),
* [bwa](https://github.com/lh3/bwa).

You do not need to install these tools if you installed parascopy through `conda`.

General usage
=============

Main focus of this tool is a *homology table* -- a database of duplications in the genome.

To construct a homology table you would need to run:
```bash
parascopy pretable -f genome.fa -o pretable.bed.gz
parascopy table -i pretable.bed.gz -f genome.fa -o table.bed.gz
```
Note, that the reference genome should be indexed with both `samtools faidx` and `bwa index`.
Alternatively, you can download a precomputed homology table [here](TODO).

To find aggregate and paralog-specific copy number (agCN and psCN) across multiple samples, you should run
```bash
# Calculate background read depth.
parascopy depth -I input.list -g hg38 -f genome.fa -o depth
# Estimate agCN and psCN for multiple samples.
parascopy cn -I input.list -t table.bed.gz -f genome.fa -R regions.bed -d depth -o out1
# Estimate agCN and psCN using model parameters from a previous run.
parascopy depth -I input2.list -g hg38 -f genome.fa -o depth2
parascopy cn-using out1/model -I input2.list -t table.bed.gz -f genome.fa -d depth2 -o out2
```

See `parascopy help` or `parascopy <command> --help` for more information.

Issues
======
Please submit issues [here](https://github.com/tprodanov/parascopy/issues) or send them to `timofey.prodanov[at]gmail.com`.

See also
========

Additionally, you may be interested in these tools:
* [Longshot](https://github.com/pjedge/longshot/): fast and accurate long read variant calling tool,
* [DuploMap](https://gitlab.com/tprodanov/duplomap): improving long read alignments to segmental duplications,
* [Pileuppy](https://gitlab.com/tprodanov/pileuppy): colorful and fast BAM pileup.
