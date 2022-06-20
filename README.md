[![GitHub](https://img.shields.io/github/v/tag/tprodanov/parascopy.svg?label=GitHub&color=blueviolet&style=flat-square)](https://github.com/tprodanov/parascopy/releases)
[![Bioconda](https://img.shields.io/conda/v/bioconda/parascopy.svg?label=Bioconda&color=blue&style=flat-square)](https://anaconda.org/bioconda/parascopy)
[![Last updated](https://anaconda.org/bioconda/parascopy/badges/latest_release_date.svg?label=Last%20updated&color=blue&style=flat-square)](https://anaconda.org/bioconda/parascopy)

Parascopy
---------

Parascopy is designed for robust and accurate estimation of paralog-specific copy number for duplicated genes using whole-genome sequencing.

Created by Timofey Prodanov `timofey.prodanov[at]gmail.com` and Vikas Bansal `vibansal[at]health.ucsd.edu` at the University of California San Diego.

Table of contents
-----------------
* [Citing Parascopy](#citing-parascopy)
* [Installation](#installation)
* [General usage](#general-usage)
    * [Input files](#input-files)
* [Visualizing output](#visualizing-output)
* [Output files](#output-files)
* [Precomputed data](#precomputed-data)
* [Test dataset](#test-dataset)
* [Frequently asked questions](#frequently-asked-questions)
* [Issues](#issues)
* [See also](#see-also)

Citing Parascopy
----------------

If you use Parascopy, please cite:
* Prodanov, T. & Bansal, V. Robust and accurate estimation of paralog-specific copy number for duplicated genes using whole-genome sequencing. *Nature Communications* **13**, 3221 (2022). https://doi.org/10.1038/s41467-022-30930-3

Installation
------------

Parascopy is written in Python (≥ v3.6) and is available on Linux and macOS.
You can install Parascopy using `conda`:
```bash
conda config --add channels bioconda
conda config --add channels conda-forge
conda install -c bioconda parascopy
```
In most cases, it takes 2-6 minutes to install Parascopy using `conda`.
If you have problems with installing Parascopy using `conda`, it is possible to create a new `conda` environment
(see more details [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)):
```bash
conda create --name paras_env parascopy
conda activate paras_env
```

Alternatively, you can install it manually using the following commands:
```bash
git clone https://github.com/tprodanov/parascopy.git
cd parascopy
python3 setup.py install
```

Parascopy depends on several Python modules [(see here)](requirements.txt).
To install Parascopy without dependencies, you can run
```bash
python3 setup.py develop --no-deps
```
Additionally, you can specify installation path using `--prefix <path>`.

In case of manual installation, Parascopy requires
* [samtools ≥ 1.11](http://samtools.sourceforge.net),
* [tabix ≥ 1.11](http://www.htslib.org/doc/tabix.html),
* [bgzip ≥ 1.11](http://www.htslib.org/doc/bgzip.html),
* [bwa ≥ 0.7](https://github.com/lh3/bwa).

General usage
-------------

Parascopy uses a database of duplications - *homology table*.
To construct a homology table you would need to run:
```bash
parascopy pretable -f genome.fa -o pretable.bed.gz
parascopy table -i pretable.bed.gz -f genome.fa -o table.bed.gz
```
Note, that the reference genome should be indexed with both `samtools faidx` and `bwa index`.
Alternatively, you can download a [precomputed homology table](#precomputed-data).

In order to find aggregate and paralog-specific copy number profiles (*agCN* and *psCN*), you can run the following commands:
```bash
# Calculate background read depth.
parascopy depth -I input.list -g hg38 -f genome.fa -o depth
# Estimate agCN and psCN for multiple samples.
parascopy cn -I input.list -t table.bed.gz -f genome.fa -R regions.bed -d depth -o out1
# Estimate agCN and psCN for single/multiple samples using model parameters from a previous run.
parascopy depth -I input2.list -g hg38 -f genome.fa -o depth2
parascopy cn-using out1/model -I input2.list -t table.bed.gz -f genome.fa -d depth2 -o out2
```

See `parascopy help` or `parascopy <command> --help` for more information.
Additionally, you can find a test dataset [here](#test-dataset).

### Input files

Input alignment files should be sorted and indexed, both `.bam` and `.cram` formats are supported.
Input filenames can be passed into Parascopy as `-i in1.bam in2.bam ...` or as `-I input-list.txt`,
where `input-list.txt` is a text file with a single filename on each line.

Additionally, you can provide/override sample names with
`-i in1.bam::sampleA in2.bam::sampleB` or, in case of `-I input-list.txt`, as a second entry on each line:
```
in1.bam sampleA
in2.bam sampleB
```

### Modifying reference copy number

After agCN estimation, parascopy searches for reliable PSVs.
However, only samples that are consistent with the reference copy number are used (in a two-copy duplication,
only samples with agCN = 4 will be used).
Additionally, by default, sex chromosomes X and Y are treated as regular chromosomes.
To solve both issues, one can use `--modify-ref` argument and provide updated reference copy numbers for some samples
and some regions.

The BED file can contain the following lines:
```
#CHROM  START    END  SAMPLES  CN  # This line is not necessary.
chr1    10000  20000  *         0  # It is known that the region is missing from all samples.
chrX        0    inf  SAMPLE1,SAMPLE2,SAMPLE3  1  # Male samples have one chrX and one chrY.
chrY        0    inf  SAMPLE1,SAMPLE2,SAMPLE3  1
chrY        0    inf  SAMPLE4,SAMPLE5,SAMPLE6  0  # Female samples are missing chrY.
```

Visualizing output
------------------

It is possible to visualize agCN and psCN detection process.
To do that you need to clone this repository and run scripts in `draw` directory.
The scripts are written in `R` language and require a number of `R` packages:
```r
install.packages(c('argparse', 'tidyverse', 'ggplot2', 'ComplexHeatmap', 'viridis', 'circlize', 'ggthemes', 'RColorBrewer'))
```

Output files
------------

See output file format [here](docs/cn_output.md).

Precomputed data
----------------

You can use the following precomputed data:
- Precomputed homology tables:
    [hg19 v1.2.2 (25 Mb)](https://dl.dropboxusercontent.com/s/93cgf3zcf8pubql/homology_table_hg19.tar) and
    [hg38 v1.2.2 (40 Mb)](https://dl.dropboxusercontent.com/s/okzeedb6gze6zzs/homology_table_hg38.tar).
- Precomputed model parameters for five continental populations (v1.2.5):
    [hg38 (11 Mb)](https://dl.dropboxusercontent.com/s/5fsohggje778dlb/models_v1.2.5.tar.gz).
    Model parameters were calculated using 2504 samples from the 1000 genomes project
    (661 AFR, 503 EUR, 504 EAS, 489 SAS, 347 AMR samples).
    Model parameters require the homology table
    [v1.2.2](https://dl.dropboxusercontent.com/s/okzeedb6gze6zzs/homology_table_hg38.tar).

Compatible reference genomes can be dowloaded from
- [UCSC (hg19 and hg38)](https://hgdownload.soe.ucsc.edu/downloads.html#human),
- [1000 genomes (hg38)](https://github.com/igsr/1000Genomes_data_indexes/blob/master/data_collections/1000_genomes_project/README.1000genomes.GRCh38DH.alignment).

Test dataset
------------

You can find the full test pipeline [here](docs/test_pipeline.sh).
Alternatively, you can follow step-by-step instructions below:

First, place reference human genome (hg38), precomputed homology tables and model parameters in `data` directory.
For single-sample analysis, we subsampled HG00113 human genome, which can downloaded
[here (360 Mb)](https://dl.dropboxusercontent.com/s/46tt30brotjml0y/HG00113.cram).
Please, index the cram file using `samtools index HG00113.cram`.

We start by calculating background read depth, which should take 2-5 minutes:
```bash
parascopy depth -i HG00113.cram -f data/hg38.fa -g hg38 -o depth_HG00113
```
Next, we calculate copy number profiles for 167 duplicated loci using precomputed model parameters
obtained using 503 European samples.
This step takes 10-40 minutes depending on the number of threads (controlled by `-@ N`).
```bash
parascopy cn-using data/models_v1.2.5/EUR \
    -i HG00113.cram -f data/hg38.fa -t data/homology_table/hg38.bed.gz \
    -d depth_HG00113 -o parascopy_HG00113
```
You can analyze a subset of loci by specifying `data/models_v1.2.5/EUR/<locus>.gz`,
for example analysis of the SMN1 locus should take less than a minute.

For multi-sample analysis, we extracted reads aligned to the GBA/GBAP1 locus for 503 European samples,
can be downloaded [here (195 Mb)](https://dl.dropboxusercontent.com/s/o4ntonnxhs780ui/GBA.tar.gz)
(extract using `tar xf GBA.tar.gz`).
Background read depth is already calculated and is located in `GBA/1kgp_depth.csv.gz`.
Calculating copy number profiles for the GBA/GBAP1 locus should take around 25-30 minutes using a single core.
```bash
parascopy cn -I GBA/input.list  -f data/hg38.fa \
    -t data/homology_table/hg38.bed.gz -d GBA/1kgp_depth.csv.gz \
    -r chr1:155231479-155244699::GBA -o parascopy_GBA
```
Here, the region is supplied using the `-r chr:start-end[::name]` format,
alternatively you can supply regions in a BED file using the `-R` argument (optional: fourth column with region names).

Sample output can be found [here (251 Mb)](https://dl.dropboxusercontent.com/s/ddkh3mflezapvqz/test_output.tar.gz),
output files description can be found [here](docs/cn_output.md).

Frequently asked questions
--------------------------

You can find FAQ [here](docs/faq.md).

Issues
------
Please submit issues [here](https://github.com/tprodanov/parascopy/issues) or send them to `timofey.prodanov[at]gmail.com`.

See also
--------

Additionally, you may be interested in these tools:
* [Longshot](https://github.com/pjedge/longshot/): fast and accurate long read variant calling tool,
* [DuploMap](https://gitlab.com/tprodanov/duplomap): improving long read alignments to segmental duplications,
* [Pileuppy](https://gitlab.com/tprodanov/pileuppy): colorful and fast BAM/CRAM pileup.
