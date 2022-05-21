Parascopy CN output files
-------------------------

`parascopy cn` and `parascopy cn-using` output directory contains many files,
most importantly `res.samples.bed.gz`, `res.paralog.bed.gz` and `psvs.vcf.gz`.
Here you can find the description of these output files.

Table of contents
-----------------
* [Copy number profiles (res.samples.bed.gz)](#copy-number-profiles)
* [Paralog-specific output (res.paralog.bed.gz)](#paralog-specific-output)
* [Matrix with CN profiles (res.matrix.bed.gz)](#matrix-with-cn-profiles)
* [Paralogous sequence variants (psvs.vcf.gz)](#paralogous-sequence-variants)

Copy number profiles
--------------------

Aggregate copy number (agCN) and paralog-specific copy number (psCN) are written in the `res.samples.bed.gz` file.
The file is sorted by genomic position and compatable with a BED file format,
you can read it using `zcat <filename>` or extract entries using `tabix <filename> chr:start-end`.

Currently, there are 13 columns in the output file:
| Number | Name | Description |
| ------ | ---- | ----------- |
|  1 | chrom | Chromosome name |
|  2 | start | 0-based genomic position (inclusive) |
|  3 | end   | 0-based genomic position (exclusive) |
|  4 | locus | Locus name (from the input list of loci) |
|  5 | sample | Sample name |
|  6 | agCN_filter | Aggregate CN estimate filter |
|  7 | agCN | Aggregate CN estimate |
|  8 | agCN_qual | Aggregate CN estimate quality |
|  9 | psCN_filter | Paralog-specific CN filter |
| 10 | psCN | Paralog-specific CN estimate
| 11 | psCN_qual | Paralog-specific CN quality |
| 12 | info | Additional information |
| 13 | homologous_regions | Regions, homologous to the region in columns 1-3 |

Each line stores a part of a copy number profile for a single sample, every sample can have multiple entries in `res.samples` file.
Column `homologous_regions` stores regions, homologous to the main region (columns 1-3), `*` if the region is not duplicated.
Homologous regions are stored in `chrom:start-end:strand` format (1-based inclusive).

### agCN and psCN

agCN and psCN columns store aggregate and paralog-specific copy number, respectively.
agCN estimate is an integer in most cases, otherwise it can be `*` (value is unknown), `>N` or `<N`
(value is too high or too low).

Paralog-specific copy number is a tuple of integers.
Represents copy number of each repeat copy, can contain `?` if CN is unknown.
Order of copies is the following: first goes CN for the main region (columns 1-3),
then CN for all homologous regions (column 13) in the same order.

### Copy number filter

`agCN_filter` and `psCN_filter` store CN filter, which can be `PASS` or some other value
(possibly several values separated by a semicolon).
If filter value is not `PASS`, CN estimates may be incorrect even if they have a high quality.

Possible `psCN_filter` values include:
- `HighCN`: copy number is too high to calculate psCN (controlled by `--copy-num-bound`),
- `NoPSVs`: there are no PSVs at all,
- `NoReliable`: there are no reliable PSVs,
- `FewReliable`: there are less than 3 reliable PSVs,
- `NoComplReliable`: there are reliable PSVs, but all of their *f*-values are less than 0.99,
- `LowInfoContent`: information content of all reliable PSVs is less than 0.9,
- `UncertainCN`: agCN value is not an integer.

### Copy number quality

`agCN_qual` stores a [Phred quality](https://en.wikipedia.org/wiki/Phred_quality_score),
quality = 10 represents 90% probability that agCN estimate is correct, quality = 20 represents 99% probability, and so on.
We recommend using quality threshold 20.

`psCN_qual` stores Phred quality score for each paralog-specific CN (same order as `psCN`).
Can contain `*` if `psCN` contains `?`, or can equal to `*` if `psCN` is completely unknown.

### Additional information

Column 12 stores additional information for each entry. Possible information tags:
- `group` and `region_ix`: region group and region index, for which were analyzed.
- `n_windows` and `hmm_windows`: number of 100bp windows in the main region (columns 1-3), and number of windows, used in the agCN HMM.
- `n_psvs`, `rel_psvs`, `semirel_psvs`: number of PSVs, reliable PSVs and semi-reliable PSVs in the region.
Tag `psv_info` stores mean information content of the reliable PSVs. `max_f_value` stores maximal *f*-value of all PSVs.
- `link`: sometimes, several output entries are used together to calculate `psCN` estimate,
in that case they have the same `link` value (unique for a sample, locus and region group).
If `link` is present, `n_psvs` and other tags related to PSVs are summed for all entries with the same `link`.
- `gene_conv`: true or false (`T` or `F`). If true, sample may have gene conversion, see **gene conversion**.
- `agCN_probs`: this tag appears if agCN quality is less than 40, and it stores -log10 probabilities of possible agCN values.
    For example `agCN_probs=7:0.09,6:0.73` means that agCN is either 7 with probability 10<sup> -0.09</sup> ≈ 81%
    or 6 with probability 10<sup> -0.73</sup> ≈ 19%.

Paralog-specific output
-----------------------

Output file `res.paralog.bed.gz` splits `res.samples.bed.gz` output into paralog-specific entries.
Output entries are available only for regions, where paralog-specific copy number is available.
The file has the following columns:
| Number | Name | Description |
| ------ | ---- | ----------- |
| 1 | chrom | Chromosome name |
| 2 | start | 0-based genomic position (inclusive) |
| 3 | end   | 0-based genomic position (exclusive) |
| 4 | sample   | Sample name |
| 5 | filter   | Union over aggregate and paralog-specific CN filter |
| 6 | copy_num | Paralog-specific copy number of the region |
| 7 | qual     | Minimum over the aggregate CN and paralog-specific CN qualities |
| 8 | main_region | See main_region in the `res.samples` file for more information |

Matrix with CN profiles
-----------------------

Output file `res.matrix.bed.gz` summarizes `res.samples.bed.gz` file: it contains agCN and psCN frequencies
across input samples for multiple *overlapping* regions.
Additionally, in most cases it contains agCN and psCN estimates for each sample and each region.

First 9 columns are
| Number | Name | Description |
| ------ | ---- | ----------- |
|  1 | chrom | Chromosome name |
|  2 | start | 0-based genomic position (inclusive) |
|  3 | end   | 0-based genomic position (exclusive) |
|  4 | locus | Locus name (from the input list of loci) |
|  5 | refCN | Reference copy number of the region |
|  6 | agCN_freq | Frequencies of various agCN values across input samples |
|  7 | psCN_freq | Frequencies of various psCN tuples across input samples |
|  8 | info | Additional information |
|  9 | homologous_regions | Regions, homologous to the region in columns 1-3 |

Additional information contains such tags as
- `len`: length of the region,
- `samples`: number of samples and subset of sample names, for which there is a region in the `res.samples` file
    that completely matches genomic coordinates of the region in columns 1-3.

Columns 10-.. contain 7 values for each input sample (separated by spaces):
| Number | Name | Description |
| ------ | ---- | ----------- |
| 1 | agCN | Aggregate CN estimate |
| 2 | agCN_filter | Aggregate CN estimate filter |
| 3 | agCN_qual | Phred-quality of the aggregate CN estimate |
| 4 | psCN | Paralog-specific CN estimate |
| 5 | psCN_filter | Paralog-specific CN estimate filter |
| 6 | psCN_qual | Phred-qualities of the paralog-specific CN estimates |
| 7 | overlap | How much of the region in `res.matrix` file is covered by regions from `res.samples` file |

See above for the descriptions of agCN, psCN estimates as well as CN filters and qualities.

If a sample has several CN estimates for the region in columns 1-3, the corresponding entry in `res.matrix`
file will contain `"!"` symbols. Therefore, `res.samples` file contains more information for individual samples.

Paralogous sequence variants
----------------------------

Paralogous sequence variants (PSVs) are small differences between repeat copies.
`psvs.vcf.gz` is a VCF file that contains all PSVs in the analyzed loci, as well as allelic read counts and genotypes
for input samples.

For each variant there are several `INFO` fields that are equal for all samples:
- `pos2`: PSV positions in other copies.
    If there are *n* repeat copies in the duplication, `pos2` will contain *n-1* entries, (first copy is in the columns 1-2).
    Format of each entry is `chrom:pos:strand[:allele_index]` (pos is 1-based).
    Allele index is present if there are more than 2 repeat copies and it shows
    which allele is in the reference sequence of the corresponding repeat copy.
    Allele index = 0 represents `REF` allele (column 4), allele index = 1 represents 1st allele from `ALT` alleles (column 5),
    and so on.
    First repeat copy (columns 1-2) always has `REF` allele in its reference sequence.
- `fval`: PSV *f*-values (frequency of the reference allele for each repeat copy), length = number of repeat copies.
    Can be NaN if *f*-values are unknown.
- `info`: PSV information content. If information content is close to 1, all of the PSV alleles are observed in most samples,
    on the other hand if information content is close to 0, only one PSV allele is observed in most samples.
- `rel`: PSV reliability:
    * r (reliable) - all *f*-values are at least 0.95,
    * s (semi-reliable) - all *f*-values are at least 0.8,
    * u (unreliable) - *f*-values are unknown or at least one less than 0.8.

Additionally, each sample may have the following `FORMAT` values:
- `AD`: allelic read depth.
- `DP`: total read depth.
    Total read depth may be bigger than `sum(AD)` as there may be reads that do not support any of the PSV alleles,
    this is usually due to low-complexity sequence and not due to new alleles.
- `GT`: PSV genotype. Genotype `0/0/0/1/1` would mean that sample agCN = 5 and there are three copies with REF allele
    and two copies with ALT allele.
- `GQ`: Phred-quality of the PSV genotype.
- `psCN`: sample paralog-specific CN as if it was calculated using only this PSV.
- `psCNq`: paralog-specific CN quality.
