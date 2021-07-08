Parascopy CN output files
-------------------------

`parascopy cn` and `parascopy cn-using` output directory contains many files, most importantly `res.samples.bed`
and `res.matrix.bed`. You can find output files description here.

Table of contents
-----------------
* [Copy number profiles](#copy-number-profiles)

Copy number profiles
--------------------

Aggregate copy number (agCN) and paralog-specific copy number (psCN) are written in `res.samples.bed`.
The file is sorted by genomic position and compatable with a BED file format,
which means you can use `tabix` and `bedtools` on it.

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

Each line stores a part of a copy number profile for a single sample, every sample can have multiple entries in `res.samples.bed`.
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

Column 12 stores additional information for each entry. Possible information keys:
- `group` and `region_ix`: region group and region index, for which were analyzed.
- `n_windows` and `hmm_windows`: number of 100bp windows in the main region (columns 1-3), and number of windows, used in the agCN HMM.
- `n_psvs`, `rel_psvs`, `semirel_psvs`: number of PSVs, reliable PSVs and semi-reliable PSVs in the region.
Key `psv_info` stores mean information content of the reliable PSVs. `max_f_value` stores maximal *f*-value of all PSVs.
- `link`: sometimes, several output entries are used together to calculate `psCN` estimate,
in that case they have the same `link` value (unique for a sample, locus and region group).
If `link` is present, `n_psvs` and other keys related to PSVs are summed for all entries with the same `link`.
- `gene_conv`: true or false (`T` or `F`). If true, sample may have gene conversion, see **gene conversion**.
