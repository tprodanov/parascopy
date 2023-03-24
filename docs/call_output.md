ParascopyVC output files
------------------------

Parascopy variant calling produces several output files:
- `variants.vcf.gz`,
- `variants_pooled.vcf.gz`,
- `variants.bed.gz`,
- `variants_pooled.bed.gz`.

Output VCF files
----------------

Two output VCF files contain paralog-specific variants and their genotypes.
Here, the variants are written for each repeat copy, with their paralog-specific genotype stored in the `GT` field,
genotype quality in the `GQ` field.
In certain cases, sample paralog-specific copy number is marked by paralog-specific filter
(see [here](docs/cn_output.md#copy-number-profiles)),
such filter is then copied to the `FILT` field.

In addition to `GT` fields, `variants.vcf.gz` contains aggregate (pooled) genotypes,
called across all repeat copies. Aggregate genotypes and their qualities are stored in the `PGT` and `PGQ` fields.
It is possible, especially in complex duplications,
that many variants will not have their paralog-specific genotypes, but will have aggregate genotypes.

There are two INFO fields with valuable information: `pos2` contains positions of all homologous variant positions,
and `overlPSV` shows if the variant overlaps PSV (`T`) or not (`F`).

Pooled vcf file (`variants_pooled.vcf.gz`) contains variants, aggregated across all repeat copies.
File format and fields are similar to these in the `variants.vcf.gz` file, with the difference that
the variants are stored only for the main repeat copy (provided as input repeat copy to the `parascopy cn`).
Additionally, all genotypes in this file are aggregate (pooled), and are stored in the `GT` and `GQ` fields.

In both files, read depth values (`DP`, `AD`, `ADq`) are aggregated across all repeat copies.

Output BED files
----------------

ParascopyVC output two BED files: `variants_pooled.bed.gz` and `variants.bed.gz`,
very similar in content to `res.samples.bed.gz` and `res.paralog.bed.gz`, respectively.
These files can be used to filter output variants and select `parascopyVC` high-confidence regions.
Note, that `res.*.bed.gz` files can also be used to filter output variants, as they are very similar to `variants*.bed.gz` files,
and contain almost the same regions.

| Column in | Column in | Name | Description |
| --- | --- | --- | --- |
| `variants_pooled` | `variants` | | |
| 1 | 1 | chrom | Chromosome name |
| 2 | 2 | start | 0-based genomic position (inclusive) |
| 3 | 3 | end   | 0-based genomic position (exclusive) |
| 4 | 4 | sample | Sample name |
| 5 | 5 | filter | CN estimate filter |
| 6 | 6 | CN | CN estimate |
| 7 | 7 | qual | CN estimate quality |
| - | 8 | agCN_filter | Aggregate CN estimate filter |
| - | 9 | agCN | Aggregate CN estimate |
| - | 10 | agCN_qual | Aggregate CN estimate quality |
| 8 | 11 | homologous_regions | Regions, homologous to the region in columns 1-3 |

`variants.bed.gz` contains both paralog-specific and aggregate copy number values (and their qualities),
while `variants_pooled.bed.gz` contains only aggregate copy number values.

In contrast to `res.*.bed.gz`, it is possible that the output `variants*.bed.gz`
files would contain overlapping regions if the same repeat copy
is covered multiple times from different input duplications.
Nevertheless, one can just merge filtered BED files using `bedtools merge -i ...`.
