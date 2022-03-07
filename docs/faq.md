# Frequently asked questions

## Does Parascopy work with exome data?

Parascopy only works with WGS data. Parascopy cannot analyze exome or targeted sequencing data as it is not possible to
easily estimate background read depth in non-duplicated regions.

## Is it possible to analyze non-human genomes?

Yes, it is possible; however, one would need to construct a set of non-duplicated windows that can be
used in `parascopy depth` to calculate background read depth. Please contact the authors for more details.

## Is it possible to analyze only one sample (small number of samples)?

Yes, it is possible. In most cases, the best course of action is to use
precomputed model parameters [(11 Mb)](https://dl.dropboxusercontent.com/s/5fsohggje778dlb/models_v1.2.5.tar.gz).
Alternatively, you can simply run `parascopy cn` on a new sample, but note, that only aggregate copy number will be
estimated as it is impossible to find reliable PSVs from a single sample.

## What samples should I analyze together?

It is preferable to analyze samples from the same population, close populations or at least the same continental population.
Additionally, if sequencing data have different library preparation techniques, it may exhibit different
sequencing biases, which would lower copy number estimation accuracy.
If that is the case, you can try using `--no-multipliers` flag.

## Output file is empty.

It may be the case that the target region is too short, or reference copy number jumps too much.
By default, Parascopy estimates copy number for regions longer than 1100 bp with constant reference copy number
(usually, even longer regions are required). Try using larger regions; lowering `--min-windows` or raising
`--window-filtering` parameters.

## Aggregate copy number jumps too much, often goes to zero.

it is possible that the alignment file is missing reads for some duplicated loci.
You can try to map unaligned reads, or map all reads using a different mapping tool.
Use `samtools view input.bam "*"` to extract unaligned reads (does not extract unmapped reads with a mapped mate).

## Is it possible to get paralog-specific copy number for regions with many copies?

It is possible to raise `--max-ref-cn` abd `--pscn-bound` parameters, but this would lead to long runtime,
as the number of possible paralog-specific copy number tuples grows exponentially with the aggregate copy number.
