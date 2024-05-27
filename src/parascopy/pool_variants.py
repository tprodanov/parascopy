#!/usr/bin/env python3

import argparse
import pysam
import os
import itertools
import parasail
import sys
import operator
import re
import tempfile
import numpy as np
import pathlib
from enum import Enum
from collections import defaultdict

from .inner import common
from .inner import variants as variants_
from .inner import itree
from .inner import alignment
from .inner.genome import Genome, Interval
from .inner.cigar import Cigar, Operation
from .inner import errors
from . import pool_reads
from . import detect_cn
from . import psvs as psvs_
from . import long_version


PADDING = 1


def _check_phased(variants, gts, sample, unphased_dist):
    """
    Returns pairs (start, end) of the discarded regions.
    """
    if len(variants) <= 1:
        return ()
    gt_is_homozygous = [len(set(gt)) <= 1 for gt in gts]

    disc_regions = []
    count_unphased = 0
    unphased_example = None
    for var_i, variant in enumerate(variants):
        if gt_is_homozygous[var_i] or variant.samples[sample].phased:
            continue
        var_end = variant.start + len(variant.ref)

        for next_i in range(var_i + 1, len(variants)):
            next_var = variants[next_i]
            if next_var.start - var_end > unphased_dist:
                break
            if not gt_is_homozygous[next_i]:
                count_unphased += 1
                if unphased_example is None:
                    unphased_example = '{}:{} and {}'.format(variant.chrom, variant.pos, next_var.pos)
                disc_regions.append((variant.start, next_var.start + len(next_var.ref)))

    if count_unphased:
        common.log('WARN: {} close-by unphased heterozygous variants, such as {}'
            .format(count_unphased, unphased_example))
    return disc_regions


def _get_genotypes(variants, sample):
    gts = []
    for var_i, variant in enumerate(variants):
        if 'GT' not in variant.samples[sample]:
            raise ValueError('Variant {}:{} has no genotype for sample {}'.format(variant.chrom, variant.pos, sample))
        gt = variant.samples[sample]['GT']
        if gt is None or gt[0] is None:
            raise ValueError('Variant {}:{} has no genotype for sample {}'.format(variant.chrom, variant.pos, sample))
        if gts and len(gts[0]) != len(gt):
            raise ValueError('Variants {}:{} and {}:{} have different copy number for sample {}'
                .format(variant.chrom, variants[0].pos, variant.chrom, variant.pos, sample))
        gts.append(gt)
    return gts


def _get_allele_cigars(variant, parasail_args):
    ref = variant.ref
    cigars = [[(len(ref), Operation.SeqMatch)]]
    for alt in variant.alts:
        subaln = parasail.nw_trace_scan_sat(alt, ref, *parasail_args)
        curr_cigar = []
        for value in subaln.cigar.seq:
            length = value >> 4
            op = Operation(value & 0xf)
            curr_cigar.append((length, op))
        cigars.append(curr_cigar)
    return cigars


class _Flag(Enum):
    Discard = 0
    Keep = 1
    Ref = 2

    def __lt__(self, oth):
        return self.value.__lt__(oth.value)


def _reconstruct_regions(endpoints, interval, consider_ref=False):
    """
    Reconstruct regions from endpoints: list (pos, is_start: bool, flag: _Flag).
    If consider_ref is True, output only regions, where the copy number is equal to the reference copy number.
    Otherwise, flag = 2 is forbidden.
    """
    endpoints.sort()
    result = []
    depth_keep = 0
    depth_disc = 0
    depth_ref = 0

    last_pos = None
    for pos, is_start, flag in endpoints:
        pos = np.clip(pos, interval.start, interval.end)
        if last_pos is not None and last_pos < pos and depth_disc == 0 and \
                depth_keep > 0 and depth_ref == consider_ref * depth_keep:
            result.append(Interval(interval.chrom_id, last_pos, pos))
        last_pos = pos

        inc = 2 * is_start - 1
        if flag == _Flag.Discard:
            depth_disc += inc
        elif flag == _Flag.Keep:
            depth_keep += inc
        elif flag == _Flag.Ref:
            assert consider_ref
            depth_ref += inc
        else:
            assert False
    assert depth_disc == 0 and depth_keep == 0 and depth_ref == 0
    return result


def _create_consensus_alns(vcf_file, region, region_seq, genome, parasail_args, limit_tree, unphased_dist):
    """
    Creates consensus sequences and alignments from variants within a given region.
    Returns
        - Keep regions: regions where variant calls were added.
        - matrix (n_samples x n_copies) with tuples (consensus sequence: str, Alignment).
    """
    endpoints = []
    if limit_tree is None:
        endpoints.append((region.start, True, _Flag.Keep))
        endpoints.append((region.end, False, _Flag.Keep))
    else:
        for limit_region in limit_tree.overlap_iter(region):
            endpoints.append((limit_region.start, True, _Flag.Keep))
            endpoints.append((limit_region.end, False, _Flag.Keep))
    # if not endpoints:
    #     pass

    samples = vcf_file.header.samples
    n_samples = len(samples)

    variants = list(common.checked_fetch(vcf_file, region, genome))
    # Matrix (n_samples x n_variants) of genotypes: each element tuple of allele indices.
    sample_gts = [_get_genotypes(variants, sample) for sample in samples]
    # Vector (n_samples) of copy numbers (int).
    sample_cns = [len(gts[0]) if gts else 2 for gts in sample_gts]
    # Matrix (n_samples x n_copies) of consensus sequences (str).
    consensus_seqs = [[''] * sample_cn for sample_cn in sample_cns]
    # Matrix (n_samples x n_copies) of consensus CIGARs (each element = list).
    consensus_cigars = [[[] for _ in range(sample_cn)] for sample_cn in sample_cns]

    for sample_id, sample in enumerate(samples):
        for disc_start, disc_end in _check_phased(variants, sample_gts[sample_id], sample, unphased_dist):
            endpoints.append((disc_start - PADDING, True, _Flag.Discard))
            endpoints.append((disc_end + PADDING, False, _Flag.Discard))

    region_start = region.start
    last_seq_upd = 0
    for variant_i, variant in enumerate(variants):
        ref_len = len(variant.ref)
        variant_region = Interval(region.chrom_id, variant.start, variant.start + ref_len)
        overlap_size = ref_len if limit_tree is None else limit_tree.intersection_size(variant_region)
        is_indel = any(len(allele) != ref_len for allele in variant.alts)

        if not region.contains(variant_region) or overlap_size != ref_len or \
            (is_indel and (variant_region.start == region.start or variant_region.end == region.end)):
            # Variant overlaps the edge of the duplication OR is out of the limiting regions (not inside complitely).
            if overlap_size:
                # Do not add discard region if there is no overlap between the limit_tree and the variant_region.
                endpoints.append((variant_region.start - PADDING, True, _Flag.Discard))
                endpoints.append((variant_region.end + PADDING, False, _Flag.Discard))
            continue

        add_seq_len = variant.start - region_start - last_seq_upd
        if add_seq_len < 0:
            common.log('WARN: Variants are not sorted or overlap: see variant {}:{}'
                .format(variant.chrom, variant.pos))

            assert variant_i > 0
            endpoints.append((variants[variant_i - 1].start - PADDING, True, _Flag.Discard))
            endpoints.append((variant_region.end + PADDING, False, _Flag.Discard))
            # rel_end = variant_region.end - region_start
            # if rel_end > last_seq_upd:
            #     add_seq_len = rel_end - last_seq_upd
            #     for sample_id in range(n_samples):
            #         for copy_i in range(sample_cns[sample_id]):
            #             Cigar.append(consensus_cigars[sample_id][copy_i], add_seq_len, Operation.SeqMatch)
            #     last_seq_upd = rel_end
            continue

        add_seq = region_seq[last_seq_upd : variant.start - region_start]
        allele_cigars = _get_allele_cigars(variant, parasail_args)

        for sample_id in range(n_samples):
            for copy_i in range(sample_cns[sample_id]):
                allele_ix = sample_gts[sample_id][variant_i][copy_i]
                consensus_seqs[sample_id][copy_i] += add_seq + variant.alleles[allele_ix]
                cigar = consensus_cigars[sample_id][copy_i]
                if add_seq_len:
                    Cigar.append(cigar, add_seq_len, Operation.SeqMatch)
                Cigar.extend(cigar, iter(allele_cigars[allele_ix]))
        last_seq_upd = variant_region.end - region_start

    add_seq_len = len(region) - last_seq_upd
    if add_seq_len < 0:
        raise ValueError('Last variant overlaps end of the duplication')
    add_seq = region_seq[last_seq_upd:]

    # Matrix (n_samples x n_copies) of consensus alignments.
    consensus_alns = [[None] * sample_cn for sample_cn in sample_cns]
    for sample_id in range(n_samples):
        for copy_i in range(sample_cns[sample_id]):
            consensus_seq = consensus_seqs[sample_id][copy_i] + add_seq
            cigar = consensus_cigars[sample_id][copy_i]
            if add_seq_len:
                Cigar.append(cigar, add_seq_len, Operation.SeqMatch)

            cigar = Cigar.from_tuples(cigar)
            assert len(region_seq) == cigar.ref_len and len(consensus_seq) == cigar.read_len
            consensus_alns[sample_id][copy_i] = (consensus_seq, alignment.Alignment(cigar, region, strand=True))

    keep_regions = _reconstruct_regions(endpoints, region, consider_ref=False)
    return keep_regions, consensus_alns


def _realign_consensus_alns(dupl, consensus_alns, weights):
    new_alns = []
    for sample_alns in consensus_alns:
        sample_new_alns = []
        for seq, aln in sample_alns:
            new_seq, new_aln = dupl.align_read(seq, aln, weights, full_cigar=True)
            assert new_aln.ref_interval == dupl.region1
            sample_new_alns.append((new_seq, new_aln))
        new_alns.append(sample_new_alns)
    return new_alns


def _create_vcf_header(in_header, genome, chrom_id):
    header = psvs_.create_vcf_header(genome, argv=sys.argv)
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.add_line('##FORMAT=<ID=cn,Number=.,Type=Integer,Description="Copy number of each copy. Sum = genotype size. '
        'Order = [chrom:pos] + pos2.">')
    for sample in in_header.samples:
        header.add_sample(sample)
    return header


def _set_pos2(record, alleles, dupl_ixs, duplications, genome):
    need_pos2_update = True
    while need_pos2_update:
        pos2 = []
        need_pos2_update = False
        for dupl_i in dupl_ixs:
            if dupl_i != -1:
                try:
                    var_pos2 = duplications[dupl_i].align_variant(record)
                except errors.VariantOutOfBounds:
                    var_pos2 = None
                if var_pos2 is None:
                    pos2.append('???')
                    continue

                allele = var_pos2.sequence
                if allele in alleles:
                    allele_ix = alleles.index(allele)
                else:
                    allele_ix = len(alleles)
                    alleles.append(allele)
                    # Need to update all pos2 as we added a new allele.
                    need_pos2_update = True
                    record.alleles = tuple(alleles)
                    break
                pos2.append('{}:{}:{}:{}'.format(var_pos2.region.chrom_name(genome), var_pos2.region.start_1,
                    '+' if var_pos2.strand else '-', allele_ix))
    record.info['pos2'] = pos2


def _pre_psv_to_variant(pre_psv, header, interval, interval_seq, duplications, genome, all_cons_alns):
    record = header.new_record()
    record.chrom = interval.chrom_name(genome)
    record.start = pre_psv.start
    assert pre_psv.start >= interval.start and pre_psv.end <= interval.end
    alleles = [interval_seq[pre_psv.start - interval.start : pre_psv.end - interval.end]]

    n_samples = len(header.samples)
    active_dupl = sorted(pre_psv.active_dupl)
    dupl_ixs = sorted(set(map(operator.itemgetter(1), active_dupl)))
    n_dupl = len(dupl_ixs)

    sample_gts = [[] for _ in range(n_samples)]
    sample_keys = [[] for _ in range(n_samples)]
    sample_cns = [[0] * n_dupl for _ in range(n_samples)]
    for sample_id, dupl_i, aln_i in active_dupl:
        sample_gts[sample_id].append(0)
        sample_keys[sample_id].append((dupl_i, aln_i))
        sample_cns[sample_id][dupl_ixs.index(dupl_i)] += 1

    for reg2 in pre_psv.reg2:
        sample_id, dupl_i, aln_i = reg2.dupl_index
        cons_seq = all_cons_alns[dupl_i + 1][sample_id][aln_i][0]
        allele = cons_seq[reg2.start : reg2.end]
        if allele in alleles:
            allele_ix = alleles.index(allele)
            assert allele_ix > 0
        else:
            allele_ix = len(alleles)
            alleles.append(allele)
        haplotype = sample_keys[sample_id].index((dupl_i, aln_i))
        sample_gts[sample_id][haplotype] = allele_ix

    if all(map(bool, alleles)):
        record.alleles = tuple(alleles)
        _set_pos2(record, alleles, dupl_ixs, duplications, genome)
    else:
        # There is an empty allele -- meaning that the variant overlaps a boundary of the duplication.
        record.alleles = tuple(allele or '???' for allele in alleles)
        record.info['pos2'] = ('???',)

    for sample_id in range(n_samples):
        record.samples[sample_id]['GT'] = sample_gts[sample_id]
        record.samples[sample_id]['cn'] = sample_cns[sample_id]
        record.samples[sample_id].phased = True
    return record


def _consensus_alns_to_variants(interval, interval_seq, genome, duplications, all_cons_alns, out_header):
    tangled_searcher = itree.NonOverlTree.empty()
    aligned_pos = []
    psvs = []
    duplication_starts1 = {}

    for dupl_i, dupl_cons_alns in enumerate(all_cons_alns, -1):
        if dupl_i == -1:
            reg1 = interval
            reg1_seq = interval_seq
        else:
            reg1 = duplications[dupl_i].region1
            reg1_seq = duplications[dupl_i].seq1

        for sample_id, sample_alns in enumerate(dupl_cons_alns):
            for aln_i, (cons_seq, cons_aln) in enumerate(sample_alns):
                dupl_key = (sample_id, dupl_i, aln_i)
                assert cons_aln.ref_interval == reg1
                duplication_starts1[dupl_key] = reg1.start
                psvs_.duplication_differences(reg1, reg1_seq, cons_seq, cons_aln.cigar, dupl_key, psvs, aligned_pos,
                    in_region=interval, tangled_searcher=tangled_searcher)

    records = []
    for pre_psv in psvs_.combine_psvs(psvs, aligned_pos, duplication_starts1):
        records.append(_pre_psv_to_variant(pre_psv, out_header, interval, interval_seq, duplications, genome,
            all_cons_alns))
    return records


def pool(vcf_file, out_filename, interval, duplications, genome, limit_tree, unphased_dist):
    weights = alignment.Weights()
    parasail_args = weights.parasail_args()

    interval_seq = interval.get_sequence(genome)
    endpoints = []
    endpoints.append((interval.start, True, _Flag.Ref))
    endpoints.append((interval.end, False, _Flag.Ref))

    all_cons_alns = []
    # common.log('Main copy: {}'.format(interval.to_str_comma(genome)))
    keep_regions, cons_alns = _create_consensus_alns(vcf_file, interval, interval_seq, genome, parasail_args,
        limit_tree, unphased_dist)
    all_cons_alns.append(cons_alns)
    for region in keep_regions:
        endpoints.append((region.start, True, _Flag.Keep))
        endpoints.append((region.end, False, _Flag.Keep))

    for dupl in duplications:
        # common.log('Duplication: {}'.format(dupl.to_str_pretty(genome)))
        region2_seq = common.cond_rev_comp(dupl.seq2, strand=dupl.strand)
        keep_regions, cons_alns = _create_consensus_alns(vcf_file, dupl.region2, region2_seq, genome, parasail_args,
            limit_tree, unphased_dist)
        endpoints.append((dupl.region1.start, True, _Flag.Ref))
        endpoints.append((dupl.region1.end, False, _Flag.Ref))
        for region2 in keep_regions:
            region1 = dupl.subregion1(region2.start, region2.end)
            if region1 is not None:
                endpoints.append((region1.start, True, _Flag.Keep))
                endpoints.append((region1.end, False, _Flag.Keep))
        all_cons_alns.append(_realign_consensus_alns(dupl, cons_alns, weights))

    out_header = _create_vcf_header(vcf_file.header, genome, interval.chrom_id)
    records = _consensus_alns_to_variants(interval, interval_seq, genome, duplications, all_cons_alns, out_header)
    variants_.open_and_write_vcf(out_filename, out_header, records, 'tabix')
    return _reconstruct_regions(endpoints, interval, consider_ref=True)


def _load_limit_regions(filename, genome):
    if filename is None:
        return None

    with common.open_possible_gzip(filename) as inp:
        intervals = []
        for line in inp:
            if line.startswith('#'):
                continue
            line = line.strip().split('\t')
            intervals.append(Interval(genome.chrom_id(line[0]), int(line[1]), int(line[2])))
    intervals.sort()
    intervals = Interval.combine_overlapping(intervals)
    return itree.MultiNonOverlTree(intervals)


def _filter_variant_regions(records, genome, keep_regions, max_len):
    endpoints = defaultdict(list)
    for region in keep_regions:
        chrom_id = region.chrom_id
        endpoints[chrom_id].append((region.start, True, _Flag.Keep))
        endpoints[chrom_id].append((region.end, False, _Flag.Keep))

    for record in records:
        if max(map(len, record.alleles)) > max_len or 'GT' not in record.samples[0] or '???' in record.info['pos2']:
            chrom_id = genome.chrom_id(record.chrom)
            endpoints[chrom_id].append((record.start - PADDING, True, _Flag.Discard))
            endpoints[chrom_id].append((record.start + len(record.ref) + PADDING, False, _Flag.Discard))

    res_regions = []
    for chrom_id in sorted(endpoints.keys()):
        full_interval = genome.chrom_interval(chrom_id, named=False)
        res_regions.extend(_reconstruct_regions(endpoints[chrom_id], full_interval, consider_ref=False))
    return res_regions


def process(vcf_file, genome, table, args):
    limit_tree = _load_limit_regions(args.limit_regions, genome)
    wdir_context = tempfile.TemporaryDirectory(prefix='parascopy') if args.tmp_dir is None \
        else common.EmptyContextManager()
    with wdir_context as wdir:
        if wdir is None:
            wdir = args.tmp_dir
            common.mkdir(wdir)
        common.log('Using temporary directory {}'.format(wdir))

        tmp_filenames = []
        keep_regions = []
        for region in common.get_regions(args, genome):
            common.log('Analyzing {}'.format(region.full_name(genome)))
            duplications = pool_reads.load_duplications(table, genome, region, args.exclude)

            tmp_filename = os.path.join(wdir, '{}.vcf.gz'.format(region.os_name))
            curr_regions = pool(vcf_file, tmp_filename, region, duplications, genome, limit_tree, args.unphased_dist)
            keep_regions.extend(curr_regions)
            tmp_filenames.append(tmp_filename)
        res_records = detect_cn.join_vcf_files(tmp_filenames, args.output[0], genome, 'tabix',
            merge_duplicates=True, in_filenames=True)

    keep_regions = _filter_variant_regions(res_records, genome, keep_regions, args.max_len)
    with common.open_possible_gzip(args.output[1], 'w') as out:
        for region in Interval.combine_overlapping(keep_regions):
            out.write(region.to_bed(genome) + '\n')


def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Pool VCF records from various copies of a duplication.',
        formatter_class=common.SingleMetavar, add_help=False,
        usage='{} -i <vcf> -t <table> -f <fasta> (-r <region> [...] | -R <bed>) -o <vcf> <bed>'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-i', '--input', metavar='<file>', required=True,
        help='Input VCF file.')
    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed table with information about segmental duplications.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<vcf> <bed>', required=True, nargs=2,
        help='Output VCF and BED files.')

    reg_args = parser.add_argument_group('Region arguments')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region> [<region> ...]',
        help='Region(s) in format "chr" or "chr:start-end".\n'
            'Start and end are 1-based inclusive. Commas are ignored.\n'
            'Optionally, you can provide region name using the format "region::name"\n'
            'For example "-r chr5:70,925,087-70,953,015::SMN1".')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file> [<file> ...]',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).\n'
            'Optional fourth column will be used as a region name.')
    reg_args.add_argument('-l', '--limit-regions', metavar='<file>', required=False,
        help='Optional: limit input VCF file to these regions.')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>', default='length < 500 && seq_sim < 0.97',
        help='Exclude duplications for which the expression is true\n[default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-d', '--unphased-dist', metavar='<int>', type=int, default=10,
        help='Allow unphased variants if they are over <int> bp\n'
            'from other heterozygous variants [default: %(default)s].')
    opt_args.add_argument('-m', '--max-len', metavar='<int>', type=int, default=30,
        help='Discard regions that cover variants longer than <int> bp [default: %(default)s].')
    opt_args.add_argument('--tmp-dir', metavar='<dir>',
        help='Put temporary files in the following directory (does not remove temporary\n'
            'files after finishing). Otherwise, creates a temporary directory.')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    # Touch output files to make sure they can be created
    pathlib.Path(args.output[0]).touch()
    pathlib.Path(args.output[1]).touch()

    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.table, parser=pysam.asTuple()) as table, \
            pysam.VariantFile(args.input) as vcf_file:
        if vcf_file.index is None:
            raise ValueError('Input vcf file "{}" has no index.'.format(args.input))
        process(vcf_file, genome, table, args)
    common.log('Success!')


if __name__ == '__main__':
    main()
