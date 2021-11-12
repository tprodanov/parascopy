#!/usr/bin/env python3

import argparse
import pysam
import collections
import sys
from datetime import datetime

from .view import parse_expression
from .inner import common
from .inner.genome import Genome, Interval
from .inner.duplication import Duplication
from . import long_version


_AlignedRegion = collections.namedtuple('_AlignedRegion', 'region1 region2 aligned_seq2 strand')


def _msa_from_pairs(region1, seq1, aligned_regions):
    lengths = [1] * len(seq1)
    conservation = ['*'] * len(seq1)
    seqs2 = []
    for al_region in aligned_regions:
        shift1 = al_region.region1.start - region1.start
        seq2 = ['-'] * shift1
        for i, subseq2 in enumerate(al_region.aligned_seq2):
            seq2.append(subseq2)
            if subseq2[0] != seq1[shift1 + i]:
                conservation[shift1 + i] = ' '
            if len(subseq2) > 1:
                lengths[shift1 + i] = max(lengths[shift1 + i], len(subseq2))
        seq2.extend('-' * (region1.end - al_region.region1.end))
        seqs2.append(seq2)

    res = []
    res_seq1 = ''
    for i, (nt, length) in enumerate(zip(seq1, lengths)):
        res_seq1 += nt
        if length > 1:
            res_seq1 += '-' * (length - 1)
            conservation[i] += ' ' * (length - 1)
    res.append(res_seq1)

    for seq2 in seqs2:
        res_seq2 = ''
        for i, (subseq2, length) in enumerate(zip(seq2, lengths)):
            res_seq2 += subseq2
            if len(subseq2) < length:
                res_seq2 += '-' * (length - len(subseq2))
        res.append(res_seq2)
    res.append(''.join(conservation))
    l = sum(lengths)
    assert all(len(seq) == l for seq in res)
    return res


def _write_aligned_regions_csv(region, region_seq, aligned_regions, genome, outp):
    outp.write('# %s\n' % ' '.join(sys.argv))
    outp.write('chrom2\tpos2\tchrom1\tpos1\tnt1\tnt2\n')

    for al_region in aligned_regions:
        if al_region is None:
            continue

        chrom1 = al_region.region1.chrom_name(genome)
        start1 = al_region.region1.start + 1
        seq1 = region_seq[al_region.region1.start - region.start : al_region.region1.end - region.start]

        chrom2 = al_region.region2.chrom_name(genome)
        pos2 = al_region.region2.start + 1 if al_region.strand else -al_region.region2.end

        for i, subseq2 in enumerate(al_region.aligned_seq2):
            if subseq2 == '-':
                outp.write('NA\tNA\t%s\t%d\t%s\tNA\n' % (chrom1, start1 + i, seq1[i]))
                continue
            outp.write('%s\t%d\t%s\t%d\t%s\t%s\n' % (chrom2, abs(pos2), chrom1, start1 + i, seq1[i], subseq2[0]))

            if len(subseq2) > 1:
                summand = 1 / len(subseq2)
                for j in range(1, len(subseq2)):
                    outp.write('%s\t%d\t%s\t%.2f\tNA\t%s\n'
                        % (chrom2, abs(pos2 + j), chrom1, start1 + i + j * summand, subseq2[j]))
            pos2 += len(subseq2)


def _al_region_key(al_region):
    if al_region is not None:
        return (0, al_region.region1.start, al_region.region2)
    return (1,)


def _write_msa(region, region_seq, duplications, aligned_regions, genome, outp, true_clustal, width):
    name_counter = collections.Counter()
    names = ['%s:%d' % (region.chrom_name(genome), region.start_1)]
    name_counter[names[0]] += 1
    for al_region in aligned_regions:
        if al_region is None:
            names.append('-')
            continue

        name = '%s:%d' % (al_region.region2.chrom_name(genome), al_region.region2.start_1)
        prev_count = name_counter[name]
        name_counter[name] += 1
        if prev_count:
            name += '_%s' % common.string_suffix(prev_count)
        names.append(name)
    name_len = max(map(len, names))
    padded_names = [name + ' ' * (name_len - len(name)) for name in names]

    outp.write('#=======================================\n#\n')
    outp.write('# Aligned sequences: %d\n' % (len(duplications) + 1))
    outp.write('# 1: %s\n' % names[0])
    for i, (dupl, al_region) in enumerate(zip(duplications, aligned_regions)):
        if al_region is None:
            outp.write('# !!! Region of interest within deletion (not shown).')
        else:
            outp.write('# {}: {}. Aligned region: {}, {}.'.format(i + 2, names[i + 1],
                al_region.region1.to_str(genome), al_region.region2.to_str(genome)))
        outp.write('    Full duplication: {reg1}, {reg2}, {strand} strand; aligned length: {alen}; '
            'differences: {diff}; sequence similarity: {seq_sim}; complexity: {compl}\n'
            .format(reg1=dupl.region1.to_str(genome), reg2=dupl.region2.to_str(genome), strand=dupl.strand_str,
            alen=dupl.info.get('ALENGTH'), seq_sim=dupl.info.get('SS'), diff=dupl.info.get('DIFF'),
            compl=dupl.info.get('compl')))
    outp.write('#\n#=======================================\n\n')

    aligned_regions = list(filter(bool, aligned_regions))
    pos1 = region.start
    positions = []
    for al_region in aligned_regions:
        positions.append(al_region.region2.start if al_region.strand else -al_region.region2.end - 1)
    start_positions = list(positions)

    msa_seqs = _msa_from_pairs(region, region_seq, aligned_regions)
    for i in range(0, len(msa_seqs[0]), width):
        subseq1 = msa_seqs[0][i : i + width]
        new_pos1 = pos1 + len(subseq1) - subseq1.count('-')
        if true_clustal:
            outp.write('{}    {} {}\n'.format(padded_names[0], subseq1, new_pos1 - region.start))
        else:
            outp.write('{}    {} {:,}\n'.format(padded_names[0], subseq1, new_pos1))

        more_than_one = False
        for j, (msa_seq2, al_region) in enumerate(zip(msa_seqs[1:-1], aligned_regions)):
            if true_clustal or (pos1 < al_region.region1.end and new_pos1 > al_region.region1.start):
                more_than_one = True
                subseq2 = msa_seq2[i : i + width]
                positions[j] += len(subseq2) - subseq2.count('-')
                if true_clustal:
                    outp.write('{}    {} {}\n'.format(padded_names[j + 1], subseq2, positions[j] - start_positions[j]))
                else:
                    outp.write('{}    {} {:,}\n'.format(padded_names[j + 1], subseq2, abs(positions[j])))
        pos1 = new_pos1
        if more_than_one:
            outp.write('{}    {}\n'.format(' ' * name_len, msa_seqs[-1][i : i + width]))
        outp.write('\n')


def construct_msa(region, table, genome, excl_dupl, outp, outp_csv, true_clustal, width):
    region.trim(genome)
    duplications = []
    for tup in table.fetch(region.chrom_name(genome), region.start, region.end):
        dupl = Duplication.from_tuple(tup, genome)
        if not dupl.is_tangled_region and not excl_dupl(dupl, genome):
            dupl.set_cigar_from_info()
            dupl.set_sequences(genome=genome)
            duplications.append(dupl)

    aligned_regions = []
    region_len = len(region)
    for dupl in duplications:
        shift1 = dupl.region1.start - region.start
        seq2 = dupl.seq2
        aligned_seq2 = []

        start = None
        # expected position 1, expected position 2.
        exp1 = None
        exp2 = None
        for pos2, pos1 in dupl.full_cigar.aligned_pairs(ref_start=max(0, -shift1)):
            if start is None:
                start = (pos1, pos2)
            if pos1 + shift1 >= region_len:
                break
            if exp1 is not None and (exp1 != pos1 or exp2 != pos2):
                for _ in range(exp1, pos1):
                    aligned_seq2.append('-')
                aligned_seq2[-1] += seq2[exp2:pos2]

            aligned_seq2.append(seq2[pos2])
            exp1 = pos1 + 1
            exp2 = pos2 + 1

        if exp1 is None:
            aligned_regions.append(None)
            continue

        # end = (exp1, exp2)
        al_region1 = Interval(dupl.region1.chrom_id, dupl.region1.start + start[0], dupl.region1.start + exp1)
        if dupl.strand:
            al_region2 = Interval(dupl.region2.chrom_id, dupl.region2.start + start[1], dupl.region2.start + exp2)
        else:
            al_region2 = Interval(dupl.region2.chrom_id, dupl.region2.end - exp2, dupl.region2.end - start[1])
        aligned_regions.append(_AlignedRegion(al_region1, al_region2, aligned_seq2, dupl.strand))
    ixs = sorted(range(len(aligned_regions)), key=lambda i: _al_region_key(aligned_regions[i]))
    aligned_regions = [aligned_regions[i] for i in ixs]
    duplications = [duplications[i] for i in ixs]

    region_seq = region.get_sequence(genome)
    if outp_csv:
        _write_aligned_regions_csv(region, region_seq, aligned_regions, genome, outp_csv)
    _write_msa(region, region_seq, duplications, aligned_regions, genome, outp, true_clustal, width)


def main(prog_name=None, in_args=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Visualize multiple sequence alignment of homologous regions.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -i <table> -f <fasta> (-r <region> | -R <bed>) [-o <clustal>] [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-i', '--input', metavar='<file>', required=True,
        help='Input indexed bed.gz homology table.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=False,
        help='Optional: output in clustal format.')
    io_args.add_argument('-O', '--out-csv', metavar='<file>', required=False,
        help='Optional: output csv file with aligned positions.')

    reg_args = parser.add_argument_group('Region arguments (at least one is required)')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file>',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>', default='length < 500',
        help='Exclude duplications for which the expression is true [default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-c', '--true-clustal', action='store_true',
        help='Outputs true clustal format: writes gaps outside the boundary of duplication,\n'
            'writes number of bases instead of genomic position.')
    opt_args.add_argument('-w', '--width', type=int, metavar='<int>', default=60,
        help='Number of basepairs per line [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_args)

    with Genome(args.fasta_ref) as genome, \
            pysam.TabixFile(args.input, parser=pysam.asTuple()) as table, \
            common.open_possible_gzip(args.output, 'w') as outp, \
            common.open_possible_empty(args.out_csv, 'w') as outp_csv:

        outp.write(
'''CLUSTAL W
########################################
# Program:  homologytools msa
# Rundate:  %s
# Command:  %s
########################################\n''' % (datetime.now().strftime('%b %d %Y %H:%M:%S'), ' '.join(sys.argv)))

        excl_dupl = parse_expression(args.exclude)
        for region in common.get_regions(args, genome, only_unique=False):
            construct_msa(region, table, genome, excl_dupl, outp, outp_csv, args.true_clustal, args.width)


if __name__ == '__main__':
    main()
