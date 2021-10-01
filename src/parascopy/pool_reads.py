#!/usr/bin/env python3

import argparse
import os
import pysam
import numpy as np
from collections import defaultdict
from enum import Enum

from .inner import common
from .inner import duplication as duplication_
from .inner.genome import Genome, Interval
from .inner.alignment import Alignment, Weights
from .inner import bam_file as bam_file_
from .view import parse_expression
from . import long_version


_MAX_INSERT_SIZE = 50000


def _create_record(name, seq, qual, start, cigar_tuples, header, old_record, read_groups, status):
    record = pysam.AlignedSegment(header)
    record.query_name = name
    record.query_sequence = seq
    record.query_qualities = qual
    if cigar_tuples:
        record.reference_id = 0
        record.reference_start = start
        record.mapping_quality = 60
        record.cigartuples = cigar_tuples
    else:
        record.is_unmapped = True

    read_group = old_record.get_tag('RG') if old_record.has_tag('RG') else None
    new_read_group = read_groups[read_group][0]
    record.set_tag('RG', new_read_group)

    if old_record.is_paired:
        record.is_paired = True
        if old_record.is_read1:
            record.is_read1 = True
        else:
            record.is_read2 = True

        if old_record.is_proper_pair and old_record.reference_id == old_record.next_reference_id \
                and abs(old_record.reference_start - old_record.next_reference_start) <= _MAX_INSERT_SIZE:
            record.is_proper_pair = True

    oa_tag = '{},{},{},{},{},{};'.format(old_record.reference_name, old_record.reference_start + 1,
        '-' if old_record.is_reverse else '+', old_record.cigarstring, old_record.mapping_quality,
        old_record.get_tag('NM') if old_record.has_tag('NM') else '')
    record.set_tag('OA', oa_tag)
    record.set_tag('st', status.value)
    return record


def _create_header(genome, chrom_id, read_groups, max_mate_dist):
    header = '@SQ\tSN:{}\tLN:{}\n'.format(genome.chrom_name(chrom_id), genome.chrom_len(chrom_id))
    for curr_read_groups in read_groups:
        for group_id, sample in curr_read_groups.values():
            assert isinstance(sample, str)
            header += '@RG\tID:{}\tSM:{}\n'.format(group_id, sample)
    header += '@CO\tmax_mate_dist={}\n'.format(max_mate_dist)
    return pysam.AlignmentHeader.from_text(header)


def _extract_reads(in_bam, out_bam, read_groups, region, genome, out_header, read_positions, max_mate_dist):
    for record in bam_file_.fetch(in_bam, region, genome):
        if record.flag & 3844:
            continue

        curr_read_pos = read_positions[record.query_name]
        assert curr_read_pos.add_realigned_pos(record.is_read2, record.reference_start)
        curr_read_pos.record_mate_pos(record, genome, max_mate_dist)
        new_rec = _create_record(record.query_name, record.query_sequence, record.query_qualities,
            record.reference_start, record.cigartuples, out_header, record, read_groups, bam_file_.ReadStatus.SameLoc)
        out_bam.write(new_rec)


class _ReadPositions:
    MAX_REALIGNED_DIST = 100

    def __init__(self):
        self.realigned_pos = ([], [])
        self.mate_written = [False, False]
        self.mate_position = None

    def add_realigned_pos(self, is_read2, new_pos):
        positions = self.realigned_pos[is_read2]
        if any(abs(prev_pos - new_pos) < _ReadPositions.MAX_REALIGNED_DIST for prev_pos in positions):
            return False
        positions.append(new_pos)
        return True

    def record_mate_pos(self, record, genome, max_mate_dist):
        is_read2 = record.is_read2
        self.mate_written[is_read2] = True
        mate_ix = 1 - is_read2
        if self.mate_written[mate_ix] or not record.is_proper_pair or max_mate_dist == 0:
            return

        if self.mate_position is not None:
            assert self.mate_position[0] == mate_ix
            return

        mate_same_chrom = record.reference_id == record.next_reference_id
        if not mate_same_chrom and np.isfinite(max_mate_dist):
            return

        mate_pos = record.next_reference_start
        if abs(record.reference_start - mate_pos) <= max_mate_dist:
            self.mate_position = (mate_ix, genome.chrom_id(record.next_reference_name), mate_pos)

    def mate_pos(self):
        if self.mate_position is None or self.mate_written[self.mate_position[0]]:
            return None
        return self.mate_position


def _extract_reads_and_realign(in_bam, out_bam, read_groups, dupl, genome, out_header, weights,
        read_positions, max_mate_dist):
    """
    Load reads from dupl.region2 and aligns them to dupl.region1.

    read_positions: dictionary, key = read_name, values = _ReadPositions.
    """
    for record in bam_file_.fetch(in_bam, dupl.region2, genome):
        if record.flag & 3844:
            continue

        orig_aln = Alignment.from_record(record, genome)
        reg1_aln = dupl.align_read(record.query_sequence, orig_aln, weights, calc_score=False)

        new_pos = reg1_aln.ref_interval.start
        if not read_positions[record.query_name].add_realigned_pos(record.is_read2, new_pos):
            continue

        same_strand = reg1_aln.strand == orig_aln.strand
        cigar_tuples = reg1_aln.cigar.to_pysam_tuples() if reg1_aln.cigar is not None else None
        new_rec = _create_record(record.query_name,
            common.cond_rev_comp(record.query_sequence, strand=same_strand),
            common.cond_reverse(record.query_qualities, strand=same_strand),
            new_pos, cigar_tuples, out_header, record, read_groups, bam_file_.ReadStatus.Realigned)
        read_positions[record.query_name].record_mate_pos(record, genome, max_mate_dist)
        out_bam.write(new_rec)


def _get_fetch_regions(fetch_positions, max_dist=100):
    fetch_positions.sort()
    fetch_regions = []

    start_chrom, start_pos = fetch_positions[0]
    for i in range(1, len(fetch_positions)):
        prev_chrom, prev_pos = fetch_positions[i - 1]
        curr_chrom, curr_pos = fetch_positions[i]
        if prev_chrom != curr_chrom or curr_pos - prev_pos > max_dist:
            fetch_regions.append(Interval(start_chrom, start_pos, prev_pos + 1))
            start_chrom = curr_chrom
            start_pos = curr_pos

    last_chrom, last_pos = fetch_positions[-1]
    assert start_chrom == last_chrom
    fetch_regions.append(Interval(start_chrom, start_pos, last_pos + 1))
    return fetch_regions


def _write_mates(in_bam, out_bam, read_positions, genome, out_header, read_groups):
    fetch_positions = []
    mate_names = set()

    for read_name, read_pos in read_positions.items():
        mate_pos = read_pos.mate_pos()
        if mate_pos is None:
            continue

        is_read2, chrom_id, pos = mate_pos
        fetch_positions.append((chrom_id, pos))
        mate_names.add((read_name, is_read2))

    if not fetch_positions:
        return

    for region in _get_fetch_regions(fetch_positions):
        for record in bam_file_.fetch(in_bam, region, genome):
            if record.flag & 3844:
                continue
            key = (record.query_name, record.is_read2)
            if key not in mate_names:
                continue
            mate_names.remove(key)

            new_rec = _create_record(record.query_name, record.query_sequence, record.query_qualities,
                record.reference_start, record.cigartuples, out_header, record, read_groups,
                bam_file_.ReadStatus.ReadMate)
            out_bam.write(new_rec)


_MATE_DISTANCE = 5000

def pool_reads_inner(bam_files, read_groups, out_path, interval, duplications, genome, weights, *,
        samtools, max_mate_dist=_MATE_DISTANCE, verbose=True, time_log=None):
    out_header = _create_header(genome, interval.chrom_id, read_groups, max_mate_dist)

    if verbose:
        common.log('Extracting and realigning reads')

    if time_log is not None:
        time_log.log('Pooling reads')
    tmp_path = out_path + '.tmp'
    with pysam.AlignmentFile(tmp_path, 'wb', header=out_header) as tmp_bam:
        for bam_file_num, (bam_file, curr_read_groups) in enumerate(zip(bam_files, read_groups), start=1):
            if verbose:
                common.log('    Extracting reads from {:3d}: {}'.format(bam_file_num, bam_file.filename.decode()))
            read_positions = defaultdict(_ReadPositions)

            _extract_reads(bam_file, tmp_bam, curr_read_groups, interval, genome, out_header,
                read_positions, max_mate_dist)
            for dupl in duplications:
                _extract_reads_and_realign(bam_file, tmp_bam, curr_read_groups, dupl, genome,
                    out_header, weights, read_positions, max_mate_dist)
            if max_mate_dist != 0:
                _write_mates(bam_file, tmp_bam, read_positions, genome, out_header, curr_read_groups)

    if verbose:
        common.log('Sorting pooled reads')
    if time_log is not None:
        time_log.log('Sorting pooled reads')
    if samtools == 'none' or samtools is None:
        pysam.sort('-o', out_path, tmp_path)
    else:
        common.Process([samtools, 'sort', '-o', out_path, tmp_path]).finish(zero_code_stderr=False)
    os.remove(tmp_path)

    if time_log is not None:
        time_log.log('Indexing pooled reads')
    if samtools == 'none' or samtools is None:
        pysam.index(out_path)
    else:
        common.Process([samtools, 'index', out_path]).finish(zero_code_stderr=False)


def pool_reads(bam_files, read_groups, out_path, interval, table, genome, args):
    weights = Weights()
    duplications = []
    exclude_dupl = parse_expression(args.exclude)

    for tup in table.fetch(interval.chrom_name(genome), interval.start, interval.end):
        dupl = duplication_.Duplication.from_tuple(tup, genome)
        if dupl.is_tangled_region or exclude_dupl(dupl, genome):
            continue
        dupl.set_cigar_from_info()
        dupl = dupl.sub_duplication(interval)
        dupl.set_sequences(genome=genome)
        dupl.set_padding_sequences(genome, 200)
        duplications.append(dupl)

    pool_reads_inner(bam_files, read_groups, out_path, interval, duplications, genome, weights,
        samtools=args.samtools, max_mate_dist=args.mate_dist, verbose=args.verbose)


def load_bam_files(input, input_list, genome, allow_unnamed=False):
    """
    Loads BAM files from either input or input-list.
    Returns two lists with the same length and other:
        - bam_files: list of pysam.AlignmentFile,
        - read_groups: list of dictionaries { old_read_group : (new_read_group, sample_name) }.
            Length of the dictionary is the same as the length of bam_files.
    """
    # List of tuples (filename, sample).
    FILENAME_SPLIT = '::'

    filenames = []
    if input:
        for filename in input:
            if FILENAME_SPLIT in filename:
                filenames.append(tuple(filename.split(FILENAME_SPLIT, 1)))
            else:
                filenames.append((filename, None))
    else:
        list_dir = os.path.dirname(input_list)
        with open(input_list) as inp:
            try:
                for line in inp:
                    line = line.strip().split(maxsplit=1)
                    if len(line) == 1:
                        filename = line[0]
                        sample = None
                    else:
                        filename, sample = line
                    filenames.append((os.path.join(list_dir, filename), sample))
            except UnicodeDecodeError:
                raise ValueError('Cannot read input list -I {0}, perhaps you want to use -i {0}?'.format(input_list))

    bam_files = []
    read_groups = []
    for bam_file_num, (filename, sample) in enumerate(filenames):
        bam_file = pysam.AlignmentFile(filename, reference_filename=genome.filename, require_index=True)
        old_read_groups = bam_file_.get_read_groups(bam_file)
        new_read_groups = {}

        if sample is not None:
            # Associate reads without read_group with sample from the input file.
            new_read_groups[None] = ('__{}'.format(bam_file_num), sample)
        elif allow_unnamed:
            # Otherwise, use Unnamed-X for all reads without read group.
            unnamed_sample = 'Unnamed-{}'.format(bam_file_num + 1)
            new_read_groups[None] = ('__{}'.format(bam_file_num), unnamed_sample)

        for old_read_group, old_sample in old_read_groups:
            new_read_group = '{}-{}'.format(old_read_group, bam_file_num)
            # If BAM file is associated with sample name, all reads should have it. Otherwise, old sample name is kept.
            new_sample = sample or old_sample
            new_read_groups[old_read_group] = (new_read_group, new_sample)

        if not new_read_groups:
            common.log('ERROR: Input file {} has no read groups in the header.'.format(filename))
            common.log('Please specify sample name in "-I input-list.txt"')
            exit(1)

        bam_files.append(bam_file)
        read_groups.append(new_read_groups)
    return bam_files, read_groups


def main(prog_name=None, in_args=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Pool reads from various copies of a given homologous region.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} (-i <bam> [...] | -I <bam-list>) -t <table> -f <fasta> -r <region> -o <bam>'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')

    inp_me = io_args.add_mutually_exclusive_group(required=True)
    inp_me.add_argument('-i', '--input', metavar='<file>', nargs='+',
        help='Input BAM/CRAM files. All reads should have read groups with sample name.\n'
            'Mutually exclusive with --input-list.')
    inp_me.add_argument('-I', '--input-list', metavar='<file>',
        help='A file containing a list of BAM/CRAM files to analyze. Each line should follow\n'
        'format "path[ sample]". If sample is non-empty, all reads will use this sample name.\n'
        'If sample is empty, original read groups will be used.\n'
        'Raises an error if sample is empty and BAM/CRAM file does not have read groups.\n'
        'Mutually exclusive with --input.\n\n')

    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed table with information about segmental duplications.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True)
    io_args.add_argument('-r', '--region', metavar='<region>',
        help='Single region in format "chr:start-end". Start and end are 1-based inclusive.\nCommas are ignored.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output BAM file.')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>',
        default='length < 500 && seq_sim < 0.97',
        help='Exclude duplications for which the expression is true\n[default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-m', '--mate-dist', metavar='<int>|infinity', type=float, default=_MATE_DISTANCE,
        help='Output read mates even if they are outside of the duplication,\n'
            'if the distance between mates is less than <int> [default: %(default)s].\n'
            'Use 0 to skip all mates outside the duplicated regions.\n'
            'Use inf|infinity to write all mapped read mates.\n')
    opt_args.add_argument('-q', '--quiet', action='store_false', dest='verbose',
        help='Do not write information to the stderr.')
    opt_args.add_argument('--samtools', metavar='<path>|none', default='samtools',
            help='Path to samtools executable [default: %(default)s].\n'
                'Use python wrapper if "none", can lead to errors.')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_args)

    if args.samtools != 'none':
        common.check_executable(args.samtools)

    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.table, parser=pysam.asTuple()) as table:
        interval = Interval.parse(args.region, genome)
        bam_files, read_groups = load_bam_files(args.input, args.input_list, genome, allow_unnamed=False)

        pool_reads(bam_files, read_groups, args.output, interval, table, genome, args)
        for bam_file in bam_files:
            bam_file.close()


if __name__ == '__main__':
    main()
