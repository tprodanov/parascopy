#!/usr/bin/env python3

import argparse
import os
import operator
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


UNDEF = common.UNDEF


def _create_record(orig_record, header, read_groups, status, max_mate_dist,
        *, dupl_strand=True, start=UNDEF, cigar_tuples=UNDEF, seq=UNDEF, qual=UNDEF):
    """
    Creates a new record by taking the orig_record as a template.
    If start, cigar_tuples, seq, qual are not provided, take them frmo the original record.
    """
    record = pysam.AlignedSegment(header)
    record.query_name = orig_record.query_name
    record.query_sequence = common.cond_rev_comp(orig_record.query_sequence, strand=dupl_strand) \
        if seq is UNDEF else seq
    record.query_qualities = common.cond_reverse(orig_record.query_qualities, strand=dupl_strand) \
        if qual is UNDEF else qual

    if cigar_tuples is UNDEF:
        assert dupl_strand
        cigar_tuples = orig_record.cigartuples
    # This is either input cigar_tuples, or orig_record.cigartuples.
    if cigar_tuples:
        record.reference_id = 0
        record.reference_start = orig_record.reference_start if start is UNDEF else start
        record.mapping_quality = 60
        record.cigartuples = cigar_tuples
        if orig_record.is_reverse == dupl_strand:
            record.is_reverse = True
    else:
        record.is_unmapped = True

    read_group = orig_record.get_tag('RG') if orig_record.has_tag('RG') else None
    new_read_group = read_groups[read_group][0]
    record.set_tag('RG', new_read_group)

    if orig_record.is_paired:
        record.is_paired = True
        if orig_record.is_read1:
            record.is_read1 = True
        else:
            record.is_read2 = True

        if orig_record.is_proper_pair and orig_record.reference_id == orig_record.next_reference_id \
                and abs(orig_record.reference_start - orig_record.next_reference_start) <= max_mate_dist:
            record.is_proper_pair = True

    oa_tag = '{},{},{},{},{},{};'.format(orig_record.reference_name, orig_record.reference_start + 1,
        '-' if orig_record.is_reverse else '+', orig_record.cigarstring, orig_record.mapping_quality,
        orig_record.get_tag('NM') if orig_record.has_tag('NM') else '')
    record.set_tag('OA', oa_tag)
    record.set_tag('st', status.value)
    return record


def _create_header(genome, chrom_id, bam_wrappers, max_mate_dist):
    header = '@SQ\tSN:{}\tLN:{}\n'.format(genome.chrom_name(chrom_id), genome.chrom_len(chrom_id))
    for bam_wrapper in bam_wrappers:
        for group_id, sample in bam_wrapper.read_groups().values():
            assert isinstance(sample, str)
            header += '@RG\tID:{}\tSM:{}\n'.format(group_id, sample)
    header += '@CO\tmax_mate_dist={}\n'.format(max_mate_dist)
    return pysam.AlignmentHeader.from_text(header)


def _extract_reads(in_bam, out_bam, read_groups, region, genome, out_header, read_positions, max_mate_dist):
    for record in common.checked_fetch(in_bam, region, genome):
        if record.flag & 3844:
            continue

        curr_read_pos = read_positions[record.query_name]
        if not curr_read_pos.add_primary_pos(record.is_read2, record.reference_start):
            common.log('WARN: Read {} (read mate {}) has several primary alignments ({}:{} and more).'
                .format(record.query_name, record.is_read2 + 1, record.reference_name, record.reference_start + 1) +
                ' Skipping second alignment')
            continue
        curr_read_pos.save_mate_pos(record, genome, max_mate_dist)
        new_rec = _create_record(record, out_header, read_groups, bam_file_.ReadStatus.SameLoc, max_mate_dist)
        out_bam.write(new_rec)


class _ReadPositions:
    MAX_REALIGNED_DIST = 100

    def __init__(self):
        self.from_main_copy = False
        self.realigned_pos = ([], [])
        self.mate_written = [False, False]
        self.mate_position = None

    def add_primary_pos(self, is_read2, pos):
        self.from_main_copy = True
        positions = self.realigned_pos[is_read2]
        if positions:
            return False
        positions.append(pos)
        return True

    def add_realigned_pos(self, is_read2, new_pos):
        positions = self.realigned_pos[is_read2]
        if any(abs(prev_pos - new_pos) < _ReadPositions.MAX_REALIGNED_DIST for prev_pos in positions):
            return False
        positions.append(new_pos)
        return True

    def save_mate_pos(self, record, genome, max_mate_dist):
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
    for record in common.checked_fetch(in_bam, dupl.region2, genome):
        if record.flag & 3844:
            continue

        orig_aln = Alignment.from_record(record, genome)
        read_seq, reg1_aln = dupl.align_read(record.query_sequence, orig_aln, weights, calc_score=False)
        new_start = reg1_aln.ref_interval.start
        if not read_positions[record.query_name].add_realigned_pos(record.is_read2, new_start):
            continue

        cigar_tuples = reg1_aln.cigar.to_pysam_tuples() if reg1_aln.cigar is not None else None
        new_rec = _create_record(record, out_header, read_groups, bam_file_.ReadStatus.Realigned, max_mate_dist,
            dupl_strand=dupl.strand, seq=read_seq, cigar_tuples=cigar_tuples, start=new_start)
        read_positions[record.query_name].save_mate_pos(record, genome, max_mate_dist)
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


def _write_mates(in_bam, out_bam, read_positions, genome, out_header, read_groups, max_mate_dist):
    fetch_positions = []
    for read_name, read_pos in read_positions.items():
        mate_pos = read_pos.mate_pos()
        if mate_pos is not None:
            _is_read2, chrom_id, pos = mate_pos
            fetch_positions.append((chrom_id, pos))
    if not fetch_positions:
        return

    for region in _get_fetch_regions(fetch_positions):
        for record in common.checked_fetch(in_bam, region, genome):
            if record.flag & 3844:
                continue

            read_pos = read_positions.get(record.query_name)
            mate_pos = None if read_pos is None else read_pos.mate_pos()
            if mate_pos is None or mate_pos[0] != record.is_read2 or mate_pos[2] != record.pos:
                continue
            if read_pos.from_main_copy:
                new_rec = _create_record(record, out_header, read_groups, bam_file_.ReadStatus.ReadMate, max_mate_dist)
            else:
                new_rec = _create_record(record, out_header, read_groups, bam_file_.ReadStatus.ReadMate, max_mate_dist,
                    cigar_tuples=None, start=read_pos.realigned_pos[1 - record.is_read2])
            out_bam.write(new_rec)


DEFAULT_MATE_DISTANCE = 5000

def pool(bam_wrappers, out_path, interval, duplications, genome, *,
        samtools='samtools', weights=None, max_mate_dist=DEFAULT_MATE_DISTANCE, verbose=True, time_log=None):
    if weights is None:
        weights = Weights()
    if verbose:
        common.log('Extracting and realigning reads')
    if time_log is not None:
        time_log.log('Pooling reads')

    out_header = _create_header(genome, interval.chrom_id, bam_wrappers, max_mate_dist)
    tmp_path = out_path + '.tmp'
    with pysam.AlignmentFile(tmp_path, 'wb', header=out_header) as tmp_bam:
        for bam_index, bam_wrapper in enumerate(bam_wrappers, 1):
            if verbose:
                common.log('    [{:3d} / {}]  {}'.format(bam_index, len(bam_wrappers), bam_wrapper.filename))
            read_positions = defaultdict(_ReadPositions)

            read_groups = bam_wrapper.read_groups()
            with bam_wrapper.open_bam_file(genome) as bam_file:
                _extract_reads(bam_file, tmp_bam, read_groups, interval, genome, out_header, read_positions,
                    max_mate_dist)
                for dupl in duplications:
                    _extract_reads_and_realign(bam_file, tmp_bam, read_groups, dupl, genome,
                        out_header, weights, read_positions, max_mate_dist)
                if max_mate_dist != 0:
                    _write_mates(bam_file, tmp_bam, read_positions, genome, out_header, read_groups, max_mate_dist)

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


def load_duplications(table, genome, interval, exclude_str):
    exclude_dupl = parse_expression(exclude_str)
    duplications = []
    for tup in table.fetch(interval.chrom_name(genome), interval.start, interval.end):
        dupl = duplication_.Duplication.from_tuple(tup, genome)
        if dupl.is_tangled_region or exclude_dupl(dupl, genome):
            continue
        dupl.set_cigar_from_info()
        dupl = dupl.sub_duplication(interval)
        dupl.set_sequences(genome=genome)
        dupl.set_padding_sequences(genome, 200)
        duplications.append(dupl)
    return duplications


class BamWrapper:
    def __init__(self, filename, sample, genome, store_contigs=False):
        self._filename = filename
        self._input_sample = sample
        with self.open_bam_file(genome) as bam_file:
            self._old_read_groups = bam_file_.get_read_groups(bam_file)
            if store_contigs:
                self._contigs = tuple(bam_file.references)
            else:
                self._contigs = None
        self._read_groups = None

    def init_new_read_groups(self, samples):
        # Dictionary old_read_group -> (new_read_group, sample_name).
        self._read_groups = {}

        if self._input_sample is not None:
            # Associate reads without read_group with sample from the input file.
            self._read_groups[None] = ('__{}'.format(samples.id(self._input_sample)), self._input_sample)

        for old_read_group, old_sample in self._old_read_groups:
            # If BAM file is associated with sample name, all reads should have it. Otherwise, old sample name is kept.
            new_sample = self._input_sample or old_sample
            new_read_group = '{}-{}'.format(old_read_group, samples.id(new_sample))
            self._read_groups[old_read_group] = (new_read_group, new_sample)

        if not self._read_groups:
            common.log('ERROR: Input file {} has no read groups in the header.'.format(self._filename))
            common.log('Please specify sample name as "-i filename::sample" or in "-I input-list.txt"')
            exit(1)

    @property
    def filename(self):
        return self._filename

    @property
    def contigs(self):
        return self._contigs

    def clear_contigs(self):
        self._contigs = None

    def open_bam_file(self, genome):
        genome_filename = genome if isinstance(genome, str) else genome.filename
        return pysam.AlignmentFile(self._filename, reference_filename=genome_filename, require_index=True)

    def read_groups(self):
        return self._read_groups

    def present_samples(self):
        if self._input_sample is not None:
            return (self._input_sample,)
        return set(map(operator.itemgetter(1), self._old_read_groups))


def load_bam_files(input, input_list, genome):
    """
    Loads BAM files from either input or input-list.
    Returns list of BamWrapper's.
    """
    # List of tuples (filename, sample).
    FILENAME_SPLIT = '::'

    filenames = []
    if input:
        for filename in input:
            if FILENAME_SPLIT in filename:
                filename, sample = filename.split(FILENAME_SPLIT, 1)
                filenames.append((filename, sample))
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
                    filename = os.path.join(list_dir, filename)
                    filenames.append((filename, sample))
            except UnicodeDecodeError:
                raise ValueError('Cannot read input list "-I {0}", perhaps you want to use "-i {0}"?'
                    .format(input_list))

    bam_wrappers = [BamWrapper(filename, sample, genome, store_contigs=True) for filename, sample in filenames]
    bam_file_.compare_contigs(bam_wrappers, genome)

    samples = bam_file_.Samples.from_bam_wrappers(bam_wrappers)
    for bam_wrapper in bam_wrappers:
        bam_wrapper.init_new_read_groups(samples)
        bam_wrapper.clear_contigs()
    return bam_wrappers, samples

###
def get_only_regions(args):
    """
    Writes regions used for pooling/realining reads
    """
    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.table, parser=pysam.asTuple()) as table:
        interval = Interval.parse(args.region, genome)
        duplications = load_duplications(table, genome, interval, args.exclude)
     
    hom_regions = [dupl.region2.to_bed(genome) for dupl in duplications]
    with open(f"{args.only_regions}", "a") as outf:
        outf.writelines("\n".join(hom_regions) + "\n")
###

def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Pool reads from various copies of a duplication.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} (-i <bam> [...] | -I <bam-list>) -t <table> -f <fasta> -r <region> -o <bam>'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')

    inp_me = io_args.add_mutually_exclusive_group(required=True)
    inp_me.add_argument('-i', '--input', metavar='<file>', nargs='+',
        help='Input indexed BAM/CRAM files.\n'
            'All entries should follow the format "filename[::sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input-list.')
    inp_me.add_argument('-I', '--input-list', metavar='<file>',
        help='A file containing a list of input BAM/CRAM files.\n'
            'All lines should follow the format "filename[ sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input.\n\n')

    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed table with information about segmental duplications.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-r', '--region', metavar='<region>',
        help='Single region in format "chr:start-end". Start and end are 1-based inclusive.\nCommas are ignored.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output BAM file.')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>',
        default='length < 500 && seq_sim < 0.97',
        help='Exclude duplications for which the expression is true\n[default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-m', '--mate-dist', metavar='<int>|infinity', type=float, default=DEFAULT_MATE_DISTANCE,
        help='Output read mates even if they are outside of the duplication,\n'
            'if the distance between mates is less than <int> [default: %(default)s].\n'
            'Use 0 to skip all mates outside the duplicated regions.\n'
            'Use inf|infinity to write all mapped read mates.\n')
    opt_args.add_argument('-q', '--quiet', action='store_false', dest='verbose',
        help='Do not write information to the stderr.')
    opt_args.add_argument('--samtools', metavar='<path>|none', default='samtools',
            help='Path to samtools executable [default: %(default)s].\n'
                'Use python wrapper if "none", can lead to errors.')
    ###
    opt_args.add_argument('-x', '--only-regions', metavar='<file>', 
        help='File to fetch intervals for pooling and realigning.')
    ###

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    if args.samtools != 'none':
        common.check_executable(args.samtools)

    ###
    if args.only_regions:
        get_only_regions(args)
        return
    ###

    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.table, parser=pysam.asTuple()) as table:
        interval = Interval.parse(args.region, genome)
        bam_wrappers, _samples = load_bam_files(args.input, args.input_list, genome)
        duplications = load_duplications(table, genome, interval, args.exclude)
        pool(bam_wrappers, args.output, interval, duplications, genome,
            samtools=args.samtools, max_mate_dist=args.mate_dist, verbose=args.verbose)


if __name__ == '__main__':
    main()
