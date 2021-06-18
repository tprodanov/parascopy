#!/usr/bin/env python3

import argparse
import os
import pysam

from .inner import common
from .inner import duplication as duplication_
from .inner.genome import Genome, Interval
from .inner.alignment import Alignment, Weights
from .inner import bam_file as bam_file_
from .view import parse_expression
from . import long_version


_MAX_INSERT_SIZE = 50000

def _create_record(name, seq, qual, start, cigar_tuples, header, old_record, read_groups):
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
    return record


def _create_header(genome, chrom_id, read_groups):
    header = '@SQ\tSN:%s\tLN:%d\n' % (genome.chrom_name(chrom_id), genome.chrom_len(chrom_id))
    for curr_read_groups in read_groups:
        for group_id, sample in curr_read_groups.values():
            assert isinstance(sample, str)
            header += '@RG\tID:%s\tSM:%s\n' % (group_id, sample)
    return pysam.AlignmentHeader.from_text(header)


def _extract_reads(bam_file, read_groups, region, genome, out_header):
    for record in bam_file_.fetch(bam_file, region, genome):
        if record.flag & 3844:
            continue

        new_rec = _create_record(record.query_name, record.query_sequence, record.query_qualities,
            record.reference_start, record.cigartuples, out_header, record, read_groups)
        yield new_rec


_MAX_REALIGNED_DIST = 100

def _extract_reads_and_realign(bam_file, read_groups, dupl, genome, out_header, weights, realigned_positions):
    """
    Load reads from dupl.region2 and aligns them to dupl.region1.
    Returns iterator over records.

    realigned_positions: dictionary (read_name, is_read2): [realigned_pos].
    """
    for record in bam_file_.fetch(bam_file, dupl.region2, genome):
        if record.flag & 3844:
            continue

        orig_aln = Alignment.from_record(record, genome)
        reg1_aln = dupl.align_read(record.query_sequence, orig_aln, weights, calc_score=False)

        new_pos = reg1_aln.ref_interval.start
        key = (record.query_name, record.is_read2)
        if reg1_aln.cigar is not None and key in realigned_positions:
            if any(abs(prev_pos - new_pos) < _MAX_REALIGNED_DIST for prev_pos in realigned_positions[key]):
                continue
            realigned_positions[key].append(new_pos)
        else:
            realigned_positions[key] = [new_pos]

        rev_strand = reg1_aln.strand != orig_aln.strand
        cigar_tuples = reg1_aln.cigar.to_pysam_tuples() if reg1_aln.cigar is not None else None
        new_rec = _create_record(record.query_name,
            common.cond_rev_comp(record.query_sequence, rev_strand),
            common.cond_reverse(record.query_qualities, rev_strand),
            new_pos, cigar_tuples, out_header, record, read_groups)
        yield new_rec


def pool_reads_inner(bam_files, read_groups, out_path, interval, duplications, genome, weights, samtools, verbose=True,
        time_log=None):
    out_header = _create_header(genome, interval.chrom_id, read_groups)

    if verbose:
        common.log('Extracting and realigning reads')

    if time_log is not None:
        time_log.log('Pooling reads')
    tmp_path = out_path + '.tmp'
    with pysam.AlignmentFile(tmp_path, 'wb', header=out_header) as tmp_bam:
        for i, (bam_file, curr_read_groups) in enumerate(zip(bam_files, read_groups)):
            if verbose:
                common.log('    Extracting reads from %3d: %s' % (i + 1, bam_file.filename.decode()))
            realigned_positions = {}
            for dupl in duplications:
                for record in _extract_reads_and_realign(bam_file, curr_read_groups, dupl, genome,
                        out_header, weights, realigned_positions):
                    tmp_bam.write(record)
            for record in _extract_reads(bam_file, curr_read_groups, interval, genome, out_header):
                tmp_bam.write(record)

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

    pool_reads_inner(bam_files, read_groups, out_path, interval, duplications, genome, weights, verbose=args.verbose)


def load_bam_files(input, input_list, genome, allow_unnamed=False):
    """
    Loads BAM files from either input or input-list.
    Returns two lists with the same length and other:
        - bam_files: list of pysam.AlignmentFile,
        - read_groups: list of dictionaries { old_read_group : (new_read_group, sample_name) }.
            Length of the dictionary is the same as the length of bam_files.
    """
    # List of tuples (filename, sample).
    filenames = []
    if input:
        filenames.extend((filename, None) for filename in input)
    else:
        list_dir = os.path.dirname(input_list)
        with open(input_list) as inp:
            for line in inp:
                line = line.strip().split(maxsplit=1)
                if len(line) == 1:
                    filename = line[0]
                    sample = None
                else:
                    filename, sample = line
                filenames.append((os.path.join(list_dir, filename), sample))

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
