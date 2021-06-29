#!/usr/bin/env python3

import sys
import os
import collections
import argparse
import operator
import itertools
import pysam
import numpy as np
import tempfile
import subprocess
import enum
import multiprocessing
import functools

from .inner import common
from .inner.genome import Genome, Interval
from .inner.cigar import Cigar
from .inner.duplication import Duplication, TangledRegion
from . import __pkg_name__, __version__, long_version


class _Stats:
    def __init__(self):
        self.total = 0
        self.cases = [0] * len(_Reason)
        self.many_hits = 0

    def update(self, reason):
        self.total += 1
        self.cases[reason.value] += 1

    def log(self, n_retained):
        self_aln = self.cases[_Reason.SelfAlignment.value]
        common.log('    Analyzed {:,} hits ({:,} self-alignments)'.format(self.total, self_aln))
        non_self = self.total - self_aln
        if non_self == 0:
            return
        common.log('    Of the remaining {:,} hits:'.format(non_self))

        perc = 100 / non_self
        too_short = self.cases[_Reason.TooShort.value]
        low_seq_sim = self.cases[_Reason.LowSeqSim.value]
        invalid = self.cases[_Reason.Invalid.value]
        partial_self_aln = self.cases[_Reason.PartialSelfAln.value]
        common.log('        Short alignments:    {:7,} ({:5.2f}%)'.format(too_short, too_short * perc))
        common.log('        Low seq similarity:  {:7,} ({:5.2f}%)'.format(low_seq_sim, low_seq_sim * perc))
        common.log('        Invalid alignments:  {:7,} ({:5.2f}%)'.format(invalid, invalid * perc))
        common.log('        Partial self-alns:   {:7,} ({:5.2f}%)'.format(partial_self_aln, partial_self_aln * perc))
        common.log('        High copy number:    {:7,} ({:5.2f}%)'.format(self.many_hits, self.many_hits * perc))
        common.log('        Retained alignments: {:7,} ({:5.2f}%)'.format(n_retained, n_retained * perc))
        retained2 = non_self - too_short - low_seq_sim - invalid - partial_self_aln - self.many_hits
        if n_retained != retained2:
            common.log('WARN: Number of retained alignments do not match: {} != {}'.format(n_retained, retained2))


class _Reason(enum.Enum):
    SelfAlignment = 0
    TooShort = 1
    LowSeqSim = 2
    Invalid = 3
    PartialSelfAln = 4
    Good = 5

    def is_tangled(self):
        return self is _Reason.PartialSelfAln


def _aln_to_dupl(record, curr_region, curr_region_seq, min_aln_len, min_seq_sim, keep_partial_self_aln):
    """
    Returns pair (Duplication, _Reason). If duplication is discarded, first item is None.
    """
    strand = not record.is_reverse
    ref_region = Interval(record.reference_id, record.reference_start, record.reference_end)
    self_intersection = ref_region.intersection_size(curr_region)
    if self_intersection >= 0.9 * len(curr_region):
        return (None, _Reason.SelfAlignment)
    if not keep_partial_self_aln and self_intersection > 0:
        return (ref_region.combine(curr_region), _Reason.PartialSelfAln)
    if len(ref_region) < min_aln_len:
        return (None, _Reason.TooShort)

    cigar = Cigar.from_pysam_tuples(record.cigartuples)
    if cigar.aligned_len < min_aln_len:
        return (None, _Reason.TooShort)

    left_clip, right_clip, cigar = cigar.remove_clipping()
    if not cigar[0][1].consumes_read() or not cigar[-1][1].consumes_read():
        common.log('ERROR: Self-alignment of %r produces invalid CIGAR: %s.' % (curr_region, cigar.to_str()))
        # For some reason this line does not work without calling str(.)
        common.log('ERROR (cont): BWA alignment: %s' % str(record))
        return (None, _Reason.Invalid)

    if strand:
        read_region = Interval(curr_region.chrom_id, curr_region.start + left_clip, curr_region.end - right_clip)
        read_seq = curr_region_seq[left_clip : len(curr_region_seq) - right_clip]
    else:
        read_region = Interval(curr_region.chrom_id, curr_region.start + right_clip, curr_region.end - left_clip)
        read_seq = common.rev_comp(curr_region_seq[right_clip : len(curr_region_seq) - left_clip])

    ext_cigar, ref_seq = cigar.to_extended_with_md(read_seq, record.get_tag('MD'))
    aln_stats = ext_cigar.calculate_stats()
    if aln_stats.seq_similarity < min_seq_sim:
        return (None, _Reason.LowSeqSim)

    dupl = Duplication(ref_region, read_region, strand)
    dupl.set_sequences(seq1=ref_seq, seq2=read_seq)
    dupl.set_full_cigar(ext_cigar)
    dupl.store_stats(aln_stats)
    dupl.estimate_complexity()
    dupl.info['clip'] = '%d,%d' % (left_clip, right_clip)
    return (dupl, _Reason.Good)


class _FakeRecord:
    def __init__(self, record):
        self.query_name = record.query_name
        self.is_reverse = record.is_reverse
        self.reference_id = record.reference_id
        self.reference_start = record.reference_start
        self.reference_end = record.reference_end
        self.cigartuples = tuple(record.cigartuples)
        self.md_tag = record.get_tag('MD')

    def get_tag(self, tag):
        assert tag == 'MD'
        return self.md_tag


def _aln_to_dupl_wrapper(record, regions, region_seqs,
        step_size, read_len, min_aln_len, min_seq_sim, keep_partial_self_aln):
    region_ix, read_ix = map(int, record.query_name.split('-'))
    region = regions[region_ix]
    shift = step_size * read_ix
    curr_seq = region_seqs[region_ix][shift : shift + read_len]
    curr_region = Interval(region.chrom_id, region.start + shift, region.start + shift + len(curr_seq))

    dupl_res = _aln_to_dupl(record, curr_region, curr_seq, min_aln_len, min_seq_sim, keep_partial_self_aln)
    return (region_ix, read_ix), dupl_res


def _write_genome_reads(region_seqs, read_len, step, min_aln_len, outp):
    count = 0
    for seq_i, seq in enumerate(region_seqs):
        for i, start in enumerate(range(0, len(seq) - min_aln_len + 1, step)):
            curr_seq = seq[start : start + read_len]
            # Because there could be unexpected letters (not only N).
            if curr_seq.count('A') + curr_seq.count('C') + curr_seq.count('G') + curr_seq.count('T') != len(curr_seq):
                continue
            outp.write('>%d-%d\n%s\n' % (seq_i, i, curr_seq))
            count += 1
    return count


def _run_bwa(in_path, out_path, genome, args):
    command = [
        args.bwa, 'mem',
        genome.filename, in_path, '-o', out_path,
        '-k', args.seed_len,
        '-t', args.threads,
        '-D', '%.3f' % (args.min_aln_len / args.read_len),
        '-a',
    ]
    bwa_process = common.Process(command)
    if not bwa_process.finish(zero_code_stderr=False):
        raise RuntimeError('BWA finished with non-zero status')


def _filter_high_copy_num(duplications, tangled_regions, max_homologies, min_aln_len):
    """
    Tries to select one region with copy number higher than max_homologies.
    Adds this region to tangled_regions and returns a list of duplications that go beyond it.
    The functions merges regions if there is more than one region with high copy number.
    """
    endpoints = []
    for dupl in duplications:
        endpoints.append((dupl.region2.start, True))
        endpoints.append((dupl.region2.end, False))
    endpoints.sort()

    copy_num = 0
    region_start = None
    region_end = None
    for pos, is_start in endpoints:
        if is_start:
            copy_num += 1
            if copy_num == max_homologies and region_start is None:
                region_start = pos
        else:
            copy_num -= 1
            if copy_num == max_homologies - 1:
                region_end = pos
    assert copy_num == 0
    assert (region_start is None) == (region_end is None)
    if region_start is None:
        return duplications

    region = Interval(duplications[0].region2.chrom_id, region_start, region_end)
    filtered_dupls = []
    for dupl in duplications:
        intersection = dupl.region2.intersection_size(region)
        if len(dupl.region2) - intersection >= min_aln_len:
            filtered_dupls.append(dupl)
    if not filtered_dupls:
        region = Interval(duplications[0].region2.chrom_id, endpoints[0][0], endpoints[-1][0])
    tangled_regions.append(region)
    return filtered_dupls


def _analyze_hits(regions, region_seqs, aln_file, genome, args):
    fn = functools.partial(_aln_to_dupl_wrapper,
        regions=regions, region_seqs=region_seqs, step_size=args.step_size, read_len=args.read_len,
        min_aln_len=args.min_aln_len, min_seq_sim=args.min_seq_sim, keep_partial_self_aln=args.keep_self_alns)

    threads = args.threads
    if threads == 1:
        results = map(fn, aln_file)
    else:
        pool = multiprocessing.Pool(threads)
        fake_records = map(_FakeRecord, aln_file)
        results = pool.map(fn, fake_records)
        pool.terminate()

    tangled_regions = []
    stats = _Stats()
    grouped_results = collections.defaultdict(list)
    for key, (dupl, reason) in results:
        stats.update(reason)
        if reason.is_tangled():
            # dupl is really an Interval here.
            tangled_regions.append(dupl)
        elif dupl is not None:
            grouped_results[key].append(dupl)

    min_aln_len = args.min_aln_len
    max_homologies = args.max_homologies
    retained = []
    for single_read_alns in grouped_results.values():
        n_alns = len(single_read_alns)
        if n_alns > max_homologies:
            single_read_alns = _filter_high_copy_num(single_read_alns, tangled_regions, max_homologies, min_aln_len)
            stats.many_hits += n_alns - len(single_read_alns)
            n_alns = len(single_read_alns)
        for dupl in single_read_alns:
            dupl.info['NA'] = n_alns
            retained.append(dupl)

    stats.log(len(retained))
    retained = _remove_subalignments(retained)
    tangled_regions = Interval.combine_overlapping(tangled_regions, max_dist=100)
    if tangled_regions:
        sum_len = sum(map(len, tangled_regions))
        common.log('    Removed {:,} tangled regions (sum length {})'
            .format(len(tangled_regions), common.fmt_len(sum_len)))
    res = retained
    for region in tangled_regions:
        res.append(TangledRegion(region))
    return res


def _remove_subalignments(duplications):
    duplications.sort(key=operator.attrgetter('region1'))
    keep = np.ones(len(duplications), dtype=bool)
    overview_i = 0

    for i, dupl in enumerate(duplications):
        region1 = dupl.region1
        region2 = dupl.region2
        while overview_i < i and not (keep[overview_i] and duplications[overview_i].region1.intersects(region1)):
            overview_i += 1

        for j in range(overview_i, i):
            prev = duplications[j]
            if not keep[j] or prev.strand != dupl.strand:
                continue
            if prev.region1.contains(region1) and prev.region2.contains(region2):
                keep[i] = False
                break
            if region1.contains(prev.region1) and region2.contains(prev.region2):
                keep[j] = False
    res = list(itertools.compress(duplications, keep))
    if len(res) != len(duplications):
        common.log('    Removed {:,} subalignments'.format(len(duplications) - len(res)))
    return res


def align_to_genome(regions, genome, args, wdir, outp):
    """
    Self-align segment to the genome using BWA.
    Returns list of duplications if `out` is `None`, otherwise writes to `out` and does not return anything.
    """
    region_seqs = [region.get_sequence(genome) for region in regions]

    common.log('    Writing artificial reads')
    reads_path = os.path.join(wdir, 'reads.fa')
    with open(reads_path, 'w') as reads_outp:
        n_reads = _write_genome_reads(region_seqs, args.read_len, args.step_size, args.min_aln_len, reads_outp)

    common.log('    Running BWA on {:,} artificial reads'.format(n_reads))
    aln_path = os.path.join(wdir, 'reads.sam')
    _run_bwa(reads_path, aln_path, genome, args)

    common.log('    Analyzing BWA hits')
    with pysam.AlignmentFile(aln_path) as aln_file:
        duplications = _analyze_hits(regions, region_seqs, aln_file, genome, args)

    for dupl in duplications:
        outp.write(dupl.to_str(genome))
        outp.write('\n')


def write_header(genome, out, argv):
    if argv:
        out.write('# {}\n'.format(' '.join(argv)))
    out.write('# {} {}\n'.format(__pkg_name__, __version__))
    out.write('# Genomic intervals (columns 2-3 and 6-7) are 0-based, semi-inclusive.\n')
    out.write('# In CIGAR: columns 1-3 represent reference and columns 5-7 represent reads.\n')
    out.write('# Info fields:\n')
    out.write('#     - compl: Region1 complexity (number of unique kmers / max possible kmers).\n')
    out.write('#     - av_mult: Average k-mer multiplicity (number of times it appears in region1).\n')

    out.write('# Chromosomes: ')
    out.write(','.join(map('%s:%d'.__mod__, genome.names_lengths)))
    out.write('\n')


def _group_regions(regions, min_len=1e6, max_len=20e6):
    res = []
    small_group = []
    small_group_len = 0

    for region in regions:
        if len(region) >= min_len:
            res.append((region,))
            continue
        if len(region) + small_group_len >= max_len:
            res.append(tuple(small_group))
            small_group.clear()
            small_group_len = 0
        small_group.append(region)
        small_group_len += len(region)

    if small_group:
        res.append(tuple(small_group))
    return res


def _sort_output(in_path, out, args, genome, bgzip_output):
    sort_command = [args.bedtools, 'sort', '-i', in_path, '-faidx', genome.filename + '.fai', '-header']
    sort_process = subprocess.Popen(sort_command,
        stdout=subprocess.PIPE if bgzip_output else out, stderr=subprocess.PIPE)

    if bgzip_output:
        bgzip_command = [args.bgzip, '-c', '-@', str(max(1, args.threads - 1))]
        bgzip_process = subprocess.Popen(bgzip_command, stdin=sort_process.stdout, stdout=out, stderr=subprocess.PIPE)
        bgzip_out, bgzip_err = bgzip_process.communicate()
        sort_err = None
    else:
        _, sort_err = sort_process.communicate()

    sort_returncode = sort_process.poll()
    # Want to catch None as well.
    if sort_returncode != 0:
        if sort_err is None:
            sort_err = sort_process.stderr.read()
        common.log('ERROR: Command "%s" finished with non-zero code %s' % (' '.join(sort_command), sort_returncode))
        common.log('Stderr:\n%s' % sort_err.decode()[:1000].strip())
        raise RuntimeError('Bedtools finished with non-zero code')

    if bgzip_output and bgzip_process.returncode:
        common.log('ERROR: Command "%s" finished with non-zero code %s'
            % (' '.join(bgzip_command), bgzip_process.returncode))
        common.log('Stderr:\n%s' % bgzip_err.decode()[:1000].strip())
        raise RuntimeError('Bgzip finished with non-zero code')


def main(prog_name=None, in_args=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Create homology pre-table.\n'
            'This command aligns genomic regions back to the genome to find homologous regions.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -f <fasta> [-r <regions> | -R <bed>] -o <bed> [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file. Should have BWA index.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output bed[.gz] file.')

    reg_args = parser.add_argument_group('Region arguments (optional, mutually exclusive)')
    reg_me = reg_args.add_mutually_exclusive_group(required=False)
    reg_me.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_me.add_argument('-R', '--regions-file', metavar='<file>',
        help='Input bed[.gz] file containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Filtering arguments')
    filt_args.add_argument('--min-aln-len', type=int, metavar='<int>', default=250,
        help='Minimal alignment length [default: %(default)s].')
    filt_args.add_argument('--min-seq-sim', type=float, metavar='<float>', default=0.96,
        help='Minimal sequence similarity [default: %(default)s].')
    filt_args.add_argument('--max-homologies', type=int, metavar='<int>', default=10,
        help='Skip regions with more than <int> homologies (copies) in the genome [default: %(default)s].')
    filt_args.add_argument('--keep-self-alns', action='store_true',
        help='If true, partial self-alignments will be kept\n'
            '(for example keep an alignment with a 300bp shift).\n'
            'Full self-alignments are always discarded.')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('--read-len', type=int, metavar='<int>', default=900,
        help='Artificial read length [default: %(default)s].')
    opt_args.add_argument('--step-size', type=int, metavar='<int>', default=150,
        help='Artificial reads step size [default: %(default)s].')
    opt_args.add_argument('-F', '--force', action='store_true',
        help='Force overwrite output file.')
    opt_args.add_argument('-@', '--threads', type=int, metavar='<int>', default=4,
        help='Use <int> threads [default: %(default)s].')

    exe_args = parser.add_argument_group('Executable arguments')
    exe_args.add_argument('-b', '--bwa', metavar='<path>', default='bwa',
        help='Path to BWA executable [default: %(default)s].')
    exe_args.add_argument('--seed-len', type=int, metavar='<int>', default=16,
        help='Minimum BWA seed length [default: %(default)s].')
    exe_args.add_argument('--tabix', metavar='<path>', default='tabix',
        help='Path to "tabix" executable [default: %(default)s].')
    exe_args.add_argument('--bedtools', metavar='<path>', default='bedtools',
        help='Path to "bedtools" executable [default: %(default)s].')
    exe_args.add_argument('--bgzip', metavar='<path>', default='bgzip',
        help='Path to "bgzip" executable [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help',
        help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version',
        version=long_version(),
        help='Show version.')
    args = parser.parse_args(in_args)

    common.check_executable(args.bwa, args.tabix, args.bgzip)
    args.threads = max(1, args.threads)
    common.log('Using %d threads' % args.threads)

    if not args.force and os.path.exists(args.output):
        sys.stderr.write('Output file "%s" exists, please use -F/--force to overwrite.\n' % args.output)
        exit(1)

    with Genome(args.fasta_ref) as genome, open(args.output, 'wb') as out, \
            tempfile.TemporaryDirectory(prefix='pretable') as wdir:
        common.log('Using temporary directory {}'.format(wdir))
        regions = common.get_regions(args, genome)
        region_groups = _group_regions(regions)
        n_groups = len(region_groups)
        common.log('Analyzing {:,} regions split into {:,} groups'.format(len(regions), n_groups))
        prefix_fmt = '[{:%d} / {}] ' % len(str(n_groups))

        common.log('Self alignment to find homology')
        temp_path = os.path.join(wdir, 'pretable.bed')
        with open(temp_path, 'w') as temp_outp:
            write_header(genome, temp_outp, sys.argv)
            for i, regions in enumerate(region_groups):
                if len(regions) == 1:
                    common.log((prefix_fmt + 'Aligning {} back to the genome:')
                        .format(i + 1, n_groups, regions[0].to_str_comma(genome)))
                else:
                    common.log((prefix_fmt + 'Aligning {} regions back to the genome [{} ... {}]:')
                        .format(i + 1, n_groups, len(regions),
                        regions[0].to_str_comma(genome), regions[-1].to_str_comma(genome)))
                align_to_genome(regions, genome, args, wdir, temp_outp)

        common.log('Sorting output file')
        _sort_output(temp_path, out, args, genome, args.output.endswith('.gz'))

    if args.output.endswith('.gz'):
        common.log('Indexing output with tabix')
        common.Process([args.tabix, '-p', 'bed', args.output]).finish()

    common.log('Success')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write('Keyboard interrupt\n')
        exit(1)
    except Exception:
        sys.stderr.write('Panic due to an error:\n===========\n')
        raise
