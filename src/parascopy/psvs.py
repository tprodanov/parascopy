#!/usr/bin/env python3

import operator
import sys
import argparse
import pysam

from .view import parse_expression
from .inner import common
from .inner.genome import Genome, Interval
from .inner.duplication import Duplication
from .inner import variants as variants_
from .inner import itree
from . import long_version


class _SecondRegion:
    def __init__(self, start, end, dupl_index):
        self.start = start
        self.end = end
        self.dupl_index = dupl_index

    def copy(self):
        return _SecondRegion(self.start, self.end, self.dupl_index)

    def __str__(self):
        return 'SR(start={}, end={}, dupl={})'.format(self.start, self.end, self.dupl_index)

    def __repr__(self):
        return self.__str__()


class _Psv:
    def __init__(self, start, end, reg2=None):
        self.start = start
        self.end = end
        # List of second regions.
        # Fake PSVs (at the start and end of duplications) will have self.end = None and
        # self.reg2 = [(dupl_i, is_start)].
        self.reg2 = []
        if reg2 is not None:
            self.reg2.append(reg2)
        self.active_dupl = None

    def overlaps_boundary(self, duplications):
        for region2 in self.reg2:
            dupl = duplications[region2.dupl_index]
            if self.start < dupl.region1.start or dupl.region1.end < self.end:
                return True
        return False

    def to_complex_record(self, vcf_header, chrom_name, duplications):
        record = vcf_header.new_record()
        record.chrom = chrom_name
        record.start = self.start
        record.stop = self.end
        record.alts = ('<COMPLEX>',)
        record.qual = 0
        record.filter.add('BOUNDARY')

        for region2 in self.reg2:
            dupl = duplications[region2.dupl_index]
            dupl_start1 = dupl.region1.start
            if dupl_start1 <= self.start and self.end <= dupl.region1.end:
                record.ref = dupl.seq1[self.start - dupl_start1 : self.end - dupl_start1]
                break
        return record

    def to_record(self, vcf_header, chrom_name, duplications, genome, aligned_pos):
        record = vcf_header.new_record()
        record.chrom = chrom_name
        record.start = self.start
        alleles = []
        alleles_dict = {}

        # Defined here because self.active_dupl will change.
        need_allele = len(self.active_dupl) > 1
        pos2_info = []
        for region2 in self.reg2:
            dupl = duplications[region2.dupl_index]
            self.active_dupl.remove(region2.dupl_index)

            if not alleles:
                start1 = dupl.region1.start
                ref = dupl.padded_subseq1(self.start - start1, self.end - start1)
                alleles.append(ref)
                alleles_dict[ref] = 0

            alt = dupl.padded_subseq2(region2.start, region2.end)
            allele_ix = alleles_dict.get(alt)
            if allele_ix is None:
                allele_ix = len(alleles)
                alleles.append(alt)
                alleles_dict[alt] = allele_ix
            pos2 = dupl.region2.start + region2.start if dupl.strand else dupl.region2.end - region2.end
            pos2_info.append((dupl.region2.chrom_id, pos2 + 1, dupl.strand_str, allele_ix))

        for dupl_i in self.active_dupl:
            dupl = duplications[dupl_i]
            region1_start = dupl.region1.start
            region1_end = dupl.region1.end
            if self.start < region1_start or self.end > region1_end:
                continue

            pos2 = None
            if dupl.strand:
                tmp_pos = self.start
                while region1_start <= tmp_pos:
                    if aligned_pos[dupl_i][tmp_pos - region1_start] is not None:
                        pos2 = dupl.region2.start + aligned_pos[dupl_i][tmp_pos - region1_start] + 1
                        break
                    tmp_pos -= 1
            else:
                tmp_pos = self.end - 1
                while tmp_pos < region1_end:
                    if aligned_pos[dupl_i][tmp_pos - region1_end] is not None:
                        pos2 = dupl.region2.end - aligned_pos[dupl_i][tmp_pos - region1_end]
                        break
                    tmp_pos += 1
            if pos2 is not None:
                pos2_info.append((dupl.region2.chrom_id, pos2, dupl.strand_str, 0))

        record.alleles = alleles
        pos2_info.sort()
        fmt = '{}:{}:{}'
        if need_allele:
            fmt += ':{}'
        pos2_info = [fmt.format(genome.chrom_name(chrom_id), pos, strand, allele)
            for chrom_id, pos, strand, allele in pos2_info]
        record.info['pos2'] = pos2_info
        record.qual = 100
        return record

    def defined(self):
        return self.start is not None

    def is_fake(self):
        return self.end is None

    def __lt__(self, other):
        return self.start < other.start

    def __str__(self):
        return 'start={}, end={}, reg2={}, active={}'.format(self.start, self.end, self.reg2, self.active_dupl)

    def copy(self):
        res = _Psv(self.start, self.end)
        for entry in self.reg2:
            res.reg2.append(entry.copy())
        if self.active_dupl is not None:
            res.active_dupl = set(self.active_dupl)
        return res


def create_vcf_header(genome, chrom_ids=None, argv=None):
    vcf_header = pysam.VariantHeader()
    if argv is not None:
        vcf_header.add_line(common.vcf_command_line(argv))
    for chrom_id in (chrom_ids or range(genome.n_chromosomes)):
        vcf_header.add_line('##contig=<ID={},length={}>'
            .format(genome.chrom_name(chrom_id), genome.chrom_len(chrom_id)))
    vcf_header.add_line('##ALT=<ID=COMPLEX,Description="Long and complex PSV.">')
    vcf_header.add_line('##FILTER=<ID=BOUNDARY,Description="PSV overlaps boundary of a duplication.">')
    vcf_header.add_line('##INFO=<ID=pos2,Number=.,Type=String,Description="Second positions of the PSV. '
        'Format: chrom:pos:strand[:allele]">')
    return vcf_header


def duplication_differences(region1, reg1_seq, reg2_seq, full_cigar, dupl_i, psvs, aligned_pos,
        in_region, tangled_searcher, check_psvs=False):
    reg1_start = region1.start
    psvs.append(_Psv(reg1_start, None, (dupl_i, True)))
    psvs.append(_Psv(region1.end, None, (dupl_i, False)))

    for start1, end1, start2, end2 in full_cigar.find_differences():
        psv_size = max(end1 - start1, end2 - start2)
        if check_psvs and psv_size < variants_.MAX_SHIFT_VAR_SIZE and end1 - start1 != end2 - start2:
            assert start1 >= 0 and start2 >= 0
            ref = reg1_seq[start1:end1]
            alt = reg2_seq[start2:end2]
            psv_start = reg1_start + start1
            move_res = variants_.move_left(psv_start, ref, (alt,), reg1_seq, reg1_start)
            if move_res is not None:
                new_start, shifted_alleles = move_res
                shift = psv_start - new_start
                s = 'WARN: Variant/PSV [{}]:{:,}  Alleles: {}, {}   is not in a canonical representation:\n'.format(
                    region1.chrom_id, psv_start + 1, ref, alt)
                s += '    New position {:,}  Alleles: {}   ({} bp shift)'.format(
                    new_start + 1, ', '.join(shifted_alleles), shift)
                common.log(s)

        psv_start = reg1_start + start1
        psv_end = reg1_start + end1
        if tangled_searcher.n_overlaps(psv_start, psv_end) == 0:
            psvs.append(_Psv(psv_start, psv_end, _SecondRegion(start2, end2, dupl_i)))

    curr_aligned_pos = []
    exp_pos1 = 0
    for pos2, pos1 in full_cigar.aligned_pairs():
        if exp_pos1 != pos1:
            for _ in range(exp_pos1, pos1):
                curr_aligned_pos.append(None)
        exp_pos1 = pos1 + 1
        curr_aligned_pos.append(pos2)
    assert len(curr_aligned_pos) == len(region1)
    aligned_pos.append(curr_aligned_pos)


def combine_psvs(psvs, aligned_pos, duplication_starts1):
    psvs.sort()
    to_save = _Psv(None, None)
    to_save.active_dupl = set()
    active_dupl = set()

    for psv in psvs:
        assert len(psv.reg2) == 1
        psv_reg2 = psv.reg2[0]

        if psv.is_fake():
            dupl_i, is_start = psv_reg2
            if is_start:
                active_dupl.add(dupl_i)
                if to_save.defined() and to_save.end > duplication_starts1[dupl_i]:
                    to_save.active_dupl.add(dupl_i)
            else:
                active_dupl.remove(dupl_i)
            continue

        if to_save.defined() and to_save.end > psv.start:
            skip_append = False
            end_shift = max(0, int(psv.end) - int(to_save.end))
            to_save.end += end_shift
            for region2 in to_save.reg2:
                region2.end += end_shift
                skip_append |= region2.dupl_index == psv_reg2.dupl_index

            if not skip_append:
                start_shift = psv.start - to_save.start
                end_shift = to_save.end - psv.end
                to_save.reg2.append(
                    _SecondRegion(psv_reg2.start - start_shift, psv_reg2.end + end_shift, psv_reg2.dupl_index))

        else:
            if to_save.defined():
                yield to_save.copy()
            to_save.start = psv.start
            to_save.end = psv.end
            to_save.reg2.clear()
            to_save.reg2.append(psv_reg2.copy())
            to_save.active_dupl = set(active_dupl)
    if to_save.defined():
        yield to_save.copy()


def create_psv_records(duplications, genome, vcf_header, in_region, tangled_regions):
    if not duplications:
        return []

    tangled_searcher = itree.NonOverlTree(tangled_regions, itree.start, itree.end)
    # Contains list of lists.
    # aligned_pos[dupl_i][pos1] = pos2, where either pos1 and pos2 are aligned, or pos2 = None.
    # Both positions are relative to the duplication.
    aligned_pos = []
    # List of _Psv. reg2 is used as a single _SecondRegion.
    # Also contains fake PSVs, which have end = None, they indicate starts and ends of duplications.
    # In that case reg2 is a pair (dupl_i, is_start).
    psvs = []

    for dupl_i, dupl in enumerate(duplications):
        duplication_differences(dupl.region1, dupl.seq1, dupl.seq2, dupl.full_cigar, dupl_i, psvs, aligned_pos,
            in_region, tangled_searcher)

    chrom_name = genome.chrom_name(duplications[0].region1.chrom_id)
    psv_records = []
    duplication_starts1 = [dupl.region1.start for dupl in duplications]
    for pre_psv in combine_psvs(psvs, aligned_pos, duplication_starts1):
        if pre_psv.end < in_region.start or pre_psv.start >= in_region.end:
            continue
        if pre_psv.overlaps_boundary(duplications):
            psv_records.append(pre_psv.to_complex_record(vcf_header, chrom_name, duplications))
        else:
            psv_records.append(pre_psv.to_record(vcf_header, chrom_name, duplications, genome, aligned_pos))
    psv_records.sort(key=operator.attrgetter('start'))
    return psv_records


def write_psvs(psv_records, vcf_header, vcf_path, tabix):
    gzip = vcf_path.endswith('.gz')
    with pysam.VariantFile(vcf_path, 'wz' if gzip else 'w', header=vcf_header) as vcf_file:
        for psv in psv_records:
            vcf_file.write(psv)
    if gzip and tabix != 'none':
        common.Process([tabix, '-p', 'vcf', vcf_path]).finish()


def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Output PSVs (paralogous-sequence variants) between homologous regions.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -t <table> -f <fasta> (-r <region> | -R <bed>) -o <vcf> [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed.gz homology table.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output vcf[.gz] file.')

    reg_args = parser.add_argument_group('Region arguments (required)')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file>',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>', default='length < 500',
        help='Exclude duplications for which the expression is true [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.table, parser=pysam.asTuple()) as table:
        regions = Interval.combine_overlapping(common.get_regions(args, genome))
        chrom_ids = sorted(set(map(operator.attrgetter('chrom_id'), regions)))
        vcf_header = create_vcf_header(genome, chrom_ids, sys.argv)
        records = []

        excl_dupl = parse_expression(args.exclude)
        for region in regions:
            duplications = []
            tangled_regions = []
            for tup in table.fetch(region.chrom_name(genome), region.start, region.end):
                dupl = Duplication.from_tuple(tup, genome)
                if dupl.is_tangled_region:
                    tangled_regions.append(dupl.region1)
                elif not excl_dupl(dupl, genome):
                    dupl.set_cigar_from_info()
                    dupl.set_sequences(genome=genome)
                    dupl.set_padding_sequences(genome, 100)
                    duplications.append(dupl)
            records.extend(create_psv_records(duplications, genome, vcf_header, region, tangled_regions))
        write_psvs(records, vcf_header, args.output, 'tabix')


if __name__ == '__main__':
    main()
