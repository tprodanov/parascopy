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
from . import long_version


class _SecondRegion:
    def __init__(self, start, end, dupl_index):
        self.start = start
        self.end = end
        self.dupl_index = dupl_index

    def copy(self):
        return _SecondRegion(self.start, self.end, self.dupl_index)

    def __str__(self):
        return 'SR(start=%d, end=%d, dupl=%d)' % (self.start, self.end, self.dupl_index)

    def __repr__(self):
        return self.__str__()


# reg2 will be used as list of _SecondRegion, or as a single _SecondRegion.
class _Psv:
    def __init__(self, start, end, reg2):
        self.start = start
        self.end = end
        self.reg2 = reg2

    def to_record(self, vcf_header, chrom_name, duplications, genome, aligned_pos):
        record = vcf_header.new_record()
        record.chrom = chrom_name
        record.start = self.start
        alleles = []

        # Defined here because self.active_dupl will change.
        need_allele = len(self.active_dupl) > 1
        pos2_info = []
        for region2 in self.reg2:
            dupl = duplications[region2.dupl_index]
            self.active_dupl.remove(region2.dupl_index)

            if not alleles:
                start1 = dupl.region1.start
                alleles.append(dupl.padded_subseq1(self.start - start1, self.end - start1))

            alt = dupl.padded_subseq2(region2.start, region2.end)
            try:
                allele_ix = alleles.index(alt)
            except ValueError:
                allele_ix = len(alleles)
                alleles.append(alt)
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
        return 'start=%s, end=%s, reg2=%s' % (self.start, self.end, self.reg2)


def create_vcf_header(genome, chrom_ids=None, argv=None):
    vcf_header = pysam.VariantHeader()
    if argv is not None:
        vcf_header.add_line('##command="%s"' % ' '.join(argv))
    for chrom_id in (chrom_ids or range(genome.n_chromosomes())):
        vcf_header.add_line('##contig=<ID=%s,length=%d>' % (genome.chrom_name(chrom_id), genome.chrom_len(chrom_id)))
    vcf_header.add_line('##INFO=<ID=pos2,Number=.,Type=String,Description="Second positions of the PSV. '
        'Format: chrom:pos:strand[:allele]">')
    return vcf_header


def create_psv_records(duplications, genome, vcf_header, region, tangled_regions):
    if not duplications:
        return []

    tangled_searcher = common.NonOverlappingSet.from_regions(tangled_regions)
    # Contains list of lists.
    # aligned_pos[dupl_i][pos1] = pos2, where either pos1 and pos2 are aligned, or pos2 = None.
    # Both positions are relative to the duplication.
    aligned_pos = []
    # List of _Psv. reg2 is used as a single _SecondRegion.
    # Also contains fake PSVs, which have end = None, they indicate starts and ends of duplications.
    # In that case reg2 is a pair (dupl_i, is_start).
    psvs = []

    for i, dupl in enumerate(duplications):
        dupl_seq1 = dupl.seq1
        dupl_seq2 = dupl.seq2
        reg1_start = dupl.region1.start
        psvs.append(_Psv(reg1_start, None, (i, True)))
        psvs.append(_Psv(dupl.region1.end, None, (i, False)))

        for (start1, end1, start2, end2) in dupl.full_cigar.find_differences():
            if end1 - start1 != end2 - start2:
                assert start1 >= 0 and start2 >= 0
                ref = dupl_seq1[start1:end1]
                alt = dupl_seq2[start2:end2]
                psv_start = reg1_start + start1
                new_start = variants_.move_left(psv_start, ref, (alt,), dupl_seq1, reg1_start, skip_alleles=True)
                if new_start is not None:
                    shift = psv_start - new_start
                    start1 -= shift
                    end1 -= shift
                    start2 -= shift
                    end2 -= shift

            psv_start = reg1_start + start1
            psv_end = reg1_start + end1
            if psv_start < region.end and region.start < psv_end and tangled_searcher.is_empty(psv_start, psv_end):
                psvs.append(_Psv(psv_start, psv_end, _SecondRegion(start2, end2, i)))

        curr_aligned_pos = []
        exp_pos1 = 0
        for pos2, pos1 in dupl.full_cigar.aligned_pairs():
            if exp_pos1 != pos1:
                for _ in range(exp_pos1, pos1):
                    curr_aligned_pos.append(None)
            exp_pos1 = pos1 + 1
            curr_aligned_pos.append(pos2)
        assert len(curr_aligned_pos) == len(dupl.region1)
        aligned_pos.append(curr_aligned_pos)
    psvs.sort()

    chrom_name = genome.chrom_name(duplications[0].region1.chrom_id)
    psv_records = []
    # _Psv instance. reg2 is used as a list of _SecondRegion.
    to_save = _Psv(None, None, [])
    to_save.active_dupl = set()
    active_dupl = set()

    for psv in psvs:
        if psv.is_fake():
            dupl_i, is_start = psv.reg2
            if is_start:
                active_dupl.add(dupl_i)
                if to_save.defined() and to_save.end > duplications[dupl_i].region1.start:
                    to_save.active_dupl.add(dupl_i)
            else:
                active_dupl.remove(dupl_i)
            continue

        if to_save.defined() and to_save.end > psv.start:
            skip_append = False
            end_shift = max(0, psv.end - to_save.end)
            to_save.end += end_shift
            for region2 in to_save.reg2:
                region2.end += end_shift
                skip_append |= region2.dupl_index == psv.reg2.dupl_index

            if not skip_append:
                start_shift = psv.start - to_save.start
                end_shift = to_save.end - psv.end
                to_save.reg2.append(
                    _SecondRegion(psv.reg2.start - start_shift, psv.reg2.end + end_shift, psv.reg2.dupl_index))

        else:
            if to_save.defined():
                psv_records.append(to_save.to_record(vcf_header, chrom_name, duplications, genome, aligned_pos))
            to_save.start = psv.start
            to_save.end = psv.end
            to_save.reg2.clear()
            to_save.reg2.append(psv.reg2.copy())
            to_save.active_dupl = set(active_dupl)
    if to_save.defined():
        psv_records.append(to_save.to_record(vcf_header, chrom_name, duplications, genome, aligned_pos))

    psv_records.sort(key=operator.attrgetter('start'))
    return psv_records


def write_psvs(psv_records, vcf_header, vcf_path, tabix):
    gzip = vcf_path.endswith('.gz')
    with pysam.VariantFile(vcf_path, 'wz' if gzip else 'w', header=vcf_header) as vcf_file:
        for psv in psv_records:
            vcf_file.write(psv)
    if gzip and tabix != 'none':
        common.Process([tabix, '-p', 'vcf', vcf_path]).finish()


def main(prog_name=None, in_args=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Output PSVs (paralogous-sequence variants) between homologous regions.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -i <table> -f <fasta> (-r <region> | -R <bed>) -o <vcf> [args]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-i', '--input', metavar='<file>', required=True,
        help='Input indexed bed.gz homology table.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output vcf[.gz] file.')

    reg_args = parser.add_argument_group('Region arguments (required, mutually exclusive)')
    reg_me = reg_args.add_mutually_exclusive_group(required=True)
    reg_me.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_me.add_argument('-R', '--regions-file', metavar='<file>',
        help='Input bed[.gz] file containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>', default='length < 500',
        help='Exclude duplications for which the expression is true [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_args)

    with Genome(args.fasta_ref) as genome, pysam.TabixFile(args.input, parser=pysam.asTuple()) as table:
        regions = common.get_regions(args, genome)
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
                    duplications.append(dupl)
            records.extend(create_psv_records(duplications, genome, vcf_header, region, tangled_regions))
        write_psvs(records, vcf_header, args.output, 'tabix')


if __name__ == '__main__':
    main()
