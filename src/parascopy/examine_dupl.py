#!/usr/bin/env python3

import operator
import argparse
import pysam

from .view import parse_expression
from .inner import common
from .inner.genome import ChromNames, Interval
from .inner.duplication import Duplication
from .inner.paralog_cn import Filters
from .inner import errors
from . import __pkg_name__, __version__, long_version


class Entry:
    def __init__(self, region):
        self.region1 = region
        self.filter = Filters()
        self.unknown_ref_cn = False
        self.homologous_regions = []
        self.info = {}

    def to_str(self, genome):
        s = '{}\t{}\t'.format(self.region1.to_bed(genome), self.filter.to_str())
        s += '{}{}\t'.format('>=' if self.unknown_ref_cn else '', 2 + 2 * len(self.homologous_regions))
        s += ';'.join('{}={}'.format(key, val) for key, val in self.info.items())
        s += '\t'
        s += ','.join(self.homologous_regions) if self.homologous_regions else '*'
        return s


def examine_duplications(region, table, genome, excl_dupl, out, args):
    duplications = []
    for tup in table.fetch(region.chrom_name(genome), region.start, region.end):
        dupl = Duplication.from_tuple(tup, genome)
        if dupl.is_tangled_region:
            duplications.append(dupl)
        elif not excl_dupl(dupl, genome):
            dupl.set_cigar_from_info()
            duplications.append(dupl)

    n_dupl = len(duplications)
    dupl_intervals = []
    for dupl in duplications:
        dupl_intervals.append(dupl.region1.intersect(region))
    # Need this to have non-duplicated regions in the output.
    dupl_intervals.append(region)

    disj_intervals = Interval.get_disjoint_subregions(dupl_intervals)
    for subregion, ixs in disj_intervals:
        if len(subregion) < args.min_length:
            continue

        entry = Entry(subregion)
        seq_similarities = []

        for i in ixs:
            if i == n_dupl:
                continue
            dupl = duplications[i]
            if dupl.is_tangled_region:
                entry.filter.add('TANGLED')
                entry.unknown_ref_cn = True
                continue

            subregion2 = dupl.subregion2(subregion.start, subregion.end)
            if subregion2 is None:
                # subregion falls in a deletion within the duplication.
                entry.homologous_regions.append('DEL:{}:{}'.format(dupl.region2.to_str(genome), dupl.strand_str))
            else:
                entry.homologous_regions.append('{}:{}'.format(subregion2.to_str(genome), dupl.strand_str))
            seq_similarities.append(dupl.info['SS'])

        entry.info['length'] = str(len(subregion))
        if seq_similarities:
            entry.info['seq_sim'] = ','.join(seq_similarities)
        out.write(entry.to_str(genome) + '\n')


def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Split input regions by reference copy number.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -t <table> -o <table> [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed.gz homology table.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output bed[.gz] file.')

    reg_args = parser.add_argument_group('Region arguments (optional)')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file>',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>', default='length < 500',
        help='Exclude duplications for which the expression is true [default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-m', '--min-length', type=int, metavar='<int>', default=0,
        help='Do not output entries shorter that the minimal length [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    with pysam.TabixFile(args.table, parser=pysam.asTuple()) as table, \
            common.open_possible_gzip(args.output, 'w', bgzip=True) as out:
        genome = ChromNames.from_table(table)
        excl_dupl = parse_expression(args.exclude)

        try:
            regions = common.get_regions(args, genome, only_unique=False)
            regions.sort()
            regions = Interval.combine_overlapping(regions)
        except errors.EmptyResult:
            regions = genome.all_chrom_intervals()

        out.write('## {}\n'.format(common.command_to_str()))
        out.write('## {} v{}\n'.format(__pkg_name__, __version__))
        out.write('#chrom\tstart\tend\tfilter\tref_CN\tinfo\thomologous_regions\n')

        for region in regions:
            examine_duplications(region, table, genome, excl_dupl, out, args)

    if args.output.endswith('.gz'):
        common.log('Index output with tabix')
        common.Process(['tabix', '-p', 'bed', args.output]).finish()


if __name__ == '__main__':
    main()
