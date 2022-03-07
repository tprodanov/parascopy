#!/usr/bin/env python3

import argparse
import pysam
import re
import errno

from simpleeval import simple_eval

from .inner import common
from .inner.genome import ChromNames
from .inner.duplication import Duplication
from . import long_version


_ATTRIBUTES = { 'chrom1', 'chrom2', 'start1', 'start2', 'end1', 'end2', 'strand' }


def _parse_value(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _name_handler(dupl, genome):
    def inner(name):
        name = name.id
        if name in _ATTRIBUTES:
            if name == 'chrom1':
                return dupl.region1.chrom_name(genome)
            elif name == 'chrom2':
                return dupl.region2.chrom_name(genome)
            elif name == 'start1':
                return dupl.region1.start
            elif name == 'start2':
                return dupl.region2.start
            elif name == 'end1':
                return dupl.region1.end
            elif name == 'end2':
                return dupl.region2.end
            elif name == 'strand':
                return dupl.strand_str
            else:
                assert False

        if name in dupl.info:
            value = dupl.info[name]
        else:
            value = dupl.info[name.upper()]
        return _parse_value(value)
    return inner


def parse_expression(expr):
    expr = expr.replace('&&', ' and ').replace('||', ' or ').lower()
    expr = re.sub(r'\btrue\b', 'True', expr)
    expr = re.sub(r'\bfalse\b', 'False', expr)
    expr = re.sub(r'\bseq_sim\b', 'SS', expr)
    expr = re.sub(r'\blength\b', 'ALENGTH', expr)

    def inner(dupl, genome):
        return bool(simple_eval(expr, names=_name_handler(dupl, genome)))
    return inner


def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='View and filter homology table.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} <table> [-o <table>] [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('input', metavar='<file>',
        help='Input indexed bed.gz homology table.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=False,
        help='Optional: output path.')

    reg_args = parser.add_argument_group('Region arguments (optional)')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file>',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-i', '--include', metavar='<expr>',
        help='Include duplications for which the expression is true.')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>',
        help='Exclude duplications for which the expression is true.')
    filt_args.add_argument('-t', '--skip-tangled', action='store_true',
        help='Do not show tangled regions.')

    out_args = parser.add_argument_group('Output format arguments')
    out_args.add_argument('-p', '--pretty', action='store_true',
        help='Print commas as thousand separator and split info field into tab entries.')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    with pysam.TabixFile(args.input, parser=pysam.asTuple()) as table, \
            common.open_possible_gzip(args.output, 'w') as outp:
        genome = ChromNames.from_table(table)
        include = parse_expression(args.include) if args.include else None
        exclude = parse_expression(args.exclude) if args.exclude else None
        skip_tangled = args.skip_tangled

        for tup in common.fetch_iterator(table, args, genome):
            dupl = Duplication.from_tuple(tup, genome)
            if dupl.is_tangled_region:
                if skip_tangled:
                    continue
            elif include and not include(dupl, genome):
                continue
            elif exclude and exclude(dupl, genome):
                continue
            if args.pretty:
                outp.write(dupl.to_str_pretty(genome))
            else:
                outp.write(dupl.to_str(genome))
            outp.write('\n')

    if args.output and args.output.endswith('.gz'):
        common.log('Index output with tabix')
        common.Process(['tabix', '-p', 'bed', args.output]).finish()


if __name__ == '__main__':
    main()
