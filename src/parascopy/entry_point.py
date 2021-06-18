#!/usr/bin/env python3

import sys
import os

from . import __version__, long_version


def _get_usage(prog_name):
    usage = '''{version}

Usage:   {prog_name} <command> <arguments>

Commands:

[ Creating homology table ]
    pretable    Create homology pre-table.
                This command aligns genomic regions back to the genome
                to find homologous regions.
    table       Convert homology pre-table into homology table.
                This command combines overlapping homologous regions
                into longer duplications.

[ Analyzing BAM/CRAM files ]
    depth       Calculate read depth and variance in given genomic windows.
    cn          Find aggregate and paralog-specific copy number
                for given unique and duplicated regions.
    cn-using    Same as "cn", but use input model parameters.
    pool        Pool reads from various copies of a given homologous region.

[ Querying homology table ]
    view        View and filter homology table.
    msa         Visualize multiple sequence alignment of
                homologous regions.
    psvs        Output PSVs (paralogous-sequence variants)
                between homologous regions.

[ General help ]
    help            Show this help message.
    version         Show version.
    cite            Show citation information.
'''
    usage = usage.format(version=long_version(), prog_name=prog_name)
    return usage


def _get_valid_commands():
    return 'pretable table depth cn pool view msa psvs help version'.split()


def _throw_error(prog_name, command):
    sys.stderr.write('Error: unknown command "{}"\n\n'.format(command))
    sys.stderr.write('Usage: {} <command>\n'.format(prog_name))
    sys.stderr.write('    Valid commands: {}.\n'.format(', '.join(_get_valid_commands())))
    exit(1)


def _print_citations():
    print('Publication in progress, please check later!')


def _process_exceptions(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except BrokenPipeError:
        pass
    # Add other exceptions here, if needed.


def main():
    prog_name = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(_get_usage(prog_name))
        return
    command = sys.argv[1]
    inner_args = sys.argv[2:]
    inner_prog = '{} {}'.format(prog_name, command)

    if command == '-h' or command == '--help' or command == 'help':
        print(_get_usage(prog_name))
        return

    if command == '-V' or command == '--version' or command == 'version':
        print(long_version())
        return

    if command == 'cite' or command == 'citation':
        _print_citations()
        return

    if command == 'pretable':
        from . import pretable
        _process_exceptions(pretable.main, inner_prog, inner_args)
        return

    if command == 'table':
        from . import combine_table
        _process_exceptions(combine_table.main, inner_prog, inner_args)
        return

    if command == 'depth':
        from . import depth
        _process_exceptions(depth.main, inner_prog, inner_args)
        return

    if command == 'cn' or command == 'cn-using':
        is_new = command == 'cn'
        from . import detect_cn
        _process_exceptions(detect_cn.main, inner_prog, inner_args, is_new)
        return

    if command == 'pool':
        from . import pool_reads
        _process_exceptions(pool_reads.main, inner_prog, inner_args)
        return

    if command == 'view':
        from . import view
        _process_exceptions(view.main, inner_prog, inner_args)
        return

    if command == 'msa':
        from . import msa
        _process_exceptions(msa.main, inner_prog, inner_args)
        return

    if command == 'psvs':
        from . import psvs
        _process_exceptions(psvs.main, inner_prog, inner_args)
        return

    _throw_error(prog_name, command)


if __name__ == '__main__':
    main()
