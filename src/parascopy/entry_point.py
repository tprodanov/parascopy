#!/usr/bin/env python3

import sys
import os

from . import __version__, long_version


def _get_usage(prog_name):
    usage = '''{version}

Usage:   {prog_name} <command> <arguments>

Commands:

\033[32m[ Creating homology table ]\033[0m
    pretable    Create homology pre-table.
                This command aligns genomic regions back to the genome
                to find homologous regions.
    table       Convert homology pre-table into homology table.
                This command combines overlapping homologous regions
                into longer duplications.

\033[32m[ Analyzing BAM/CRAM files ]\033[0m
    depth       Calculate read depth and variance in given genomic windows.
    cn          Find aggregate and paralog-specific copy number
                for given unique and duplicated regions.
    cn-using    Same as "cn", but use input model parameters.
    pool        Pool reads from various copies of a given homologous region.
    call        Call pooled and paralog-specific variants.

\033[32m[ Querying homology table ]\033[0m
    view        View and filter homology table.
    msa         Visualize multiple sequence alignment of
                homologous regions.
    psvs        Output PSVs (paralogous-sequence variants)
                between homologous regions.
    examine     Split input regions by reference copy number
                and write homologous regions.

\033[32m[ General help ]\033[0m
    help        Show this help message.
    version     Show version.
    cite        Show citation information.
'''
    usage = usage.format(version=long_version(), prog_name=prog_name)
    return usage


def _throw_error(prog_name, command, valid_commands):
    sys.stderr.write('Error: unknown command "{}"\n\n'.format(command))
    sys.stderr.write('Usage: {} <command>\n'.format(prog_name))
    sys.stderr.write('    Valid commands: {}.\n'.format(', '.join(valid_commands)))
    exit(1)


def _print_citations():
    print(long_version())
    print()
    print('Please cite:')
    print('  * Copy-number variation detection:')
    print('    Prodanov, T. & Bansal, V. Robust and accurate estimation of paralog-specific copy number')
    print('    for duplicated genes using whole-genome sequencing. \033[3mNature Communications\033[0m '
        '\033[1m13\033[0m, 3221 (2022)')
    print('    \033[4mhttps://doi.org/10.1038/s41467-022-30930-3\033[0m')
    print()
    print('  * Variant calling:')
    print('    Prodanov, T. & Bansal, V. A multi-locus approach for accurate variant calling in low-copy')
    print('    repeats using whole-genome sequencing. \033[3mBioinformatics\033[0m '
        '\033[1m39\033[0m, i279-i287 (2023)')
    print('    \033[4mhttps://doi.org/10.1093/bioinformatics/btad268\033[0m')
    print()


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
    inner_argv = sys.argv[2:]
    inner_prog = '{} {}'.format(prog_name, command)
    valid_commands = []

    valid_commands.append('help')
    if command == '-h' or command == '--help' or command == 'help':
        print(_get_usage(prog_name))
        return

    valid_commands.append('version')
    if command == '-V' or command == '--version' or command == 'version':
        print(long_version())
        return

    valid_commands.append('cite')
    if command == 'cite' or command == 'citation':
        _print_citations()
        return

    valid_commands.append('pretable')
    if command == 'pretable':
        from . import pretable
        return _process_exceptions(pretable.main, inner_prog, inner_argv)

    valid_commands.append('table')
    if command == 'table':
        from . import combine_table
        return _process_exceptions(combine_table.main, inner_prog, inner_argv)

    valid_commands.append('depth')
    if command == 'depth':
        from . import depth
        return _process_exceptions(depth.main, inner_prog, inner_argv)

    valid_commands.append('cn')
    valid_commands.append('cn-using')
    if command == 'cn' or command == 'cn-using':
        is_new = command == 'cn'
        from . import detect_cn
        return _process_exceptions(detect_cn.main, inner_prog, inner_argv, is_new)

    valid_commands.append('pool')
    if command == 'pool':
        from . import pool_reads
        return _process_exceptions(pool_reads.main, inner_prog, inner_argv)

    valid_commands.append('view')
    if command == 'view':
        from . import view
        return _process_exceptions(view.main, inner_prog, inner_argv)

    valid_commands.append('msa')
    if command == 'msa':
        from . import msa
        return _process_exceptions(msa.main, inner_prog, inner_argv)

    valid_commands.append('psvs')
    if command == 'psvs' or command == 'psv':
        from . import psvs
        return _process_exceptions(psvs.main, inner_prog, inner_argv)

    valid_commands.append('examine')
    if command == 'examine':
        from . import examine_dupl
        return _process_exceptions(examine_dupl.main, inner_prog, inner_argv)

    valid_commands.append('call')
    if command == 'call':
        from . import call_variants
        return _process_exceptions(call_variants.main, inner_prog, inner_argv)

    # valid_commands.append('pool-vcf')
    if command == 'pool-vcf' or command == 'pool_vcf':
        from . import pool_variants
        return _process_exceptions(pool_variants.main, inner_prog, inner_argv)

    _throw_error(prog_name, command, valid_commands)


if __name__ == '__main__':
    main()
