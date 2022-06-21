import sys
from time import perf_counter
from datetime import timedelta
import gzip
import subprocess
import itertools
import os
import operator
import numpy as np
import string
from Bio import bgzf
import pysam
import shutil
from scipy.special import logsumexp
import argparse
import shlex

from .genome import Interval, NamedInterval
from . import errors


_rev_comp = {'A':'T', 'T':'A', 'C':'G', 'G':'C','a':'T', 't':'A', 'c':'G', 'g':'C', 'N':'N', 'n':'N' }
def rev_comp(seq): # reverse complement of string
    return ''.join(_rev_comp.get(nt, 'X') for nt in seq[::-1])


def cond_rev_comp(seq, *, strand):
    if strand:
        return seq
    return rev_comp(seq)


def cond_reverse(qual, *, strand):
    return qual if strand else qual[::-1]


def gc_count(seq):
    return seq.count('C') + seq.count('G')


def gc_content(seq):
    return 100.0 * (seq.count('C') + seq.count('G')) / len(seq)


def log(string, out=sys.stderr):
    elapsed = str(timedelta(seconds=perf_counter() - log._timer_start))[:-5]
    out.write('{}  {}\n'.format(elapsed, string))
    out.flush()

log._timer_start = perf_counter()


def adjust_window_size(length, window):
    n_windows = np.ceil(length / window)
    return int(np.ceil(length / n_windows))


def open_possible_gzip(filename, mode='r', bgzip=False):
    if filename == '-' or filename is None:
        return sys.stdin if mode == 'r' else sys.stdout

    if filename.endswith('.gz'):
        if 'b' not in mode:
            mode += 't'
        if bgzip:
            return bgzf.open(filename, mode)
        return gzip.open(filename, mode)

    return open(filename, mode)


def open_possible_empty(filename, *args, **kwargs):
    if filename is not None:
        return open_possible_gzip(filename, *args, **kwargs)
    return EmptyContextManager()


def file_or_str_list(l, arg_name=None):
    """
    Input may consist of strings and files with strings (filename should contain "/").
    If arg_name is not None, include it in case of an error.
    """
    def format_command():
        s = ''
        if arg_name is not None:
            s += arg_name
        for val in l:
            s += ' ' + shlex.quote(val)
        return s

    res = []
    for val in l:
        if '/' in val or '\\' in val:
            if not os.path.isfile(val):
                raise ValueError('File {} does not exist: {}'.format(val, format_command()))
            with open(val) as inp:
                res.extend(map(str.strip, inp))
        else:
            res.append(val)
    if not res:
        raise ValueError('Failed parsing "{}": resulting list is empty'.format(format_command()))
    return res


def _normalize_output(text, max_len=2000):
    text = text.decode('utf-8', 'replace')
    if len(text) > max_len:
        return text[: max_len // 2].strip() + '\n...\n' + text[-max_len // 2 :].strip()
    return text.strip()


class Process:
    def __init__(self, command):
        self._command = list(map(str, command))
        self._process = subprocess.Popen(self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @property
    def command_str(self):
        return command_to_str(self._command, basename=False)

    def finish(self, zero_code_stderr=True):
        """
        Returns True if the process finished successfully.
        """
        out, err = self._process.communicate()

        if self._process.returncode != 0:
            log('ERROR: {} returned code {}'.format(self.command_str, self._process.returncode))
            if out:
                log('    Stdout: {}'.format(_normalize_output(out)))
            if err:
                log('    Stderr: {}'.format(_normalize_output(err)))
            return False
        elif err and zero_code_stderr:
            log('Process {} finished with code 0, but has non empty stderr: {}'
                .format(self.command_str, _normalize_output(err)))
        return True

    def terminate(self):
        self._process.terminate()


def check_executable(*paths):
    for path in paths:
        if shutil.which(path) is None:
            raise RuntimeError('Command "{}" is not executable'.format(path))


def check_writable(filename):
    if os.path.exists(filename):
        if os.path.isfile(filename):
            return os.access(filename, os.W_OK)
        return False
    pdir = os.path.dirname(filename) or '.'
    return os.access(pdir, os.W_OK)


class EmptyContextManager:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def open_vcf(filename, mode='r', can_be_none=False, **kwargs): # TODO: Use more.
    if filename is None and can_be_none:
        return EmptyContextManager()
    if filename == '-':
        return pysam.VariantFile(sys.stdin if mode == 'r' else sys.stdout, **kwargs)
    if filename.endswith('.gz'):
        return pysam.VariantFile(filename, mode=mode + 'z', **kwargs)
    return pysam.VariantFile(filename, mode=mode, **kwargs)


def get_regions_explicit(regions, regions_file, genome, only_unique=True):
    """
    Returns a list `[NamedInterval]`.
    """
    intervals = []
    for region in regions or ():
        intervals.append(NamedInterval.parse(region, genome))

    for filename in regions_file or ():
        with open_possible_gzip(filename) as inp:
            for line in inp:
                if line.startswith('#'):
                    continue

                line = line.strip().split('\t')
                chrom_id = genome.chrom_id(line[0])
                start = int(line[1])
                end = int(line[2])
                name = line[3] if len(line) > 3 else None
                interval = NamedInterval(chrom_id, start, end, genome, name)
                interval.trim(genome)
                intervals.append(interval)

    if only_unique:
        os_names = set()
        intervals.sort()
        unfilt_intervals = intervals
        intervals = []

        for i in range(len(unfilt_intervals)):
            interval = unfilt_intervals[i]
            if interval.os_name in os_names:
                log('ERROR: Region name "{}" ({}) appears twice! Ignore second entry.'
                    .format(interval.os_name, interval.full_name(genome)))
                continue
            os_names.add(interval.os_name)

            if i and unfilt_intervals[i - 1] == interval:
                log('ERROR: Region {} appears twice! Ignore second entry.'.format(interval.full_name(genome)))
                continue
            intervals.append(interval)

    if not intervals:
        raise errors.EmptyResult(
            'No regions provided! Please specify -r <region> [<region> ...] or -R <file> [<file> ...]')
    return intervals


def get_regions(args, genome, only_unique=True):
    return get_regions_explicit(args.regions, args.regions_file, genome, only_unique)


def _fetch_regions_wo_duplicates(obj, regions, genome, start_attr):
    prev_chrom_id = None
    prev_start = -1
    for region in regions:
        same_chrom = region.chrom_id == prev_chrom_id
        if not same_chrom:
            prev_start = -1
        entry = None

        # TODO: Warn if not possible to fetch.
        for entry in obj.fetch(region.chrom_name(genome), region.start, region.end):
            if same_chrom:
                start = entry.start if start_attr else int(entry[1])
                if start <= prev_start:
                    continue
            yield entry

        if entry is not None:
            start = entry.start if start_attr else int(entry[1])
            prev_start = max(prev_start, start)
        prev_chrom_id = region.chrom_id


def fetch_iterator(obj, args, genome, start_attr=False, only_unique=True, full_genome_if_empty=True):
    """
    Returns sorted iterator over pysam object with that has a `fetch` method.
    Looks at `args.regions` and `args.regions_file`. If both are `None`, returns `obj.fetch()`.

    If only_unique is True, all duplicates will be removed. However, for that it is necessary to be able to get
    the start position of each entry. If start_attr is True, it is able to use `entry.start`, otherwise,
    it is possible to call `int(entry[1])`.
    """
    if not args.regions and not args.regions_file:
        if full_genome_if_empty:
            return obj.fetch()
        raise ValueError('No regions provided! Please specify -r <region> [<region> ...] or -R <file> [<file> ...]')

    regions = Interval.combine_overlapping(get_regions(args, genome, only_unique=False))
    if not only_unique:
        return itertools.chain(*(obj.fetch(region.chrom_name(genome), region.start, region.end) for region in regions))
    return _fetch_regions_wo_duplicates(obj, regions, genome, start_attr)


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def mkdir_clear(path, rewrite):
    if os.path.exists(path) and rewrite:
        log('Cleaning directory "{}"'.format(path))
        shutil.rmtree(path)
    mkdir(path)


def parse_distance(value):
    """
    Removes commas, and analyzes prefixes.
    Possible suffixes: k, m
    """
    value = value.replace(',', '').lower()
    if value.endswith('m'):
        return int(float(value[:-1]) * 1e6)
    elif value.endswith('k'):
        return int(float(value[:-1]) * 1e3)
    return int(value)
parse_distance.__name__ = 'distance'


def record_ord_key(rec1):
    return (rec1.reference_id, rec1.reference_start)


def fmt_len(length):
    if length < 1000:
        return '{}bp'.format(length)
    elif length < 1000000:
        return '{:.1f} kb'.format(length / 1000)
    else:
        return '{:.1f} Mb'.format(length / 1e6)


def letter_suffix(index, chars=string.ascii_lowercase):
    """
    Returns one or two letter suffix. a, b, c, ..., aa, ab, ...
    """
    n = len(chars)
    if index < n:
        return chars[index]
    return chars[index // n - 1] + chars[index % n]


def tricube_kernel(values):
    # Calculates tricube_kernel(abs(values)), values over 1 are ignored.
    return np.power(1.0 - np.power(np.minimum(np.abs(values), 1.0), 3), 3)


def log1minus(x):
    """
    Return 1 - sum(x) in a logarithmic scale.
    """
    subtrahend = logsumexp(x)
    if subtrahend >= 0.0:
        return -np.inf
    return logsumexp((0.0, subtrahend), b=(1, -1))


LOG10 = np.log(10)

def phred_qual(probs, best_ix, max_value=10000):
    """
    Calculate PHRED quality of probs[best_ix], where sum(exp(probs)) = 1.
    """
    if len(probs) == 1:
        assert best_ix == 0
        return max_value
    oth_prob = logsumexp(np.delete(probs, best_ix))
    if np.isfinite(oth_prob):
        return min(-10 * oth_prob / LOG10, max_value)
    return max_value


def extended_phred_qual(probs, best_ix, *, sum_prob=None, rem_prob=None, max_value=10000):
    """
    Calculate PHRED quality of probs[best_ix], where sum(exp(probs)) = 1.
    In addition, it is known that there is an additional event with probability rem_prob.

    Exactly one of `sum_prob` and `rem_prob` should be set.
    If sum_prob is set, it should be log(1 - exp(rem_prob)).
    """
    if len(probs) == 1:
        assert best_ix == 0
        return min(-10 * rem_prob / LOG10, max_value)

    if sum_prob is None:
        sum_prob = logsumexp((0.0, min(rem_prob, 0.0)), b=(1, -1))
    else:
        assert rem_prob is None
        rem_prob = logsumexp((0.0, min(sum_prob, 0.0)), b=(1, -1))

    probs = probs + sum_prob
    probs[best_ix] = rem_prob
    oth_prob = logsumexp(probs)
    if np.isfinite(oth_prob):
        return min(-10 * oth_prob / LOG10, max_value)
    return max_value


class SingleMetavar(argparse.RawTextHelpFormatter):
    def _format_args(self, action, default_metavar):
        return self._metavar_formatter(action, default_metavar)(1)[0]


def extended_nargs(min_values=1, max_values=2):
    class RequiredLength(argparse.Action):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.nargs not in ('+', '*'):
                raise ValueError('Cannot create extended_nargs::RequiredLength - nargs must be "+" or "*", not "{}"'
                    .format(self.nargs))

        def __call__(self, parser, args, values, option_string=None):
            n_val = len(values)
            if n_val < min_values or n_val > max_values:
                msg = 'argument {} requires between {} and {} arguments (observed {})'.format(
                    '/'.join(self.option_strings), min_values, max_values, n_val)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength


def checked_fetch_coord(fetch_file, chrom, start, end):
    try:
        return fetch_file.fetch(chrom, start, end)
    except ValueError as e:
        common.log('ERROR: Cannot fetch {}:{}-{} from {} (possibly missing chromosome).'
            .format(chrom, start + 1, end, fetch_file.filename.decode()))
        return iter(())


def checked_fetch(fetch_file, region, genome):
    return checked_fetch_coord(fetch_file, region.chrom_name(genome), region.start, region.end)


def non_empty_file(filename):
    return os.path.isfile(filename) and os.path.getsize(filename) > 0


def command_to_str(command=None, basename=True, quote=True, sep=' '):
    '''
    Converts command into string.
        Replaces first argument with its basename, if `basename` is True.
        Quotes arguments, if needed and `quote` is True.

    If command is None, uses sys.argv.
    '''
    command = list(sys.argv) if command is None else list(command)
    if basename:
        command[0] = os.path.basename(command[0])
    if quote:
        for i in range(len(command)):
            command[i] = shlex.quote(command[i])
    return sep.join(command)
