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

from . import genome as genome_

# TODO Move rev_comp to genome, remove split_cigar

_rev_comp = {'A':'T', 'T':'A', 'C':'G', 'G':'C','a':'T', 't':'A', 'c':'G', 'g':'C', 'N':'N', 'n':'N' }
def rev_comp(seq): # reverse complement of string
    return ''.join(_rev_comp.get(nt, 'X') for nt in seq[::-1])


def cond_rev_comp(seq, reverse):
    if reverse:
        return rev_comp(seq)
    return seq


def cond_reverse(qual, reverse):
    return qual[::-1] if reverse else qual


def gc_count(seq):
    return seq.count('C') + seq.count('G')


def gc_content(seq):
    return 100.0 * (seq.count('C') + seq.count('G')) / len(seq)


def split_cigar(cigarstring): ## 57M2I43M13I54M -> (57, M) (2, I) (43, M) (13, I) (54, M)
    prev = 0
    cigarlist = []
    l1 = 0
    l2 = 0
    aligned_len = 0
    for i in range(len(cigarstring)):
        b = ord(cigarstring[i])
        if b < 48 or b > 57:
            ml = int(cigarstring[prev:i])
            prev = i+1
            op = cigarstring[i]
            cigarlist.append([ml, op])
            if cigarstring[i] == 'M' or cigarstring[i] == 'X' or cigarstring[i] == '=':
                l1 += ml
                l2 += ml
                aligned_len += ml
            elif cigarstring[i] == 'D' or cigarstring[i] == 'N':
                l2 += ml
            elif cigarstring[i] == 'I':
                l1 +=ml
            elif cigarstring[i] == 'S':
                l1 +=ml
            elif cigarstring[i] == 'H':
                pass
    return [l1, l2, aligned_len, cigarlist]


def log(string, out=sys.stderr):
    elapsed = str(timedelta(seconds=perf_counter() - log._timer_start))[:-5]
    out.write('%s  %s\n' % (elapsed, string))
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
        return ' '.join(('"%s"' % entry if ' ' in entry else entry) for entry in self._command)

    def finish(self, zero_code_stderr=True):
        """
        Returns True if the process finished successfully.
        """
        out, err = self._process.communicate()

        if self._process.returncode != 0:
            log('ERROR: %s returned code %d' % (self.command_str, self._process.returncode))
            if out:
                log('    Stdout: %s' % _normalize_output(out))
            if err:
                log('    Stderr: %s' % _normalize_output(err))
            return False
        elif err and zero_code_stderr:
            log('Process %s finished with code 0, but has non empty stderr: %s'
                % (self.command_str, _normalize_output(err)))
        return True

    def terminate(self):
        self._process.terminate()


def check_executable(*paths):
    for path in paths:
        if shutil.which(path) is None:
            raise RuntimeError('Cannot find path "%s"' % path)


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


def open_vcf(filename, mode='r', can_be_none=False, **kwargs):
    if filename is None and can_be_none:
        return EmptyContextManager()
    if filename == '-':
        return pysam.VariantFile(sys.stdin if mode == 'r' else sys.stdout, **kwargs)
    if filename.endswith('.gz'):
        return pysam.VariantFile(filename, mode=mode + 'b', **kwargs)
    return pysam.VariantFile(filename, mode=mode, **kwargs)


def get_regions(args, genome, sort=True):
    """
    Returns a list `[NamedInterval]`.
    """
    intervals = []
    if args.regions:
        for region in args.regions:
            if ':' in region:
                interval = genome_.NamedInterval.parse(region, genome)
            else:
                chrom_id = genome.chrom_id(region)
                end = genome.chrom_len(chrom_id)
                interval = genome_.NamedInterval(chrom_id, 0, end, genome)
            interval.trim(genome)
            intervals.append(interval)

    elif args.regions_file:
        with open_possible_gzip(args.regions_file) as inp:
            for line in inp:
                if line.startswith('#'):
                    continue

                line = line.strip().split('\t')
                chrom_id = genome.chrom_id(line[0])
                start = int(line[1])
                end = int(line[2])
                name = line[3] if len(line) > 3 else None
                interval = genome_.NamedInterval(chrom_id, start, end, genome, name)
                interval.trim(genome)
                intervals.append(interval)

    else:
        for chrom_id, length in enumerate(genome.chrom_lengths):
            intervals.append(genome_.NamedInterval(chrom_id, 0, length, genome, name=genome.chrom_name(chrom_id)))

    if sort:
        intervals.sort()
    return intervals


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


def fetch_iterator(obj, args, genome, no_duplicates=True, start_attr=False):
    """
    Returns sorted iterator over pysam object with that has a `fetch` method.
    Looks at `args.regions` and `args.regions_file`. If both are `None`, returns `obj.fetch()`.

    If no_duplicates is True, all duplicates will be removed. However, for that it is necessary to be able to get
    the start position of each entry. If start_attr is True, it is able to use `entry.start`, otherwise,
    it is possible to call `int(entry[1])`.
    """
    if not args.regions and not args.regions_file:
        return obj.fetch()

    regions = get_regions(args, genome, sort=True)
    if not no_duplicates:
        return itertools.chain(*(obj.fetch(region.chrom_name(genome), region.start, region.end) for region in regions))
    return _fetch_regions_wo_duplicates(obj, regions, genome, start_attr)


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


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


def common_prefix(seq0, *seqs):
    try:
        for i, nt in enumerate(seq0):
            for seq in seqs:
                if seq[i] != nt:
                    return i
    except IndexError:
        return i
    return len(seq0)


def common_suffix(seq0, *seqs):
    try:
        for i in itertools.count(1):
            nt = seq0[-i]
            for seq in seqs:
                if seq[-i] != nt:
                    return i - 1
    except IndexError:
        return i - 1


class NonOverlappingSet:
    """
    Stores non-overlapping intervals and allows for fast retrieval of overlapping intervals.
    """
    def __init__(self, starts, ends):
        assert len(starts) == len(ends)
        self._starts = np.array(starts)
        self._ends = np.array(ends)
        assert np.all(self._ends[:-1] <= self._starts[1:])

    @classmethod
    def from_start_end_pairs(cls, pairs):
        self = cls.__new__(cls)
        self._starts = np.fromiter(map(operator.itemgetter(0), pairs), np.int32, len(pairs))
        self._ends = np.fromiter(map(operator.itemgetter(1), pairs), np.int32, len(pairs))
        assert np.all(self._ends[:-1] <= self._starts[1:])
        return self

    @classmethod
    def from_variants(cls, variants):
        self = cls.__new__(cls)
        self._starts = np.fromiter(map(operator.attrgetter('start'), variants), np.int32, len(variants))
        self._ends = self._starts + [len(variant.ref) for variant in variants]
        assert np.all(self._ends[:-1] <= self._starts[1:])
        return self

    @classmethod
    def from_regions(cls, regions):
        self = cls.__new__(cls)
        self._starts = np.fromiter(map(operator.attrgetter('start'), regions), np.int32, len(regions))
        self._ends = np.fromiter(map(operator.attrgetter('end'), regions), np.int32, len(regions))
        assert len(set(map(operator.attrgetter('chrom_id'), regions))) <= 1
        assert np.all(self._ends[:-1] <= self._starts[1:])
        return self

    @classmethod
    def from_dupl_regions(cls, dupl_regions):
        regions1 = list(map(operator.attrgetter('region1'), dupl_regions))
        return cls.from_regions(regions1)

    def select(self, start, end):
        start_ix = self._ends.searchsorted(start, side='right')
        end_ix = self._starts.searchsorted(end, side='left')
        return start_ix, end_ix

    def select_by_pos(self, pos):
        start_ix = self._ends.searchsorted(pos, side='right')
        if start_ix == len(self._starts) or pos < self._starts[start_ix]:
            return None
        return start_ix

    def is_empty(self, start, end):
        start_ix, end_ix = self.select(start, end)
        return start_ix == end_ix

    @property
    def starts(self):
        return self._starts

    @property
    def ends(self):
        return self._ends

    def __len__(self):
        return len(self._starts)


def str_count(count, word):
    """
    str_count(10, 'word') -> '10 words'
    str_count(1, 'word') -> '1 word'
    """
    return '%d %s%s' % (count, word, '' if count == 1 else 's')


def fmt_len(length):
    if length < 1000:
        return '%dbp' % length
    elif length < 1000000:
        return '%.1f Kb' % (length / 1000)
    else:
        return '%.1f Mb' % (length / 1e6)


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
