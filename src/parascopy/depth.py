#!/usr/bin/env python3

import operator
import argparse
import pysam
import sys
import csv
import os
import numpy as np
import scipy.stats
import traceback
import shutil
import re
import gzip
import pkgutil
try:
    import multiprocessing
except ImportError:
    pass

from . import pool_reads
from .inner import common
from .inner.genome import Genome, Interval
from .inner.cigar import Cigar, Operation
from . import long_version, __pkg_name__


def _save_temp_windows(temp_windows, windows, genome, out):
    if not temp_windows:
        return
    first = temp_windows[0]
    last = temp_windows[-1]
    first_start = first.start
    seq = genome.fetch_interval(Interval(first.chrom_id, first_start, last.end))

    for window in temp_windows:
        window_seq = seq[window.start - first_start : window.end - first_start]
        if 'N' in window_seq:
            continue
        window.gc_content = int(common.gc_content(window_seq))
        windows.append(window)
        out.write('{}\t{}\n'.format(window.to_bed(genome), window.gc_content))


def _load_windows(bed_lines, genome, out):
    """
    Returns sorted two lists: windows and their GC contents.
    """
    MAX_DIST = 1000

    windows = []
    temp_windows = []

    prev_window = None
    window_len = None
    for line in bed_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        chrom_name, start, end = line.split('\t')[:3]
        start = int(start)
        end = int(end)
        chrom_id = genome.chrom_id(chrom_name)
        window = Interval(chrom_id, start, end)

        if temp_windows and temp_windows[-1].distance(window) > MAX_DIST:
            _save_temp_windows(temp_windows, windows, genome, out)
            temp_windows.clear()
        temp_windows.append(window)

        if window_len is None:
            window_len = len(window)
        elif window_len != len(window):
            common.log('ERROR: All windows should have the same size!')
            common.log('Window {} (size {}) does not match previous window size {}'.format(
                window.to_str(genome), len(window), window_len))
            exit(1)
        elif prev_window.chrom_id > window.chrom_id or (prev_window.chrom_id == window.chrom_id
                and prev_window.end > window.start):
            common.log('ERROR: Windows should be sorted and non-overlapping')
            common.log('Window {} follows window {}'.format(window.to_str(genome), prev_window.to_str(genome)))
        prev_window = window

    _save_temp_windows(temp_windows, windows, genome, out)
    return windows


def _summarize_windows(windows, tail_windows, stratify_gc):
    n_chroms = len(set(map(operator.attrgetter('chrom_id'), windows)))
    n_windows = len(windows)
    common.log('Using {:,} windows on {} chromosomes'.format(n_windows, n_chroms))

    gc_content = list(map(operator.attrgetter('gc_content'), windows))
    gc_bin_count = np.bincount(gc_content)

    bin_starts = np.arange(0, 100, 10)
    bin_ends = np.arange(10, 110, 10)
    for start, end in zip(bin_starts, bin_ends):
        common.log('    GC content {:2.0f} -{:3.0f}: {:6,d} windows'.format(start, min(100.0, end),
            np.sum(gc_bin_count[start : end])))

    gc_cumsum = np.cumsum(gc_bin_count)
    left_bound = np.where(gc_cumsum >= tail_windows)[0][0]
    right_bound = np.where(n_windows - gc_cumsum >= tail_windows)[0][-1]
    if stratify_gc:
        common.log('Using GC-content in range [{}, {}]'.format(left_bound, right_bound))
    return left_bound, right_bound + 1


class WindowCounts:
    def __init__(self, params):
        self.low_mapq_thresh = params.low_mapq_thresh
        self.max_mate_dist = params.max_mate_dist

        # Not paired reads will also get to _depth_read1.
        self._depth_read1 = 0
        self._depth_read2 = 0
        self._low_mapq_reads = 0
        self._clipped_reads = 0
        self._unpaired_reads = 0

    def add_read(self, read, cigar, trust_proper_pair=False, look_at_oa=False):
        """
        If trust_proper_pair, look only at flag is_proper_pair (ignore chromosome and distance).
        """
        if read.is_read2:
            self._depth_read2 += 1
        else:
            self._depth_read1 += 1
        if read.mapping_quality < self.low_mapq_thresh:
            self._low_mapq_reads += 1

        base_qualities = read.query_qualities
        has_clipping = cigar.has_true_clipping(base_qualities)
        if look_at_oa:
            orig_aln = read.get_tag('OA').split(',')
            orig_strand = orig_aln[2] == '+'
            orig_cigar = Cigar(orig_aln[3])
            has_clipping &= orig_cigar.has_true_clipping(common.cond_reverse(base_qualities, strand=orig_strand))
        if has_clipping:
            self._clipped_reads += 1

        if read.is_paired:
            if not read.is_proper_pair:
                self._unpaired_reads += 1
            elif trust_proper_pair:
                pass
            elif read.reference_id != read.next_reference_id \
                    or abs(read.reference_start - read.next_reference_start) > self.max_mate_dist:
                self._unpaired_reads += 1

    def __iadd__(self, other):
        self._depth_read1 += other._depth_read1
        self._depth_read2 += other._depth_read2
        self._low_mapq_reads += other._low_mapq_reads
        self._clipped_reads += other._clipped_reads
        self._unpaired_reads += other._unpaired_reads
        return self

    @property
    def total_reads(self):
        return self._depth_read1 + self._depth_read2

    @property
    def depth_read1(self):
        return self._depth_read1

    @property
    def depth_read2(self):
        return self._depth_read2

    def write_to(self, outp):
        outp.write('{}\t{}\t{}\t{}\t{}'.format(self._depth_read1, self._depth_read2,
            self._low_mapq_reads, self._clipped_reads, self._unpaired_reads))

    def passes(self, params):
        total = self.total_reads
        if self._depth_read1 > 0xffff or self._depth_read2 > 0xffff:
            return False
        if not total:
            return True
        return self._low_mapq_reads / total <= params.low_mapq_ratio \
            and self._clipped_reads / total <= params.clipped_ratio \
            and self._unpaired_reads / total <= params.unpaired_ratio


def get_read_middle(read, cigar):
    read_pos = 0
    ref_pos = read.reference_start
    end_read_pos = len(read.query_sequence) // 2

    for length, op in cigar:
        if op.consumes_both():
            if read_pos + length > end_read_pos:
                return ref_pos + end_read_pos - read_pos
            read_pos += length
            ref_pos += length

        elif op.consumes_ref():
            ref_pos += length
        elif op.consumes_read():
            if read_pos + length > end_read_pos:
                return None if op == Operation.Soft else ref_pos - 1
            read_pos += length
    return None


class Windows:
    def __init__(self, start_ix, windows, genome, window_size, are_long_reads):
        self._start_ix = start_ix
        self._window_size = window_size
        self._half_size = max(1, window_size // 2)
        self._starts = np.fromiter(map(operator.attrgetter('start'), windows), np.int32, len(windows))
        self._chrom = genome.chrom_name(windows[0].chrom_id)
        self._reg_start = windows[0].start
        self._reg_end = windows[-1].end
        self.get_windows = self._get_all_windows if are_long_reads else self._get_middle_window

    # # Function, that is either `get_middle_window` or `get_all_windows` depending on `are_long_reads` switch.
    # def get_windows(self, read, cigar):
    #     pass

    def _get_middle_window(self, read, cigar):
        middle = get_read_middle(read, cigar)
        if middle is None:
            return ()
        ix = self._starts.searchsorted(middle, side='right') - 1
        if ix == -1 or middle >= self._starts[ix] + self._window_size:
            return ()
        return (ix,)

    def _get_all_windows(self, read, cigar):
        """
        Identify all windows, covered by this read (at least 50% of the read maps to the window).
        """
        if read.reference_start >= self._reg_end or read.reference_end <= self._reg_start:
            return
        ref_start = read.reference_start
        window_ix = max(0, self._starts.searchsorted(ref_start, side='right') - 1)
        window_start = self._starts[window_ix]
        window_end = window_start + self._window_size
        window_cov = 0

        for _, ref_pos0 in cigar.aligned_pairs():
            ref_pos = ref_pos0 + ref_start
            while window_end <= ref_pos:
                if window_cov >= self._half_size:
                    yield window_ix
                window_cov = 0
                window_ix += 1
                if window_ix == len(self._starts):
                    return
                window_start = self._starts[window_ix]
                window_end = window_start + self._window_size

            if window_start <= ref_pos:
                window_cov += 1

        if window_cov >= self._half_size:
            yield window_ix

    # def get_windows(self, read, cigar, many=False):
    #     return self._get_all_windows(read, cigar) if many else self._get_middle_window(read, cigar)

    def fetch_from(self, bam_file):
        return common.checked_fetch_coord(bam_file, self._chrom, self._reg_start, self._reg_end)

    @property
    def start_ix(self):
        return self._start_ix

    @property
    def n_windows(self):
        return len(self._starts)


def _get_fetch_regions(windows, genome, window_size, are_long_reads, max_distance=100):
    start_ix = 0
    regions = []
    for i in range(1, len(windows)):
        assert not windows[i - 1].intersects(windows[i])
        if windows[i - 1].distance(windows[i]) > max_distance:
            regions.append(Windows(start_ix, windows[start_ix : i], genome, window_size, are_long_reads))
            start_ix = i
    if windows:
        regions.append(Windows(start_ix, windows[start_ix :], genome, window_size, are_long_reads))
    return regions


_DEFAULT_LOW_MAPQ = 10


class Params:
    def __init__(self,
            low_mapq_thresh=_DEFAULT_LOW_MAPQ,
            max_mate_dist=pool_reads.DEFAULT_MATE_DISTANCE,
            window_filtering_mult=1):
        self.low_mapq_thresh = low_mapq_thresh
        self.max_mate_dist = max_mate_dist

        self.window_size = None
        self.low_mapq_ratio = 0.1 * window_filtering_mult
        self.clipped_ratio = 0.1 * window_filtering_mult
        self.unpaired_ratio = 0.1 * window_filtering_mult
        self.neighbours = 1
        self.neighbours_dist = None
        self.loess_frac = None
        self.gc_bounds = None
        self._ploidy = 2
        self.long_reads = False
        self.stratify_gc = True

    def set_ploidy(self, ploidy):
        self._ploidy = ploidy

    def set_neighbours_dist(self):
        self.neighbours_dist = self.window_size * self.neighbours - self.window_size // 2

    def describe(self):
        s =  'Read depth parameters (ploidy = {}):\n'.format(self._ploidy)
        s += '    Use window size {} bp.\n'.format(self.window_size)
        s += '    Window is irregular in a sample if there are more than:\n'
        s += '    -  {:.1f}% reads with low mapping quality,\n'.format(self.low_mapq_ratio * 100)
        s += '    -  {:.1f}% reads with clipping,\n'.format(self.clipped_ratio * 100)
        s += '    -  {:.1f}% reads without pair.\n'.format(self.unpaired_ratio * 100)
        s += '    Remove {} window(s) on both sides of an irregular window.\n'.format(self.neighbours)
        if self.loess_frac is not None:
            s += '    Using LOESS fraction = {:.1f}.\n'.format(self.loess_frac)
        if self.gc_bounds is not None:
            s += '    Use windows with GC-content in [{}..{}].\n'.format(*self.gc_bounds)
        return s

    @classmethod
    def from_args(cls, args, window_size, bounds):
        self = cls(args.low_mapq, args.mate_dist)
        self.set_ploidy(args.ploidy)
        self.window_size = window_size
        self.low_mapq_ratio = args.low_mapq_perc / 100.0
        self.clipped_ratio = args.clipped_perc / 100.0
        self.unpaired_ratio = args.unpaired_perc / 100.0
        self.neighbours = args.neighbours
        self.set_neighbours_dist()
        self.loess_frac = args.loess_frac
        self.gc_bounds = tuple(bounds) if bounds is not None else None
        self.long = args.long
        self.stratify_gc = not args.long and args.stratify_gc
        return self

    def equals(self, other):
        return self._ploidy == other._ploidy \
            and self.window_size == other.window_size \
            and self.low_mapq_ratio == other.low_mapq_ratio \
            and self.clipped_ratio == other.clipped_ratio \
            and self.unpaired_ratio == other.unpaired_ratio \
            and self.neighbours == other.neighbours \
            and self.neighbours_dist == other.neighbours_dist \
            and self.loess_frac == other.loess_frac \
            and self.gc_bounds == other.gc_bounds \
            and self.long == other.long \
            and self.stratify_gc == other.stratify_gc

    @property
    def ploidy(self):
        return self._ploidy


def _filter_windows(windows, params, depth1, depth2, keep_window):
    n_windows = len(depth1)
    median_cov = np.median(depth1[keep_window])
    max_cov = median_cov * 10
    keep_window &= depth1 <= max_cov
    keep_window &= depth2 <= max_cov

    neighbours = params.neighbours
    neighbours_dist = params.neighbours_dist
    for i in np.where(~keep_window)[0]:
        for j in range(max(0, i - neighbours), min(n_windows, i + neighbours + 1)):
            if i != j and windows[j].distance(windows[i]) < neighbours_dist:
                keep_window[j] = False


def loess(x, y, xout, frac=2/3, deg=1, w=None):
    ixs = np.argsort(x)
    x = x[ixs]
    y = y[ixs]
    in_weight = w[ixs] if w is not None else None

    n = len(x)
    n_frac = int(round(n * frac))
    size = x[-1] - x[0]
    assert size > 0.0

    res = np.full(len(xout), np.nan)
    for i, xval in enumerate(xout):
        a = x.searchsorted(xval, 'left')
        b = x.searchsorted(xval, 'right')
        if b - a >= n_frac:
            res[i] = np.mean(y[a:b])
            continue

        if b - a < n_frac:
            rem = n_frac - b + a
            left = min(a, (rem // 2))
            right = min(n - b, rem - left)
            a -= left
            b += right
            assert a >= 0 and b <= n

        sub_x = x[a:b]
        sub_y = y[a:b]
        weight = common.tricube_kernel((sub_x - xval) / size)
        if in_weight is not None:
            weight *= in_weight[a:b]
        assert max(weight) > 0
        try:
            coef = np.polyfit(sub_x, sub_y, deg=deg, w=weight)
        except ValueError:
            sys.stderr.write('Polyfit failed for unknown reason (frac={:.5f}, deg={}) at x={}\n'
                .format(frac, deg, xval))
            sys.stderr.write('    subx: {}\n    suby: {}\n    w: {}\n'.format(list(sub_x), list(sub_y), list(weight)))
            raise
        res[i] = np.polyval(coef, xval)
    return res


def _predict_variance(depth_values, curr_keep_window, gc_windows, min_windows=10):
    """
    To estimate variance loess use all GC contents with more than min_windows_ratio out of total number of used windows.
    """
    x = []
    y = []
    w = []
    total_count = sum(curr_keep_window)
    for gc_content, curr_gc_windows in enumerate(gc_windows):
        filt_gc_windows = np.compress(curr_keep_window[curr_gc_windows], curr_gc_windows)
        if len(filt_gc_windows) >= min_windows:
            x.append(gc_content)
            y.append(np.var(depth_values[filt_gc_windows]))
            w.append(len(filt_gc_windows) / total_count)

    x = np.array(x)
    y = np.array(y)
    w = np.array(w)
    # Use all points, but weight by the number of windows in each bin (weighted by the distance by default).
    return loess(x, y, xout=np.arange(101), w=w, frac=1)


# def _get_faraway_pairs(windows, distance):
#     """
#     Returns a generator of window indices that satisfy two conditions:
#         - windows ixs[i] and ixs[i] + 1 are closeby (distance is less than window size),
#         - windows ixs[i] and ixs[i + 1] are far away (threshold in the function arguments).
#     """
#     last_end = -np.inf
#     n = len(windows)
#     for i, w in enumerate(windows):
#         if last_end + distance <= w.start and i + 1 < n and windows[i + 1].start < w.end + len(w):
#             yield i
#             last_end = w.end


# def _calculate_nearby_correlation(sample, windows, window_counts, mean_read_len):
#     MIN_OBSERVATIONS = 10
#     MAX_DIST = 50_000
#     ixs = _get_faraway_pairs(windows, min(mean_read_len, MAX_DIST))

#     x = []
#     y = []
#     for i in ixs:
#         x.append(window_counts[i].depth_read1)
#         y.append(window_counts[i].depth_read1)
#         print(x[-1], y[-1])
#     if len(x) < MIN_OBSERVATIONS:
#         raise RuntimeError(f'ERROR: Could not calculate correlations for {sample}. '
#             'Consider using more background windows.')
#     return np.corrcoef((x, y))


def _calculate_correlations(sample, windows, window_counts, mean_read_len):
    MIN_OBSERVATIONS = 10
    MAX_DIST = 50_000

    dist_thresh = int(min(round(mean_read_len), MAX_DIST))
    window_size = len(windows[0])
    n_shifts = (dist_thresh - 1) // window_size + 1
    # Will be a matrix NxM, where M = n_shifts, and N = number of sufficiently far-away windows.
    nearby_counts = []

    i = 0
    n = len(windows)
    while i < n:
        w = windows[i]
        curr_counts = np.full(n_shifts, np.nan)
        curr_counts[0] = window_counts[i].depth_read1
        print(i, i, 0, window_counts[i].depth_read1, sep='\t')

        for j in range(i + 1, n):
            u = windows[j]
            d = w.start_distance(u)
            if d >= dist_thresh:
                i = j
                break
            curr_counts[d // window_size] = window_counts[j].depth_read1
            print(i, j, d // window_size, window_counts[j].depth_read1, sep='\t')
        else:
            break
        nearby_counts.append(curr_counts)

    nearby_counts = np.array(nearby_counts)
    print('# Correlations:')
    x = nearby_counts[:, 0]
    for i in range(1, n_shifts):
        y = nearby_counts[:, i]
        avail = ~np.isnan(y)
        if np.sum(avail) >= MIN_OBSERVATIONS:
            c = scipy.stats.pearsonr(x[avail], y[avail])
            print(f'#{i}\t{np.sum(avail)}\t{c.statistic}')


def _summarize_sample(sample, sample_window_counts, mean_read_len, params, windows, out, res):
    n_windows = len(sample_window_counts)
    keep_window = np.ones(n_windows, dtype=np.bool_)
    depth1 = np.zeros(n_windows, dtype=np.uint32)
    depth2 = np.zeros(n_windows, dtype=np.uint32)

    for window_ix, counts in enumerate(sample_window_counts):
        keep_window[window_ix] = counts.passes(params)
        depth1[window_ix] = counts.depth_read1
        depth2[window_ix] = counts.depth_read2
    _filter_windows(windows, params, depth1, depth2, keep_window)
    if not np.any(keep_window):
        raise ValueError('Sample "{}" does not pass filters for any windows'.format(sample))

    for window_ix, counts in enumerate(sample_window_counts):
        out.write('{}\t{}\t'.format(sample, window_ix))
        counts.write_to(out)
        out.write('\t{}\n'.format('T' if keep_window[window_ix] else 'F'))

    gc_contents = np.array([window.gc_content for window in windows], dtype=np.int32)
    gc_windows = [np.where(gc_contents == gc_content)[0] for gc_content in range(101)]

    # if params.long:
    #     cor = _calculate_correlations(sample, windows, sample_window_counts, mean_read_len)
    #     out.write(f'# cor {sample} {cor:.10f}\n')

    for read_end, depth_values in enumerate((depth1, depth2), start=1):
        if read_end == 2 and max(depth_values) == 0:
            continue
        if params.stratify_gc:
            mean_loess = loess(gc_contents[keep_window], depth_values[keep_window],
                xout=np.arange(101), frac=params.loess_frac)
            assert np.all(~np.isnan(mean_loess))
            var_loess = _predict_variance(depth_values, keep_window, gc_windows)
            assert np.all(~np.isnan(var_loess))
        else:
            m = np.mean(depth_values)
            v = np.var(depth_values)
            mean_loess = [m] * 101
            var_loess = [v] * 101

        for gc_content, curr_gc_windows in enumerate(gc_windows):
            filt_gc_windows = np.compress(keep_window[curr_gc_windows], curr_gc_windows)
            line = '{}\t{}\t{}\t{}/{}\t'.format(sample, gc_content, read_end,
                len(filt_gc_windows), len(curr_gc_windows))
            values = depth_values[filt_gc_windows]
            if len(values):
                line += '{:.5f}\t{:.5f}\t{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\t'.format(
                    np.mean(values), np.var(values), *np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0)))
            else:
                line += 'nan\tnan\t*\t'

            m = mean_loess[gc_content]
            v = max(var_loess[gc_content], m + 0.01)
            nbinom_n = m * m / (v - m)
            nbinom_p = m / v
            line += '{:.5f}\t{:.5f}\t{:.8f}\t{:.8f}\n'.format(m, var_loess[gc_content], nbinom_n, nbinom_p)
            res.append(line)


def _single_file_depth(bam_index, bam_wrapper, windows, fetch_regions, genome_filename, out_prefix, params):
    """
    bam_file: either str or pysam.AlignmentFile.
    Returns dictionary { sample: [_Window] }.
    """
    need_close = False
    common.log('    Calculating background depth for file {:3}: {}'.format(bam_index, bam_wrapper.filename))

    with bam_wrapper.open_bam_file(genome_filename) as bam_file:
        # NOTE: Ideally, we should split by sample, and, potentially, by read group?
        sum_read_len = 0
        total_reads = 0

        n_windows = len(windows)
        read_groups = bam_wrapper.read_groups()
        samples = list(set(map(operator.itemgetter(1), read_groups.values())))
        window_counts = { sample: [WindowCounts(params) for _ in range(n_windows)] for sample in samples }

        for region in fetch_regions:
            start_ix = region.start_ix
            n_windows = region.n_windows
            read_iter = region.fetch_from(bam_file)

            for read in read_iter:
                if read.flag & 3844:
                    continue

                sum_read_len += len(read.query_sequence)
                total_reads += 1
                cigar = Cigar.from_pysam_tuples(read.cigartuples)
                sample = read_groups[read.get_tag('RG') if read.has_tag('RG') else None][1]
                for window_ix in region.get_windows(read, cigar):
                    window_counts[sample][start_ix + window_ix].add_read(read, cigar)

        mean_read_len = sum_read_len / total_reads
        if (mean_read_len >= 500) != params.long:
            common.log('File {} contains {} reads (mean length = {:.0f}), but `--long` argument was {}used'.format(
                bam_wrapper.filename, 'long' if mean_read_len >= 500 else 'short', mean_read_len,
                '' if params.long else 'not '))

        res = []
        for sample in samples:
            with open(f'{out_prefix}.{sample}.csv', 'w') as out:
                _summarize_sample(sample, window_counts[sample], mean_read_len, params, windows, out, res)
    return res


def _try_single_file_depth(*args, **kwargs):
    try:
        return _single_file_depth(*args, **kwargs)
    except Exception:
        fmt_exc = traceback.format_exc()
        common.log('ERROR:\n{}'.format(fmt_exc))
        raise


def all_files_depth(bam_wrappers, windows, fetch_regions, genome_filename, threads, params, out_prefix, out_means):
    if threads <= 1:
        for bam_index, bam_wrapper in enumerate(bam_wrappers, 1):
            for line in _single_file_depth(bam_index, bam_wrapper, windows, fetch_regions, genome_filename, out_prefix,
                    params):
                out_means.write(line)
            out_means.flush()
        return

    successes = []
    def callback(res):
        successes.append(None)
        for line in res:
            out_means.write(line)
        out_means.flush()

    def err_callback(exc):
        common.log('Thread finished with an exception: {}'.format(exc))
        # os._exit(1)
        pool.terminate()

    pool = multiprocessing.Pool(threads)
    for bam_index, bam_wrapper in enumerate(bam_wrappers, 1):
        pool.apply_async(_try_single_file_depth, callback=callback, error_callback=err_callback,
            args=(bam_index, bam_wrapper, windows, fetch_regions, genome_filename, out_prefix, params))
    pool.close()
    pool.join()

    if len(successes) != len(bam_wrappers):
        raise RuntimeError('One of the threads failed')


def _write_means_header(out_means, windows, params, tail_windows):
    common.log('Calculate background depth')
    window_size = len(windows[0])
    out_means.write(f'# command: {common.command_to_str()}\n')
    out_means.write(f'# ploidy: {params.ploidy}\n')
    out_means.write(f'# window size: {window_size}\n')
    out_means.write(f'# number of windows: {len(windows)}\n')
    out_means.write(f'# low MAPQ threshold: {params.low_mapq_thresh}\n')
    out_means.write(f'# max mate distance: {params.max_mate_dist}\n')
    out_means.write(f'# max percentage of low MAPQ reads: {params.low_mapq_ratio * 100:.1f}\n')
    out_means.write(f'# max percentage of clipped reads: {params.clipped_ratio * 100:.1f}\n')
    out_means.write(f'# max percentage of unpaired reads: {params.unpaired_ratio * 100:.1f}\n')
    out_means.write(f'# skip neighbour windows: {params.neighbours}\n')
    out_means.write(f'# loess fraction: {params.loess_frac:.4f}\n')
    out_means.write(f'# tail windows: {tail_windows}\n')
    if params.gc_bounds is not None:
        out_means.write(f'# GC-content range: [{params.gc_bounds[0]} {params.gc_bounds[1] + 1}]\n')
    else:
        out_means.write(f'# GC-content range: None\n')
    out_means.write(f'# long: {params.long}\n')
    out_means.write(f'# stratify GC: {params.stratify_gc}\n')
    out_means.write('sample\tgc_content\tread_end\tn_windows\tmean\tvar\tquartiles'
        '\tmean_loess\tvar_loess\tnbinom_n\tnbinom_p\n')


class Depth:
    def __init__(self, filename, samples, window_filtering_mult=1):
        """
        Input file needs to contain:
            - lines
                # window size: INT
                # ... MAPQ ...: FLOAT        (low_mapq_perc, optional)
                # ... clipped ...: FLOAT     (clipped_perc, optional)
                # ... unpaired ...: FLOAT    (unpaired_perc, optional)
                # ... neighbour ...: INT     (neighbour windows, optional)
                # GC-content range: [INT, INT]
            - columns "sample, gc_content, read_end, nbinom_n, nbinom_p".
        """
        if os.path.isdir(filename):
            filename = os.path.join(filename, 'depth.csv')
        self._params = Params(window_filtering_mult=window_filtering_mult)

        req_fields = 'sample gc_content read_end nbinom_n nbinom_p'.split()
        with common.open_possible_gzip(filename) as inp:
            fieldnames = None
            for line in inp:
                if not line.startswith('#'):
                    fieldnames = line.strip().split('\t')
                    break
                if line.count(':') == 1:
                    key, value = map(str.strip, line[1:].split(':'))
                    if key == 'ploidy':
                        self._params.set_ploidy(int(value))
                    elif 'window size' in key:
                        self._params.window_size = int(value)
                    elif 'MAPQ threshold' in key:
                        self._params.low_mapq_thresh = int(value)
                    elif 'mate' in key:
                        self._params.max_mate_dist = int(value)

                    elif 'MAPQ' in key:
                        self._params.low_mapq_ratio = float(value) / 100 * window_filtering_mult
                    elif 'clipped' in key:
                        self._params.clipped_ratio = float(value) / 100 * window_filtering_mult
                    elif 'unpaired' in key:
                        self._params.unpaired_ratio = float(value) / 100 * window_filtering_mult
                    elif 'neighbour' in key:
                        self._params.neighbours = int(value)
                    elif key == 'GC-content range':
                        if value == 'None':
                            self._params.gc_bounds = None
                        else:
                            m = re.search(r'\[(\d+)[, \t]+(\d+)\]', value)
                            self._params.gc_bounds = (int(m.group(1)), int(m.group(2)) + 1)
                    elif key == 'long':
                        self.long = bool(value)
                    elif key == 'stratify QC':
                        self.stratify_gc = bool(value)

            if self._params.window_size is None:
                common.log('ERROR: Input file does not contain line "# window size: INT"')
                exit(1)
            elif fieldnames is None or set(req_fields) - set(fieldnames):
                common.log('ERROR: Input file does not contain some of the following fields: {}'
                    .format(', '.join(req_fields)))
                exit(1)
            self._params.set_neighbours_dist()

            reader = csv.DictReader(inp, delimiter='\t', fieldnames=fieldnames)
            # Matrix n_samples x 101 x 2
            #     <sample> x <gc_content> x <nbinom params: first n, second p>.
            self._nbinom_params = np.full((len(samples), 101, 2), np.nan)

            for row in reader:
                if row['sample'] not in samples or row['read_end'] != '1':
                    continue
                sample_id = samples.id(row['sample'])
                gc_content = int(row['gc_content'])

                if not np.isnan(self._nbinom_params[sample_id, gc_content, 0]):
                    common.log('WARN: sample {} and GC content {} appear twice in the depth file'.format(
                        sample, gc_content))
                n = float(row['nbinom_n'])
                p = float(row['nbinom_p'])
                if (np.isnan(n) or np.isnan(p)) and self.gc_content_defined(gc_content):
                    raise RuntimeError('ERROR: Sample {} is missing read depth parameters (see GC-content {})'
                        .format(row['sample'], gc_content))
                self._nbinom_params[sample_id, gc_content] = (n, p)
        self._nbinom_params[:, :, 0] /= self._params.ploidy

    @classmethod
    def from_filenames(cls, filenames, samples, *args, **kwargs):
        depth = None
        has_samples = None

        for filename in filenames:
            curr_depth = Depth(filename, samples, *args, **kwargs)
            curr_has_samples = ~np.any(np.isnan(curr_depth._nbinom_params), axis=(1, 2))
            assert len(samples) == len(curr_has_samples)

            if depth is None:
                depth = curr_depth
                has_samples = curr_has_samples
                continue

            if not depth._params.equals(curr_depth._params):
                raise RuntimeError(
                    'Cannot merge background read depth from "{}" and "{}": parameters do not match.'
                    .format(filenames[0], filename))

            for sample_id in np.where(curr_has_samples)[0]:
                if has_samples[sample_id]:
                    raise RuntimeError('Cannot load background read depth: sample {} appears twice.'
                        .format(samples[sample_id]))
                depth._nbinom_params[sample_id] = curr_depth._nbinom_params[sample_id]
                has_samples[sample_id] = True

        if depth._params.gc_bounds is None:
            gc_bounds = slice(0, 101)
        else:
            gc_bounds = slice(depth._params.gc_bounds[0], depth._params.gc_bounds[1])
        for sample_id, sample in enumerate(samples):
            if np.any(np.isnan(depth._nbinom_params[sample_id, gc_bounds, :])):
                raise RuntimeError('Cannot load background read depth: Sample {} is missing some/all values.'
                    .format(sample))
        return depth

    def at(self, sample_id, gc_content):
        return self._nbinom_params[sample_id, gc_content, :]

    @property
    def window_size(self):
        return self._params.window_size

    @property
    def neighbours(self):
        return self._params.neighbours

    @property
    def neighbours_dist(self):
        return self._params.neighbours_dist

    def window_passes(self, counts):
        return counts.passes(self._params)

    def gc_content_defined(self, gc_content):
        if self._params.gc_bounds is None:
            return True
        bound_left, bound_right = self._params.gc_bounds
        return bound_left <= gc_content < bound_right

    @property
    def gc_bounds(self):
        return self._params.gc_bounds

    @property
    def params(self):
        return self._params


def check_duplicated_samples(bam_wrappers):
    samples = {}
    for i, bam_wrapper in enumerate(bam_wrappers):
        for sample in bam_wrapper.present_samples():
            if sample in samples and samples[sample] != i:
                confl_filename = bam_wrappers[samples[sample]].filename
                common.log('ERROR: Sample {} appears in two input files: {} and {}'
                    .format(sample, confl_filename, bam_wrapper.filename))
                exit(1)
            samples[sample] = i


def _concat_files(bam_wrappers, prefix, out):
    out.write('# {}\n'.format(common.command_to_str()).encode())
    out.write(b'sample\twindow_ix\tdepth1\tdepth2\tlow_mapq\tclipped\tunpaired\tuse\n')

    all_samples = set()
    for bam_wrapper in bam_wrappers:
        all_samples.update(map(operator.itemgetter(1), bam_wrapper.read_groups().values()))
    all_samples = sorted(all_samples)

    for sample in all_samples:
        filename = '{}.{}.csv'.format(prefix, sample)
        with open(filename, 'rb') as f:
            shutil.copyfileobj(f, out)
        os.remove(filename)


def _write_readme(out_dir):
    with open(os.path.join(out_dir, 'README'), 'w') as out:
        out.write('Output file "depth.csv" contains background read depth estimation binned according to GC-content.\n')
        out.write('Output files "windows.bed.gz" and "window_depth.csv.gz" store read depth '
            'for each window and each sample.\n')
        out.write('These files are not important for downstream analysis and can be removed.\n')


def _load_background_regions(genome):
    g = genome.lower().replace('hg19', 'grch37').replace('hg38', 'grch38')
    if g == 'chm13-fast':
        g = 'chm13'

    try:
        bed_data = pkgutil.get_data(__pkg_name__, os.path.join('data', 'depth_regions', g + '.bed.gz'))
    except FileNotFoundError:
        # Based on documentation, returned data will be None, if the file does not exist.
        # In practice, FileNotFoundError is thrown.
        # To handle both cases, set data to None manually.
        bed_data = None

    if bed_data is None:
        raise ValueError(f'Could not identify background regions for {genome}. '
            'Possible genomes are: GRCh37|hg19, GRCh38|hg38, CHM13; with an optional suffix `-fast`.')
    bed_data = gzip.decompress(bed_data)
    return bed_data.decode().split('\n')


def main(prog_name, in_argv):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Calculate read depth and variance in given genomic windows.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} (-i <bam> [...] | -I <bam-list>) (-g hg19|hg38 | -b <bed>) -f <fasta> -o <dir> [arguments]'
            .format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    inp_me = io_args.add_mutually_exclusive_group(required=True)
    inp_me.add_argument('-i', '--input', metavar='<file>', nargs='+',
        help='Input indexed BAM/CRAM files.\n'
            'All entries should follow the format "filename[::sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input-list.')
    inp_me.add_argument('-I', '--input-list', metavar='<file>',
        help='A file containing a list of input BAM/CRAM files.\n'
            'All lines should follow the format "filename[ sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input.\n\n')

    reg_me = io_args.add_mutually_exclusive_group(required=True)
    reg_me.add_argument('-g', '--genome', metavar='<str>',
        help='Use predefined windows for the human genome, possible values are:\n'
            'GRCh37, GRCh38, CHM13; with an optional suffix `-fast`.\n'
            'Mutually exclusive with --bed-regions.')
    reg_me.add_argument('-b', '--bed-regions', metavar='<file>',
        help='Input bed[.gz] file containing windows (tab-separated, 0-based semi-exclusive),\n'
            'which will be used to calculate read depth. Windows must be non-overlapping\n'
            'and have the same size. Mutually exclusive with --genome.\n\n')

    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<dir>', required=True,
        help='Output directory.')

    filt_args = parser.add_argument_group('Filtering arguments')
    filt_args.add_argument('-m', '--low-mapq-perc', type=float, metavar='<float>', default=10,
        help='Skip windows that have more than <float>%% reads with MAPQ < 10 [default: %(default)s].')
    filt_args.add_argument('-c', '--clipped-perc', type=float, metavar='<float>', default=10,
        help='Skip windows that have more than <float>%% reads with soft/hard clipping [default: %(default)s].')
    filt_args.add_argument('-u', '--unpaired-perc', type=float, metavar='<float>', default=10,
        help='Skip windows that have more than <float>%% reads without proper pair [default: %(default)s].')
    filt_args.add_argument('-N', '--neighbours', type=int, metavar='<int>', default=1,
        help='Discard <int> neighbouring windows to the left and to the right\n'
            'of a skipped window [default: %(default)s].')

    depth_args = parser.add_argument_group('Depth calculation arguments')
    depth_args.add_argument('--long', action='store_true',
        help='Prepare read depth for long reads. Implies --no-gc.')
    depth_args.add_argument('--no-gc', action='store_false', dest='stratify_gc',
        help='Do not stratify by GC-content. Useful for long reads.')
    depth_args.add_argument('--loess-frac', metavar='<float>', type=float, default=0.2,
        help='Loess parameter: use <float> closest windows to estimate average read depth\n'
            'for each GC-percentage [default: %(default)s].')
    depth_args.add_argument('--tail-windows', metavar='<int>', type=int, default=1000,
        help='Do not use GC-content if it lies in the left or right tail\n'
            'with less than <int> windows [default: %(default)s].')
    depth_args.add_argument('--low-mapq', metavar='<int>', type=int, default=_DEFAULT_LOW_MAPQ,
        help='Read mapping quality under <int> is considered as low [default: %(default)s].')
    depth_args.add_argument('--mate-dist', metavar='<int>', type=int, default=pool_reads.DEFAULT_MATE_DISTANCE,
        help='Insert size (~ distance between read mates) is expected to be under <int> [default: %(default)s].')
    depth_args.add_argument('--ploidy', metavar='<int>', type=int, default=2,
        help='Genome ploidy. [default: %(default)s].\n'
             'If not 2, run "parascopy cn[-using]" with "--modify-ref" parameter.')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('-@', '--threads', metavar='<int>', type=int, default=4,
        help='Number of threads [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version',
        version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)
    common.mkdir(args.output)
    genome = Genome(args.fasta_ref)

    bam_wrappers, _samples = pool_reads.load_bam_files(args.input, args.input_list, genome)
    check_duplicated_samples(bam_wrappers)

    threads = max(1, min(len(bam_wrappers), args.threads, os.cpu_count()))
    if threads > 1:
        common.log('Using {} threads'.format(threads))
        import multiprocessing
    else:
        common.log('Using 1 thread')

    if args.bed_regions:
        common.log('Loading input windows from {}'.format(args.bed_regions))
        bed_lines = common.open_possible_gzip(args.bed_regions).readlines()
    else:
        common.log('Loading predefined windows for {}'.format(args.genome))
        bed_lines = _load_background_regions(args.genome)

    with gzip.open(os.path.join(args.output, 'windows.bed.gz'), 'wt') as out:
        out.write('#chrom\tstart\tend\tgc_content\n')
        windows = _load_windows(bed_lines, genome, out)
    bounds = _summarize_windows(windows, args.tail_windows, args.stratify_gc and not args.long)
    if not args.stratify_gc or args.long:
        bounds = None

    _write_readme(args.output)
    with open(os.path.join(args.output, 'depth.csv'), 'w') as out_means:
        common.log('Start calculating coverage')
        window_size = len(windows[0])
        params = Params.from_args(args, window_size, bounds)
        common.log(params.describe() + '    ============')
        _write_means_header(out_means, windows, params, args.tail_windows)

        fetch_regions = _get_fetch_regions(windows, genome, window_size, params.long)
        genome.close()
        common.log('{} continuous regions'.format(len(fetch_regions)))

        prefix = os.path.join(args.output, 'tmp')
        all_files_depth(bam_wrappers, windows, fetch_regions, args.fasta_ref, threads, params, prefix, out_means)

    with gzip.open(os.path.join(args.output, 'window_depth.csv.gz'), 'wb') as out_depth:
        common.log('Merging output files')
        _concat_files(bam_wrappers, prefix, out_depth)

    common.log('Success')


if __name__ == '__main__':
    main()
