import argparse
import sys
import pysam
import operator
import itertools
import os
import csv
import collections
import numpy as np
import traceback
import multiprocessing
import gzip
from Bio import bgzf
from time import perf_counter
from datetime import timedelta

from . import psvs as psvs_
from .view import parse_expression
from . import depth as depth_
from . import pool_reads

from .inner import common
from .inner import duplication as duplication_
from .inner.genome import Genome, Interval, NamedInterval
from .inner.alignment import Weights
from .inner.cigar import Cigar
from .inner import variants as variants_
from .inner import bam_file as bam_file_
from .inner import cn_tools
from .inner import cn_hmm
from .inner import paralog_cn
from .inner import model_params as model_params_
from . import __pkg_name__, __version__, long_version


def _write_bed_files(interval, duplications, const_regions, genome, directory):
    directory = os.path.join(directory, 'bed')
    common.mkdir(directory)

    diploid_regions = [interval]
    diploid_regions.extend(map(operator.attrgetter('region2'), duplications))
    diploid_regions = list(Interval.combine_overlapping(diploid_regions))

    with open(os.path.join(directory, 'diploid.bed'), 'w') as outp:
        for region in diploid_regions:
            outp.write(region.to_bed(genome))
            outp.write('\n')

    by_cn_regions = collections.defaultdict(list)
    with open(os.path.join(directory, 'all.bed'), 'w') as outp:
        outp.write('#chrom\tstart\tend\tcopy_num\tuse\n')
        for const_region in const_regions:
            outp.write('{}\t{}\t{}\n'.format(const_region.region1.to_bed(genome),
                '*' if const_region.cn is None else const_region.cn,
                'skip' if const_region.skip else 'use'))
            if not const_region.skip:
                by_cn_regions[const_region.cn].append(const_region.region1)

    for cn in sorted(by_cn_regions):
        with open(os.path.join(directory, 'copy_num_{}.bed'.format(cn)), 'w') as outp:
            for region in by_cn_regions[cn]:
                outp.write(region.to_bed(genome))
                outp.write('\n')


def _update_psv_observations(record, sample_id, psvs, psv_searcher, psv_observations):
    record_start = record.reference_start
    record_end = record.reference_end
    record_seq = record.query_sequence

    start_ix, end_ix = psv_searcher.overlap_ixs(record_start, record_end)
    if start_ix == end_ix:
        return
    psv = psvs[start_ix]
    psv_end = psv.start + len(psv.ref)
    if record_start <= psv.start:
        psv_observation = ''
        extending_psv = False
    else:
        psv_observation = '-'
        extending_psv = True

    aligned_pairs = record.get_aligned_pairs()
    pairs_i = next(i for i, (_, ref_pos) in enumerate(aligned_pairs) if ref_pos is not None)
    pairs_j = next(i for i, (_, ref_pos) in enumerate(reversed(aligned_pairs)) if ref_pos is not None)
    aligned_pairs = aligned_pairs[pairs_i : len(aligned_pairs) - pairs_j]

    for read_pos, ref_pos in aligned_pairs:
        if read_pos is None:
            continue
        if not ref_pos:
            if extending_psv:
                psv_observation += record_seq[read_pos]
            continue

        while ref_pos >= psv_end:
            psv_observations[start_ix][sample_id][psv_observation] += 1
            start_ix += 1
            if start_ix == end_ix:
                return

            psv = psvs[start_ix]
            psv_end = psv.start + len(psv.ref)
            psv_observation = ''
            extending_psv = False

        if ref_pos >= psv.start:
            psv_observation += record_seq[read_pos]
            extending_psv = True

    is_insertion = any(len(alt) > len(psv.ref) for alt in psv.alts)
    if start_ix < end_ix:
        if psv_end > record_end or (psv_end == record_end and is_insertion):
            psv_observation += '-'
        psv_observations[start_ix][sample_id][psv_observation] += 1


PsvCounts = collections.namedtuple('PsvCounts', 'allele_counts partial other skip')


def _count_psv_observations(psvs, samples, psv_observations):
    """
    Returns matrix of PsvCounts (n_psvs x n_samples).
    """
    n_psvs = len(psvs)
    n_samples = len(samples)
    res = [[None] * n_samples for _ in range(n_psvs)]

    for psv_ix, psv in enumerate(psvs):
        pos2 = psv.info['pos2']

        for sample_id, sample in enumerate(samples):
            obs = psv_observations[psv_ix][sample_id]
            ref_cov = obs[psv.ref]
            alt_cov = tuple(obs[alt] for alt in psv.alts)
            partial = sum(value for key, value in obs.items() if '-' in key)
            total_reads = sum(obs.values())
            oth_cov = total_reads - ref_cov - sum(alt_cov) - partial
            skip = partial >= 0.1 * total_reads or oth_cov >= 0.1 * total_reads

            allele_counts = (ref_cov, *alt_cov)
            res[psv_ix][sample_id] = PsvCounts(allele_counts, partial, oth_cov, skip)
            format = psv.samples[sample_id]
            format['DP'] = total_reads
            format['AD'] = allele_counts
    return res


def _filter_windows(window_counts, bg_depth, dupl_hierarchy, min_windows, perc_samples):
    """
    Set in_viterbi attribute for windows.
    """
    if not window_counts:
        return
    window_size = bg_depth.window_size
    neighbours = bg_depth.neighbours
    neighbours_dist = bg_depth.neighbours_dist

    n_samples = len(window_counts[0])
    max_failed_count = n_samples * 0.01 * perc_samples
    windows = dupl_hierarchy.windows
    n_windows = len(windows)

    for window, window_counts_row in zip(windows, window_counts):
        if not bg_depth.gc_content_defined(window.gc_content):
            window.in_viterbi = False
            continue

        failed_count = n_samples - sum(map(bg_depth.window_passes, window_counts_row))
        if failed_count <= max_failed_count:
            continue

        window.in_viterbi = False
        window_ix = window.ix
        for oth_ix in range(max(0, window_ix - neighbours), min(n_windows, window_ix + neighbours + 1)):
            if oth_ix != window_ix and window.region1.distance(windows[oth_ix].region1) < neighbours_dist:
                windows[oth_ix].in_viterbi = False

    for region_group in dupl_hierarchy.region_groups:
        group_windows = [windows[i] for i in region_group.window_ixs]
        n_windows = sum(map(operator.attrgetter('in_viterbi'), group_windows))
        if n_windows < min_windows:
            for window in group_windows:
                window.in_viterbi = False


def _calculate_pooled_depth(bam_file, samples, bg_depth, read_groups_dict, dupl_hierarchy, outp):
    """
    Returns:
        - window_counts: matrix of WindowCounts (n_windows x n_samples),
        - psv_observations: matrix of Counters (n_psvs x n_samples).

    Calculate both values at the same time to iterate over BAM file only once.
    """
    windows = dupl_hierarchy.windows
    n_windows = len(windows)
    n_samples = len(samples)

    psvs = dupl_hierarchy.psvs
    psv_searcher = dupl_hierarchy.psv_searcher
    # Matrix of counters, rows: PSVs, columns: samples.
    psv_observations = [[collections.Counter() for _j in range(n_samples)] for _i in range(len(psvs))]

    window_size = bg_depth.window_size
    window_starts = np.array([window.region1.start for window in windows])
    # Matrix of WindowCounts (n_samples x n_windows).
    window_counts = [[depth_.WindowCounts() for _j in range(n_samples)] for _i in range(n_windows)]

    read_centers = [[] for _ in range(n_samples)]
    for record in bam_file:
        if record.is_unmapped:
            continue
        sample_id = read_groups_dict[record.get_tag('RG')]
        _update_psv_observations(record, sample_id, psvs, psv_searcher, psv_observations)

        cigar = Cigar.from_pysam_tuples(record.cigartuples)
        middle = depth_.get_read_middle(record, cigar)
        if middle is None:
            continue
        window_ix = window_starts.searchsorted(middle, side='right') - 1
        if window_ix == -1 or middle >= window_starts[window_ix] + window_size:
            continue
        window_counts[window_ix][sample_id].add_read(record, cigar, trust_proper_pair=True, look_at_oa=True)

    outp.write('window_ix\tsample\tdepth1\tdepth2\tlow_mapq\tclipped\tunpaired\tnorm_cn1\n')
    for window, window_counts_row in zip(windows, window_counts):
        gc_content = window.gc_content
        prefix = '{}\t'.format(window.ix)
        for sample_id, sample in enumerate(samples):
            outp.write('{}{}\t'.format(prefix, sample))
            counts = window_counts_row[sample_id]
            counts.write_to(outp)

            n, p = bg_depth.at(sample_id, gc_content)
            mean_bg_depth = n * (1 - p) / p
            outp.write('\t{:.4f}\n'.format(2 * counts.depth_read1 / mean_bg_depth))
    return window_counts, psv_observations


def _create_depth_matrix(windows, window_counts):
    if not windows:
        return None
    n_windows = len(windows)
    n_samples = len(window_counts[0])
    depth_matrix = np.full((n_windows, n_samples), np.iinfo(np.int16).min, dtype=np.int16)

    for window_ix, window in enumerate(windows):
        if window.in_viterbi:
            for sample_id, counts in enumerate(window_counts[window_ix]):
                depth_matrix[window_ix, sample_id] = counts.depth_read1
    return depth_matrix


def _write_windows(dupl_hierarchy, genome, outp):
    outp.write('#chrom\tstart\tend\tcopy_num\tgc_content\twindow_ix\tregion_ix\tregion_group\tin_viterbi\tregions2\n')
    for window in dupl_hierarchy.windows:
        outp.write(window.region1.to_bed(genome))
        outp.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(window.cn, window.gc_content, window.ix,
            window.const_region_ix, dupl_hierarchy.window_group_name(window), 'T' if window.in_viterbi else 'F',
            window.regions2_str(genome, sep=' ')))


def _write_summary(results, region_name, genome, samples, summary_out):
    summary_out.write('## {}\n'.format(' '.join(sys.argv)))
    summary_out.write('## {} {}\n'.format(__pkg_name__, __version__))
    summary_out.write('#chrom\tstart\tend\tlocus\tsample\tagCN_filter\tagCN\tagCN_qual\t'
        'psCN_filter\tpsCN\tpsCN_qual\tinfo\thomologous_regions\n')
    for res_entry in results:
        summary_out.write(res_entry.to_str(region_name, genome, samples))
        summary_out.write('\n')


def transform_duplication(dupl, interval, genome):
    dupl.set_cigar_from_info()
    dupl = dupl.sub_duplication(interval)
    dupl.set_sequences(genome=genome)
    dupl.set_padding_sequences(genome, 200)
    return dupl


class _TimeLogger:
    def __init__(self, filename):
        self._out = open(filename, 'w')
        self._start = perf_counter()

    def log(self, msg):
        curr = perf_counter()
        elapsed = str(timedelta(seconds=curr - self._start))[:-3]
        self._out.write('{}  {}\n'.format(elapsed, msg))
        self._out.flush()

    def close(self):
        self._out.close()


def _update_vcf_header(vcf_header, samples):
    vcf_header.add_line('##INFO=<ID=fval,Number=.,Type=Float,Description="f-values: '
        'frequency of the reference allele for each repeat copy">')
    vcf_header.add_line('##INFO=<ID=info,Number=1,Type=Float,Description="Information content">')
    vcf_header.add_line('##INFO=<ID=rel,Number=1,Type=String,'
        'Description="PSV reliability: r (reliable) | s (semi-reliable) | u (unreliable)">')
    vcf_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    vcf_header.add_line('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">')
    vcf_header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Read depth for each allele">')
    vcf_header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,'
        'Description="Total read count (includes reads that do not support any allele)">')
    vcf_header.add_line('##FORMAT=<ID=psCN,Number=.,Type=String,'
        'Description="Paralog-specific copy number calculated using this PSV only">')
    vcf_header.add_line('##FORMAT=<ID=psCNq,Number=.,Type=Integer,'
        'Description="Phred-quality of the psCN value">')
    for sample in samples:
        vcf_header.add_sample(sample)


def analyze_region(interval, data, samples, bg_depth, model_params):
    subdir = os.path.join(data.args.output, interval.subdir)
    duplications = []
    skip_regions = []
    genome = data.genome
    args = data.args
    exclude_dupl = data.exclude_dupl
    if model_params is None:
        model_params = model_params_.ModelParams(interval, len(samples), is_loaded=False)
        model_params.save_args(args)
    else:
        args = model_params.load_args(args)

    extra_subdir = os.path.join(subdir, 'extra')
    common.mkdir_clear(extra_subdir, rewrite=True)
    time_log = _TimeLogger(os.path.join(extra_subdir, 'time.log'))
    time_log.log('Analyzing {}'.format(interval.full_name(genome)))

    if model_params.is_loaded:
        duplications = model_params.get_duplications(data.table, interval, genome)
        duplications = [transform_duplication(dupl, interval, genome) for dupl in duplications]
        skip_regions = model_params.get_skip_regions(skip_regions, genome)
    else:
        with open(os.path.join(extra_subdir, 'regions.txt'), 'w') as outp:
            for tup in data.table.fetch(interval.chrom_name(genome), interval.start, interval.end):
                dupl = duplication_.Duplication.from_tuple(tup, genome)
                if dupl.is_tangled_region:
                    outp.write('Skip tangled region  %s\n' % dupl.region1.to_str(genome))
                    skip_regions.append(dupl.region1)
                    continue
                if exclude_dupl(dupl, genome):
                    outp.write('Skip duplication  %s\n' % dupl.to_str(genome))
                    continue
                if int(dupl.info['ALENGTH']) < args.short:
                    outp.write('Skip duplication  %s\n' % dupl.to_str(genome))
                    outp.write('    Do not analyze region %s\n' % dupl.region1.to_str(genome))
                    skip_regions.append(dupl.region1)
                    continue
                if dupl.region1.out_of_bounds(genome) or dupl.region2.out_of_bounds(genome):
                    common.log('WARN: [{}] Duplication {} {} is out of bounds, skipping it.'.format(
                        interval.name, dupl.region1.to_str(genome), dupl.region2.to_str(genome)))
                    continue

                outp.write('Use duplication   %s\n' % dupl.to_str(genome))
                model_params.add_duplication(len(duplications), dupl)
                duplications.append(transform_duplication(dupl, interval, genome))
        skip_regions = Interval.combine_overlapping(skip_regions)
        model_params.set_skip_regions(skip_regions)

    psv_header = psvs_.create_vcf_header(genome, (interval.chrom_id,))
    _update_vcf_header(psv_header, samples)
    psv_records = psvs_.create_psv_records(duplications, genome, psv_header, interval, skip_regions)

    window_size = bg_depth.window_size
    const_regions = cn_tools.find_const_regions(duplications, interval, skip_regions, genome,
        min_size=window_size, max_ref_cn=args.max_ref_cn)
    dupl_hierarchy = cn_tools.DuplHierarchy(interval, psv_records, const_regions, genome, duplications,
        window_size=bg_depth.window_size, max_ref_cn=args.max_ref_cn)
    if model_params.is_loaded:
        msg = model_params.check_dupl_hierarchy(dupl_hierarchy, genome)
        if msg:
            common.log(model_params.mismatch_warning())
            common.log(msg)
            raise RuntimeError('Model parameters mismatch')

    _write_bed_files(interval, duplications, const_regions, genome, subdir)
    pooled_bam_path = os.path.join(subdir, 'pooled_reads.bam')
    if not os.path.exists(pooled_bam_path):
        pool_reads.pool_reads_inner(data.bam_wrappers, pooled_bam_path, interval,
            duplications, genome, Weights(), samtools=args.samtools, verbose=True, time_log=time_log)

    extra_files = dict(depth='depth.csv', region_groups='region_groups.txt', windows='windows.bed',
        hmm_states='hmm_states.csv', hmm_params='hmm_params.csv',
        viterbi_summary='viterbi_summary.txt', detailed_cn='detailed_copy_num.bed',
        paralog_cn='paralog_copy_num.csv', gene_conversion='gene_conversion.bed')
    if not model_params.is_loaded:
        extra_files.update(dict(
            psv_f_values='em_f_values.csv', interm_psv_f_values='em_interm_f_values.csv',
            em_likelihoods='em_likelihoods.csv', em_sample_psv_gts='em_sample_psv_gts.csv',
            em_sample_gts='em_sample_gts.csv',
            use_psv_sample='em_use_psv_sample.csv', psv_filtering='em_psv_filtering.txt',
        ))

    results = []
    with cn_tools.OutputFiles(extra_subdir, extra_files) as out:
        cn_hmm.write_headers(out, samples)
        paralog_cn.write_headers(out, samples, args)

        read_groups_dict = {}
        for bam_wrapper in data.bam_wrappers:
            for read_group, sample in bam_wrapper.read_groups().values():
                read_groups_dict[read_group] = samples.id(sample)
        with pysam.AlignmentFile(pooled_bam_path, require_index=True) as bam_file:
            time_log.log('Calculating aggregate read depth')
            common.log('[{}] Calculating aggregate read depth'.format(interval.name))
            window_counts, psv_observations = _calculate_pooled_depth(bam_file, samples, bg_depth, read_groups_dict,
                dupl_hierarchy, out.depth)
        if not model_params.is_loaded:
            _filter_windows(window_counts, bg_depth, dupl_hierarchy, min_windows=args.min_windows, perc_samples=10.0)
            model_params.set_dupl_hierarchy(dupl_hierarchy)

        time_log.log('Writing read depth and PSV observations')
        depth_matrix = _create_depth_matrix(dupl_hierarchy.windows, window_counts)
        psv_counts = _count_psv_observations(psv_records, samples, psv_observations)
        dupl_hierarchy.summarize_region_groups(genome, out.region_groups, args.min_windows)
        _write_windows(dupl_hierarchy, genome, out.windows)
        out.flush()

        for region_group in dupl_hierarchy.region_groups:
            if not region_group.window_ixs:
                continue
            group_extra = cn_tools.RegionGroupExtra(dupl_hierarchy, region_group, psv_counts, len(samples), genome)
            if len(group_extra.hmm_windows) < args.min_windows:
                continue

            group_name = region_group.name
            time_log.log('Group {}: Run HMM to find aggregate copy number profiles'.format(group_name))
            common.log('[{}] Region group {}'.format(interval.name, group_name))
            cn_hmm.find_cn_profiles(group_extra, depth_matrix, samples, bg_depth, genome, out, model_params,
                min_samples=args.min_samples, agcn_range=args.agcn_range, agcn_jump=args.agcn_jump,
                min_trans_prob=args.transition_prob * np.log(10), use_multipliers=args.use_multipliers)
            out.flush()

            time_log.log('Group {}: PSV genotype probabilities'.format(group_name))
            variants_.calculate_all_psv_gt_probs(group_extra,
                max_agcn=args.pscn_bound[0], max_genotypes=args.pscn_bound[1])
            if model_params.is_loaded:
                group_extra.set_from_model_params(model_params, len(samples))
            else:
                time_log.log('Group {}: Run EM to find reliable PSVs'.format(group_name))
                paralog_cn.find_reliable_psvs(group_extra, samples, genome, out, args.min_samples,
                    args.reliable_threshold[1], args.pscn_bound[0])
                model_params.set_psv_f_values(group_extra, genome)
            group_extra.set_reliable_psvs(*args.reliable_threshold)

            time_log.log('Group {}: sample-PSV genotype probabilities'.format(group_name))
            variants_.calculate_support_matrix(group_extra)
            group_extra.update_psv_records(args.reliable_threshold)
            time_log.log('Group {}: paralog-specific copy number and gene conversion'.format(group_name))
            results.extend(paralog_cn.estimate_paralog_cn(group_extra, samples, genome, out,
                max_agcn=args.pscn_bound[0], max_genotypes=args.pscn_bound[1]))
            out.flush()

    time_log.log('Writing results')
    common.log('[{}] Writing PSVs'.format(interval.name))
    psvs_.write_psvs(psv_records, psv_header, os.path.join(subdir, 'psvs.vcf.gz'), args.tabix)

    results.sort()
    with bgzf.open(os.path.join(subdir, 'res.samples.bed.gz'), 'wt') as out1:
        _write_summary(results, interval.name, genome, samples, out1)
    with bgzf.open(os.path.join(subdir, 'res.matrix.bed.gz'), 'wt') as out2:
        paralog_cn.write_matrix_summary(results, interval.name, genome, samples, out2)
    time_log.log('Success')
    time_log.close()

    common.log('[{}] Success'.format(interval.name))
    # Touch empty file.
    os.mknod(os.path.join(extra_subdir, 'success'))
    return model_params


def single_region(region_ix, region, data, samples, bg_depth, model_params):
    """
    Returns tuple (region_ix, exc).
    This is needed because multithreading callback needs to know, how each region finished.

    if exc is None, region finished successfully.
    """
    if region.subdir is None:
        common.log('Skipping region %s' % region.to_str(data.genome))
        return region_ix, 'Skip'
    data.prepare()

    common.mkdir_clear(os.path.join(data.args.output, region.subdir), data.args.rerun == 'full')
    success_path = os.path.join(data.args.output, region.subdir, 'extra', 'success')
    if os.path.exists(success_path):
        if data.args.rerun == 'none':
            common.log('Skipping region %s' % region.full_name(data.genome))
            return region_ix, None
        os.remove(success_path)

    common.log('Analyzing region %s' % region.full_name(data.genome))
    try:
        model_params = analyze_region(region, data, samples, bg_depth, model_params)
        filename = os.path.join(data.args.output, 'model', '{}.gz'.format(region.subdir))
        with gzip.open(filename, 'wt') as model_out:
            model_params.write_to(model_out, data.genome)
    except Exception as exc:
        trace = traceback.format_exc().strip().split('\n')
        trace = '\n'.join(' ' * 11 + s for s in trace)
        common.log('Error in analyzing region %s:\n%s' % (region.full_name(data.genome), trace))
        exc_str = '%s: %s' % (type(exc).__name__, exc)
        return region_ix, exc_str
    finally:
        data.close()
    return region_ix, None


def _join_summaries(out_dir, regions, successful, genome, filename, tabix):
    """
    Join vcf or bed files (vcf file when is_bed_file is False).
    """
    if filename.endswith('.bed') or filename.endswith('.bed.gz'):
        is_bed_file = True
    else:
        assert filename.endswith('.vcf') or filename.endswith('.vcf.gz')
        is_bed_file = False

    entries = []
    write_header = True
    header = []

    maxsplit = 3 + is_bed_file
    for region, success in zip(regions, successful):
        if not success:
            continue
        with common.open_possible_gzip(os.path.join(out_dir, region.subdir, filename)) as inp:
            for line in inp:
                if line.startswith('#'):
                    if write_header:
                        header.append(line)
                    continue
                line = line.split('\t', maxsplit=maxsplit)
                line[0] = genome.chrom_id(line[0])
                line[1] = int(line[1])
                if is_bed_file:
                    line[2] = int(line[2])
                entries.append(line)
            write_header = not header

    entries.sort()
    out_filename = os.path.join(out_dir, filename)
    with common.open_possible_gzip(out_filename, 'w', bgzip=True) as out:
        for line in header:
            out.write(line)
        for entry in entries:
            entry[0] = genome.chrom_name(entry[0])
            entry[1] = str(entry[1])
            if is_bed_file:
                entry[2] = str(entry[2])
            out.write('\t'.join(entry))
    if out_filename.endswith('.gz') and tabix != 'none':
        common.Process([tabix, '-p', 'bed' if is_bed_file else 'vcf', out_filename]).finish()


def set_regions_subdirs(regions):
    used_names = set()
    for region in regions:
        if region.os_name not in used_names:
            name = region.os_name
        else:
            name = region.to_str(data.genome)
            if name in used_names:
                common.log('WARN: region {} appears twice, skipping.'.format(name))
                region.subdir = None
                continue
        used_names.add(name)
        region.subdir = name


def run(regions, data, samples, bg_depth, models):
    if models is None:
        models = [None] * len(regions)

    out_dir = data.args.output
    set_regions_subdirs(regions)
    threads = max(1, min(len(regions), data.args.threads))
    results = []
    if threads == 1:
        for region_ix, (region, model_params) in enumerate(zip(regions, models)):
            results.append(single_region(region_ix, region, data, samples, bg_depth, model_params))

    else:
        def callback(res):
            results.append(res)

        def err_callback(exc):
            common.log('Thread finished with an exception:\n{}'.format(exc))
            os._exit(1)

        common.log('Using %d threads' % threads)
        pool = multiprocessing.Pool(threads)
        for region_ix, (region, model_params) in enumerate(zip(regions, models)):
            pool.apply_async(single_region,
                args=(region_ix, region, data.copy(), samples, bg_depth, model_params),
                callback=callback, error_callback=err_callback)
        pool.close()
        pool.join()

    n_regions = len(regions)
    results.sort()
    assert len(results) == n_regions and all(i == region_ix for i, (region_ix, _) in enumerate(results))
    successful = [exc is None for _, exc in results]

    with open(os.path.join(out_dir, 'model', 'list.txt'), 'w') as model_list:
        for region in itertools.compress(regions, successful):
            model_list.write('{}.gz\n'.format(region.subdir))
    _join_summaries(out_dir, regions, successful, data.genome, 'res.samples.bed.gz', data.args.tabix)
    _join_summaries(out_dir, regions, successful, data.genome, 'res.matrix.bed.gz', data.args.tabix)
    _join_summaries(out_dir, regions, successful, data.genome, 'psvs.vcf.gz', data.args.tabix)

    n_successes = sum(successful)
    if n_successes < n_regions:
        common.log('==============================================')
        common.log('ERROR: Could not finish the following regions:')
        i = 0
        for region, (_, exc) in zip(regions, results):
            if exc is not None:
                common.log('    %s: %s' % (region.full_name(data.genome), exc))
                i += 1
                if i >= 10:
                    break
        if i < n_regions - n_successes:
            common.log('    ...')

    if n_successes == 0:
        common.log('Failure! No regions were analyzed successfully.')
    else:
        common.log('Success [%d regions out of %d]' % (n_successes, n_regions))


class _DataStructures:
    def __init__(self, args):
        self._args = args
        self._loaded_times = 0

        self._exclude_dupl = None
        self._genome = None
        self._table = None
        self._bam_wrappers = None

    @property
    def args(self):
        return self._args

    @property
    def genome(self):
        return self._genome

    @property
    def table(self):
        return self._table

    @property
    def bam_wrappers(self):
        return self._bam_wrappers

    @property
    def exclude_dupl(self):
        return self._exclude_dupl

    def prepare(self):
        self._loaded_times += 1
        if self._loaded_times > 1:
            return

        if self._args.is_new:
            self._exclude_dupl = parse_expression(self._args.exclude)
        else:
            self._exclude_dupl = None
        self._genome = Genome(self._args.fasta_ref)
        self._table = pysam.TabixFile(self._args.table, parser=pysam.asTuple())

    def close(self):
        self._loaded_times -= 1
        if self._loaded_times == 0:
            self._genome.close()
            self._table.close()

    def set_bam_wrappers(self, bam_wrappers):
        self._bam_wrappers = bam_wrappers

    def copy(self):
        res = _DataStructures(self._args)
        res.set_bam_wrappers(self._bam_wrappers)
        return res


def _write_command(filename):
    """
    Rewrite file with commands, leaving only the last of each entries. Add new command to the end.
    """
    lines = []
    if os.path.exists(filename):
        with open(filename) as inp:
            lines.extend(inp.readlines())

    lines.append(' '.join(sys.argv) + '\n')
    lines_cnt = collections.Counter(lines)
    with open(filename, 'w') as outp:
        for line in lines:
            if lines_cnt[line] > 1:
                lines_cnt[line] -= 1
            else:
                outp.write(line)


def get_regions(args, genome, load_models):
    if not load_models:
        return (common.get_regions(args, genome), None)

    loaded_models = model_params_.load_all(args.model, genome)
    regions = []
    for model_params in loaded_models:
        main_entry = model_params.main_entry
        region = NamedInterval.from_region(main_entry.region1, main_entry.info['name'])
        regions.append(region)
    return (regions, loaded_models)


def filter_regions(regions, loaded_models, regions_subset):
    if regions_subset:
        exclude = regions_subset[0] == '!'
        regions_subset = set(regions_subset)
        if exclude:
            regions_subset.remove('!')
            ixs = [i for i, r in enumerate(regions) if r.name not in regions_subset]
        else:
            ixs = [i for i, r in enumerate(regions) if r.name in regions_subset]

        regions = [regions[i] for i in ixs]
        if loaded_models:
            loaded_models = [loaded_models[i] for i in ixs]
    if not regions:
        common.log('Failure! No regions provided.')
        exit(1)
    return regions, loaded_models


class SingleMetavar(argparse.RawTextHelpFormatter):
    def _format_args(self, action, default_metavar):
        return self._metavar_formatter(action, default_metavar)(1)[0]


def parse_args(prog_name, in_args, is_new):
    assert prog_name is not None
    usage = ('{prog} {model}(-i <bam> [...] | -I <bam-list>) -t <table> -f <fasta> '
        '{regions}-d <bg-depth> -o <dir> [args]').format(prog=prog_name, model='' if is_new else '<model> ',
        regions='(-r <region> [...] | -R <bed>) ' if is_new else '')

    DEFAULT_PSCN_BOUND = (8, 500)
    parser = argparse.ArgumentParser(
        description='Find aggregate and paralog-specific copy number for given unique and duplicated regions.',
        formatter_class=SingleMetavar, add_help=False, usage=usage)
    io_args = parser.add_argument_group('Input/output arguments')
    if not is_new:
        io_args.add_argument('model', metavar='<model>', nargs='+',
            help='Use model parameters from an independent "parascopy cn" run.\n'
                'Allows multiple arguments of the following types:\n'
                '- model/<region>.gz,\n'
                '- file with paths to *.gz files,\n'
                '- directory: use all subfiles *.gz.\n\n')

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

    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed table with information about segmental duplications.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-d', '--depth', metavar='<file>', required=True,
        help='Input file / directory with background read depth.\n'
        'Should be created using "parascopy depth".')
    io_args.add_argument('-o', '--output', metavar='<dir>', required=True,
        help='Output directory.')

    reg_args = parser.add_argument_group('Region arguments')
    if is_new:
        reg_me = reg_args.add_mutually_exclusive_group(required=True)
        reg_me.add_argument('-r', '--regions', nargs='+', metavar='<region> [<region> ...]',
            help='Region(s) in format "chr" or "chr:start-end".\n'
                'Start and end are 1-based inclusive. Commas are ignored.\n'
                'Mutually exclusive with --regions-file.')
        reg_me.add_argument('-R', '--regions-file', metavar='<file>',
            help='Input bed[.gz] file containing regions (tab-separated, 0-based semi-exclusive).\n'
                'Optional fourth column will be used as a region name.\n'
                'Mutually exclusive with --regions.')
    reg_args.add_argument('--regions-subset', nargs='+', metavar='<str> [<str> ...]',
        help='Additionally filter input regions: only use regions with names that are in this list.\n'
            'If the first argument is "!", only use regions not in this list.')

    if is_new:
        filt_args = parser.add_argument_group('Duplications filtering arguments')
        filt_args.add_argument('-e', '--exclude', metavar='<expr>',
            default='length < 500 && seq_sim < 0.97',
            help='Exclude duplications for which the expression is true\n[default: %(default)s].')
        filt_args.add_argument('--short', type=int, metavar='<int>', default=500,
            help='Skip regions with short duplications (shorter than <int> bp),\n'
                'not excluded in the -e/--exclude argument [default: %(default)s].')
        filt_args.add_argument('--max-ref-cn', type=int, metavar='<int>', default=10,
            help='Skip regions with reference copy number higher than <int> [default: %(default)s].')

    aggr_det_args = parser.add_argument_group('Aggregate copy number detection arguments')
    if is_new:
        aggr_det_args.add_argument('--min-samples', type=int, metavar='<int>', default=50,
            help='Use multi-sample information if there are at least <int> samples present\n'
                'for the region/PSV [default: %(default)s].')
        aggr_det_args.add_argument('--min-windows', type=int, metavar='<int>', default=10,
            help='Predict aggregate and paralog copy number only in regions with at\n'
                'least <int> windows [default: %(default)s].')
        aggr_det_args.add_argument('--agcn-range', type=int, metavar='<int> <int>', nargs=2, default=(5, 7),
            help='Detect aggregate copy number in a range around the reference copy number [default: 5 7].\n'
                'For example, for a duplication with copy number 8 copy numbers 3-15 can\n'
                'be detected. In addition, sample may be identified as having copy number <3 or >15.\n'
                'In some cases, copy number within two ranges can be detected.')
        aggr_det_args.add_argument('--agcn-jump', type=int, metavar='<int>', default=6,
            help='Maximal jump in the aggregate copy number between two consecutive windows [default: %(default)s].')
        aggr_det_args.add_argument('--transition-prob', type=float, metavar='<float>', default=-5,
            help='Log10 transition probability for the aggregate copy number HMM [default: %(default)s].')
    aggr_det_args.add_argument('--no-multipliers', action='store_false', dest='use_multipliers',
        help='Do not estimate or use read depth multipliers.')

    if is_new:
        par_det_args = parser.add_argument_group('Paralog-specific copy number detection arguments')
        par_det_args.add_argument('--reliable-threshold', type=float, metavar='<float> <float>',
            nargs=2, default=(0.8, 0.95),
            help='PSV-reliability thresholds (reliable PSV has all f-values over the threshold).\n'
                'First value is used for gene conversion detection,\n'
                'second value is used to estimate paralog-specific CN [default: 0.80 0.95].')
        par_det_args.add_argument('--pscn-bound', type=int, metavar='<int> [<int>]',
            default=DEFAULT_PSCN_BOUND, nargs='+',
            help=('Do not estimate paralog-specific copy number if any of the statements is true:\n'
                '- aggregate copy number is higher than <int>[1]           [default: {}],\n'
                '- number of possible psCN tuples is higher than <int>[2]  [default: {}].')
                .format(*DEFAULT_PSCN_BOUND))

    exec_args = parser.add_argument_group('Execution parameters')
    exec_args.add_argument('--rerun', choices=('full', 'partial', 'none'), metavar='full|partial|none', default='none',
        help='Rerun CN analysis for all loci:\n'
            '    full:    complete rerun,\n'
            '    partial: use pooled reads from a previous run,\n'
            '    none:    skip successfully finished loci [default].')
    exec_args.add_argument('-@', '--threads', type=int, metavar='<int>', default=4,
        help='Number of available threads [default: %(default)s].')
    exec_args.add_argument('--samtools', metavar='<path>|none', default='samtools',
        help='Path to samtools executable [default: %(default)s].\n'
            'Use python wrapper if "none", can lead to errors.')
    exec_args.add_argument('--tabix', metavar='<path>', default='tabix',
        help='Path to "tabix" executable [default: %(default)s].\n'
            'Use "none" to skip indexing output files.')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_args)

    args.is_new = is_new
    if args.samtools != 'none':
        common.check_executable(args.samtools)
    if args.tabix != 'none':
        common.check_executable(args.tabix)

    if is_new:
        if len(args.pscn_bound) == 1:
            args.pscn_bound.append(DEFAULT_PSCN_BOUND[1])
        elif len(args.pscn_bound) != 2:
            sys.stderr.write('Unexpected number of arguments for --pscn-bound\n\n')
            parser.print_usage(sys.stderr)
            exit(1)
    return args


def main(prog_name=None, in_args=None, is_new=None):
    args = parse_args(prog_name, in_args, is_new)
    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

    data = _DataStructures(args)
    data.prepare()

    directory = args.output
    common.log('Using output directory "{}"'.format(directory))
    common.mkdir(directory)
    _write_command(os.path.join(directory, 'command.txt'))

    regions, loaded_models = get_regions(args, data.genome, load_models=not args.is_new)
    if loaded_models:
        args.min_samples = None
    common.mkdir_clear(os.path.join(directory, 'model'), args.rerun == 'full')
    regions, loaded_models = filter_regions(regions, loaded_models, args.regions_subset)

    bam_wrappers = pool_reads.load_bam_files(args.input, args.input_list, data.genome, allow_unnamed=False)
    data.set_bam_wrappers(bam_wrappers)
    depth_.check_duplicated_samples(bam_wrappers)

    samples = bam_file_.Samples.from_bam_wrappers(bam_wrappers)
    bg_depth = depth_.Depth(args.depth, samples)

    run(regions, data, samples, bg_depth, loaded_models)
    data.close()


if __name__ == '__main__':
    main()
