import argparse
import sys
import pysam
import operator
import itertools
import os
import shutil
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
from .inner.genome import Genome, ChromNames, Interval, NamedInterval
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
    Set in_hmm attribute for windows.
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
            window.in_hmm = False
            continue

        failed_count = n_samples - sum(map(bg_depth.window_passes, window_counts_row))
        if failed_count <= max_failed_count:
            continue

        window.in_hmm = False
        window_ix = window.ix
        for oth_ix in range(max(0, window_ix - neighbours), min(n_windows, window_ix + neighbours + 1)):
            if oth_ix != window_ix and window.region1.distance(windows[oth_ix].region1) < neighbours_dist:
                windows[oth_ix].in_hmm = False

    for region_group in dupl_hierarchy.region_groups:
        group_windows = [windows[i] for i in region_group.window_ixs]
        n_windows = sum(map(operator.attrgetter('in_hmm'), group_windows))
        if n_windows < min_windows:
            for window in group_windows:
                window.in_hmm = False


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
    window_counts = [[depth_.WindowCounts(bg_depth.params) for _j in range(n_samples)] for _i in range(n_windows)]

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
            outp.write('\t{:.4f}\n'.format(counts.depth_read1 / mean_bg_depth))
    return window_counts, psv_observations


def _create_depth_matrix(windows, window_counts):
    if not windows:
        return None
    n_windows = len(windows)
    n_samples = len(window_counts[0])
    depth_matrix = np.full((n_windows, n_samples), np.iinfo(np.int16).min, dtype=np.int16)

    for window_ix, window in enumerate(windows):
        if window.in_hmm:
            for sample_id, counts in enumerate(window_counts[window_ix]):
                depth_matrix[window_ix, sample_id] = counts.depth_read1
    return depth_matrix


def _write_windows(dupl_hierarchy, genome, outp):
    outp.write('#chrom\tstart\tend\tcopy_num\tgc_content\twindow_ix\tregion_ix\tregion_group\tin_hmm\tregions2\n')
    for window in dupl_hierarchy.windows:
        outp.write(window.region1.to_bed(genome))
        outp.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(window.cn, window.gc_content, window.ix,
            window.const_region_ix, dupl_hierarchy.window_group_name(window), 'T' if window.in_hmm else 'F',
            window.regions2_str(genome, sep=' ')))


def _write_summary(results, region_name, genome, samples, summary_out):
    summary_out.write('## {}\n'.format(common.command_to_str()))
    summary_out.write('## {} v{}\n'.format(__pkg_name__, __version__))
    summary_out.write('#chrom\tstart\tend\tlocus\tsample\tagCN_filter\tagCN\tagCN_qual\t'
        'psCN_filter\tpsCN\tpsCN_qual\tinfo\thomologous_regions\n')
    for res_entry in results:
        summary_out.write(res_entry.to_str(region_name, genome, samples))
        summary_out.write('\n')


def clip_duplication(dupl, interval, genome):
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


def _validate_read_groups(pooled_bam, read_groups_dict, samples):
    pooled_read_groups = bam_file_.get_read_groups(pooled_bam)
    if len(pooled_read_groups) != len(read_groups_dict):
        common.log('WARN: Number of read groups does not match: {} in pooled BAM, {} expected.'.format(
            len(pooled_read_groups), len(read_groups_dict)))

    samples_present = np.zeros(len(samples), dtype=np.bool_)
    for rg, sample in pooled_read_groups:
        if rg not in read_groups_dict:
            common.log('ERROR: Unknown pooled read group {} (sample {}). Perhaps the set of input files has changed?'
                .format(rg, sample))
            exit(1)

        exp_sample = samples[read_groups_dict[rg]]
        if sample != exp_sample:
            common.log(('ERROR: Pooled read group {} is associated with two different samples ({} and {}). '
                'Perhaps the set of input files has changed?').format(rg, sample, exp_sample))
            exit(1)
        samples_present[samples.id(sample)] = True

    if np.any(~samples_present):
        ixs = np.where(~samples_present)[0]
        common.log('WARN: {} samples are present in the input files, but not in the pooled BAM file. For example: {}'
            .format(len(ixs), samples[ixs[0]]))


def get_pool_interval(interval, genome, pool_interval=2000):
    pool_interval = interval.add_padding(pool_interval)
    pool_interval.trim(genome)
    return pool_interval


def analyze_region(interval, subdir, data, samples, bg_depth, model_params, force_agcn, modified_ref_cns):
    duplications = []
    pool_duplications = []
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

    pool_interval = get_pool_interval(interval, genome)
    if model_params.is_loaded:
        for dupl in model_params.get_duplications(data.table, interval, genome):
            dupl.set_cigar_from_info()
            duplications.append(clip_duplication(dupl, interval, genome))
            pool_duplications.append(clip_duplication(dupl, pool_interval, genome))
        skip_regions = model_params.get_skip_regions(skip_regions, genome)
    else:
        with open(os.path.join(extra_subdir, 'regions.txt'), 'w') as outp:
            for tup in data.table.fetch(interval.chrom_name(genome), interval.start, interval.end):
                dupl = duplication_.Duplication.from_tuple(tup, genome)
                if dupl.is_tangled_region:
                    outp.write('Skip tangled region  {}\n'.format(dupl.region1.to_str(genome)))
                    skip_regions.append(dupl.region1)
                    continue
                if exclude_dupl(dupl, genome):
                    outp.write('Skip duplication  {}\n'.format(dupl.to_str(genome)))
                    continue
                if int(dupl.info['ALENGTH']) < args.short:
                    outp.write('Skip duplication  {}\n'.format(dupl.to_str(genome)))
                    outp.write('    Do not analyze region {}\n'.format(dupl.region1.to_str(genome)))
                    skip_regions.append(dupl.region1)
                    continue
                if dupl.region1.out_of_bounds(genome) or dupl.region2.out_of_bounds(genome):
                    common.log('WARN: [{}] Duplication {} {} is out of bounds, skipping it.'.format(
                        interval.name, dupl.region1.to_str(genome), dupl.region2.to_str(genome)))
                    continue

                outp.write('Use duplication   {}\n'.format(dupl.to_str(genome)))
                model_params.add_duplication(len(duplications), dupl)
                dupl.set_cigar_from_info()
                duplications.append(clip_duplication(dupl, interval, genome))
                pool_duplications.append(clip_duplication(dupl, pool_interval, genome))
        skip_regions = Interval.combine_overlapping(skip_regions)
        model_params.set_skip_regions(skip_regions)

    if args.skip_unique and not duplications:
        common.log('WARN: Skipping locus {}: no duplications present'.format(interval.name))
        return None

    psv_header = psvs_.create_vcf_header(genome)
    _update_vcf_header(psv_header, samples)
    psv_records = psvs_.create_psv_records(duplications, genome, psv_header, interval, skip_regions)
    psv_records = [record for record in psv_records if record.qual > 0]

    window_size = bg_depth.window_size
    const_regions = cn_tools.find_const_regions(duplications, interval, skip_regions, genome,
        min_size=window_size, max_ref_cn=args.max_ref_cn)
    dupl_hierarchy = cn_tools.DuplHierarchy(interval, psv_records, const_regions, genome, duplications,
        window_size=bg_depth.window_size, max_ref_cn=args.max_ref_cn, max_dist=args.region_dist)
    unknown_frac = dupl_hierarchy.interval_seq.count('N') / len(dupl_hierarchy.interval_seq)
    if unknown_frac > args.unknown_seq:
        common.log('WARN: Skipping locus {}: {:.1f}% of the sequence is unknown'
            .format(interval.name, 100 * unknown_frac))
        return None
    elif unknown_frac > 0:
        common.log('WARN: [{}] {:.1f}% of the sequence is unknown'.format(interval.name, 100 * unknown_frac))

    if model_params.is_loaded:
        msg = model_params.check_dupl_hierarchy(dupl_hierarchy, genome)
        if msg:
            common.log(model_params.mismatch_warning(genome))
            common.log(msg)
            raise RuntimeError('Model parameters mismatch')

    _write_bed_files(interval, duplications, const_regions, genome, subdir)
    pooled_bam_path = os.path.join(subdir, 'pooled_reads.bam')
    if not os.path.exists(pooled_bam_path):
        pool_reads.pool(data.bam_wrappers, pooled_bam_path, interval, pool_duplications, genome,
            samtools=args.samtools, verbose=True, time_log=time_log)

    extra_files = dict(depth='depth.csv', region_groups='region_groups.txt', windows='windows.bed',
        hmm_states='hmm_states.csv', hmm_params='hmm_params.csv',
        viterbi_summary='viterbi_summary.txt', detailed_cn='detailed_copy_num.bed',
        paralog_cn='paralog_copy_num.csv', gene_conversion='gene_conversion.bed')
    if not model_params.is_loaded:
        extra_files.update(dict(
            psv_f_values='em_f_values.csv', interm_psv_f_values='em_interm_f_values.csv',
            em_pscn='em_psCN.csv', use_psv_sample='em_use_psv_sample.csv', psv_filtering='em_psv_filtering.txt',
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
        depth_matrix = _create_depth_matrix(dupl_hierarchy.windows, window_counts) if force_agcn is None else None
        psv_counts = _count_psv_observations(psv_records, samples, psv_observations)
        dupl_hierarchy.summarize_region_groups(genome, out.region_groups, args.min_windows)
        _write_windows(dupl_hierarchy, genome, out.windows)
        out.flush()

        if args.skip_cn:
            assert args.min_windows > 0
            for window in dupl_hierarchy.windows:
                window.in_hmm = False

        for region_group in dupl_hierarchy.region_groups:
            if not region_group.window_ixs:
                continue
            group_extra = cn_tools.RegionGroupExtra(dupl_hierarchy, region_group, psv_counts, len(samples), genome)
            if len(group_extra.hmm_windows) < args.min_windows:
                continue

            group_name = region_group.name
            if force_agcn is None:
                time_log.log('Group {}: Run HMM to find aggregate copy number profiles'.format(group_name))
                common.log('[{}] Region group {}'.format(interval.name, group_name))
                cn_hmm.find_cn_profiles(group_extra, depth_matrix, samples, bg_depth, genome, out, model_params,
                    min_samples=args.min_samples, agcn_range=args.agcn_range, strict_agcn_range=args.strict_agcn_range,
                    agcn_jump=args.agcn_jump, min_trans_prob=args.transition_prob * common.LOG10,
                    uniform_initial=args.uniform_initial, use_multipliers=args.use_multipliers,
                    update_agcn_qual=args.update_agcn)
                out.flush()
            else:
                group_extra.use_forced_agcn(force_agcn, samples, genome)

            time_log.log('Group {}: PSV genotype probabilities'.format(group_name))
            if not args.skip_pscn:
                variants_.calculate_all_psv_gt_probs(group_extra,
                    max_agcn=args.pscn_bound[0], max_genotypes=args.pscn_bound[1])
            if model_params.is_loaded:
                group_extra.set_from_model_params(model_params, len(samples))
            else:
                time_log.log('Group {}: Run EM to find reliable PSVs'.format(group_name))
                paralog_cn.find_reliable_psvs(group_extra, samples, genome, modified_ref_cns, out,
                    min_samples=args.min_samples, reliable_threshold=args.reliable_threshold[1],
                    max_agcn=args.pscn_bound[0])
                model_params.set_psv_f_values(group_extra, genome)
            group_extra.set_reliable_psvs(*args.reliable_threshold)

            time_log.log('Group {}: sample-PSV genotype probabilities'.format(group_name))
            variants_.calculate_support_matrix(group_extra)
            time_log.log('Group {}: paralog-specific copy number and gene conversion'.format(group_name))
            results.extend(paralog_cn.estimate_paralog_cn(group_extra, samples, genome, out, args.pscn_bound[0]))
            group_extra.update_psv_records(args.reliable_threshold)
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


def get_locus_dir(out_dir, locus_name, *, move_old):
    """
    Returns the name of the locus directory.
    If `move_old` is true, moves old directory to a new place.
    """
    locus_dir = os.path.join(out_dir, 'loci', locus_name)
    if move_old:
        old_dir = os.path.join(out_dir, locus_name)
        if os.path.exists(old_dir) and not os.path.exists(locus_dir):
            shutil.move(old_dir, locus_dir)
    return locus_dir


def single_region(region_ix, region, data, samples, bg_depth, model_params, force_agcn, modified_ref_cns):
    """
    Returns tuple (region_ix, exc).
    This is needed because multithreading callback needs to know, how each region finished.

    if exc is None, region finished successfully.
    """
    if region.os_name is None:
        return region_ix, \
            RuntimeError('Cannot create directory for region {}, os_name is None'.format(region.full_name(data.genome)))
    data.prepare()

    locus_dir = get_locus_dir(data.args.output, region.os_name, move_old=True)
    common.mkdir_clear(locus_dir, data.args.rerun == 'full')
    success_path = os.path.join(locus_dir, 'extra', 'success')
    if os.path.exists(success_path):
        if data.args.rerun == 'none':
            common.log('Skipping region {}'.format(region.full_name(data.genome)))
            return region_ix, None
        os.remove(success_path)

    common.log('Analyzing region {}'.format(region.full_name(data.genome)))
    try:
        model_params = analyze_region(region, locus_dir, data, samples, bg_depth,
            model_params, force_agcn, modified_ref_cns)
        if model_params is None:
            return region_ix, 'Skip'

        filename = os.path.join(data.args.output, 'model', '{}.gz'.format(region.os_name))
        with gzip.open(filename, 'wt') as model_out:
            model_params.write_to(model_out, data.genome)
    except Exception as exc:
        trace = traceback.format_exc().strip().split('\n')
        trace = '\n'.join(' ' * 11 + s for s in trace)
        common.log('Error: region {}:\n{}'.format(region.full_name(data.genome), trace))
        exc_str = '{}: {}'.format(type(exc).__name__, exc)
        return region_ix, exc_str
    finally:
        data.close()
    return region_ix, None


def join_bed_files(in_dirs, out_filename, genome, tabix):
    basename = os.path.basename(out_filename)
    entries = []
    header = []
    for in_dir in in_dirs:
        with common.open_possible_gzip(os.path.join(in_dir, basename)) as inp:
            write_header = not header
            for line in inp:
                if line.startswith('#'):
                    if write_header:
                        header.append(line)
                    continue

                line = line.split('\t', maxsplit=3)
                line[0] = genome.chrom_id(line[0])
                line[1] = int(line[1])
                line[2] = int(line[2])
                entries.append(line)

    entries.sort()
    with common.open_possible_gzip(out_filename, 'w', bgzip=True) as out:
        for line in header:
            out.write(line)
        for entry in entries:
            entry[0] = genome.chrom_name(entry[0])
            entry[1] = str(entry[1])
            entry[2] = str(entry[2])
            out.write('\t'.join(entry))
    if out_filename.endswith('.gz') and tabix != 'none':
        common.Process([tabix, '-p', 'bed', out_filename]).finish()


def join_vcf_files(in_entries, out_filename, genome, tabix, merge_duplicates=False, in_filenames=False):
    basename = os.path.basename(out_filename)
    records = []
    header = None
    for in_entry in in_entries:
        # If `in_filenames`, `in_entries` already contains filenames, otherwise it contains directories.
        in_filename = in_entry if in_filenames else os.path.join(in_entry, basename)
        with pysam.VariantFile(in_filename) as vcf:
            if header is None:
                header = vcf.header
            records.extend(vcf)

    records.sort(key=variants_.vcf_record_key(genome))
    if merge_duplicates:
        records = variants_.merge_duplicates(records)
    with common.open_vcf(out_filename, 'w', header=header) as vcf:
        for record in records:
            vcf.write(record)
    if out_filename.endswith('.gz') and tabix != 'none':
        common.Process([tabix, '-p', 'vcf', out_filename]).finish()
    return records


def run(regions, data, samples, bg_depth, models, force_agcn, modified_ref_cns):
    n_regions = len(regions)
    if models is None:
        models = [None] * n_regions

    threads = max(1, min(n_regions, data.args.threads))
    results = []
    if threads == 1:
        common.log('Using 1 thread')
        for region_ix, (region, model_params) in enumerate(zip(regions, models)):
            results.append(single_region(region_ix, region, data, samples, bg_depth,
                model_params, force_agcn, modified_ref_cns))

    else:
        def callback(res):
            results.append(res)

        def err_callback(exc):
            common.log('Thread finished with an exception:\n{}'.format(exc))
            os._exit(1)

        common.log('Using {} threads'.format(threads))
        pool = multiprocessing.Pool(threads)
        for region_ix, (region, model_params) in enumerate(zip(regions, models)):
            pool.apply_async(single_region,
                args=(region_ix, region, data.copy(), samples, bg_depth, model_params, force_agcn, modified_ref_cns),
                callback=callback, error_callback=err_callback)
        pool.close()
        pool.join()

    results.sort()
    assert len(results) == n_regions and all(i == region_ix for i, (region_ix, _) in enumerate(results))
    successful = [exc is None for _, exc in results]

    out_dir = data.args.output
    with open(os.path.join(out_dir, 'model', 'list.txt'), 'w') as model_list:
        for region in itertools.compress(regions, successful):
            model_list.write('{}.gz\n'.format(region.os_name))

    genome = data.genome
    successful_regions = list(itertools.compress(regions, successful))
    if successful_regions:
        common.log('Merging output files')
        tabix = data.args.tabix
        out_filename1 = os.path.join(out_dir, 'res.samples.bed.gz')
        loci_dir = [get_locus_dir(out_dir, region.os_name, move_old=False) for region in successful_regions]
        join_bed_files(loci_dir, out_filename1, genome, tabix)
        paralog_cn.summary_to_paralog_bed(out_filename1,
            os.path.join(out_dir, 'res.paralog.bed.gz'), genome, samples, tabix)
        join_bed_files(loci_dir, os.path.join(out_dir, 'res.matrix.bed.gz'), genome, tabix)
        join_vcf_files(loci_dir, os.path.join(out_dir, 'psvs.vcf.gz'), genome, tabix)

    n_successes = sum(successful)
    if n_successes < n_regions:
        common.log('==============================================')
        common.log('ERROR: Could not finish the following regions:')
        i = 0
        skipped = []
        for region, (_, exc) in zip(regions, results):
            if exc == 'Skip':
                skipped.append(region.full_name(genome))
            elif exc is not None:
                common.log('    {}: {}'.format(region.full_name(genome), exc))
                i += 1
                if i >= 10:
                    break
        n_skipped = len(skipped)
        if i < n_regions - n_successes - n_skipped:
            common.log('    ...')
        if skipped:
            common.log('    {} regions skipped: {}{}'.format(
                n_skipped, ', '.join(skipped[:10]), ', ...' if n_skipped > 10 else ''))

    if n_successes == 0:
        common.log('Failure! No regions were analyzed successfully.')
    else:
        common.log('Success [{} regions out of {}]'.format(n_successes, n_regions))


class DataStructures:
    """
    Store genome, table and other things in an unloaded/loaded state.
    Necessary for multi-threaded applications to efficiently open/close files.
    """

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

        if hasattr(self._args, 'exclude'):
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
        res = DataStructures(self._args)
        res.set_bam_wrappers(self._bam_wrappers)
        return res


def write_command(filename):
    """
    Rewrite file with commands, leaving only the last of each entries. Add new command to the end.
    """
    lines = []
    if os.path.exists(filename):
        with open(filename) as inp:
            lines.extend(inp.readlines())

    lines.append(common.command_to_str() + '\n')
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
    os_names = {}
    regions = []
    for model_params in loaded_models:
        main_entry = model_params.main_entry
        name = main_entry.get('name')
        region = NamedInterval.from_region(main_entry.region1, genome, name)
        if region.os_name is os_names:
            raise ValueError('Provided several models with the same name ({})'.format(region.os_name))
        regions.append(region)
    return (regions, loaded_models)


def filter_regions(regions, loaded_models, regions_subset):
    start_len = len(regions)
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
    if len(regions) < start_len:
        common.log('WARN: Discarded {} loci based on --regions-subset'.format(start_len - len(regions)))
    if not regions:
        common.log('Failure! No remaining regions')
        exit(1)
    return regions, loaded_models


class InCopyNums:
    def __init__(self, filename, genome, samples):
        from .inner import itree
        self._filename = filename

        n_samples = len(samples)
        sample_regions = [[] for _ in range(n_samples)]
        with common.open_possible_gzip(filename) as inp:
            for line in inp:
                if line.startswith('#'):
                    continue
                chrom, start, end, sample_list, cn = line.strip().split()[:5]
                chrom_id = genome.chrom_id(chrom)
                if end == 'inf':
                    end = genome.chrom_len(chrom_id)
                else:
                    end = int(end)
                pair = (Interval(chrom_id, int(start), end), int(cn))

                if sample_list == '*':
                    sample_ids = range(n_samples)
                else:
                    sample_ids = []
                    for sample in sample_list.split(','):
                        if sample in samples:
                            sample_ids.append(samples.id(sample))
                for sample_id in sample_ids:
                    sample_regions[sample_id].append(pair)

        self._trees = []
        region_getter = operator.itemgetter(0)
        for regions in sample_regions:
            if not regions:
                self._trees.append(None)
            else:
                regions.sort()
                self._trees.append(itree.MultiNonOverlTree(regions, region_getter))

    @property
    def filename(self):
        return self._filename

    def from_region(self, sample_id, region):
        tree = self._trees[sample_id]
        if tree is None:
            return ()
        return tuple(tree.overlap_iter(region))

    def from_regions(self, regions, sample_id, sample=None, genome=None, ploidy=2):
        """
        Returns pair:
            - tuple of copy numbers for each region (use ploidy when unknown).
            - total number of unknown copy numbers (where ploidy was used).
        """
        tree = self._trees[sample_id]
        n_regions = len(regions)
        if tree is None:
            pscn = (ploidy,) * n_regions
            return pscn, n_regions

        pscn = []
        n_unknown = 0
        for region in regions:
            cn_regions = tuple(tree.overlap_iter(region))
            if len(cn_regions) == 0:
                pscn.append(ploidy)
                n_unknown += 1
            elif len(cn_regions) > 1:
                common.log('ERROR: Input BED file {} contains several entries for sample {} and region {}'.format(
                    self._filename,
                    sample if sample else '#{}'.format(sample_id),
                    region.to_str(genome) if genome else region))
                return None, n_regions
            elif not cn_regions[0][0].contains(region):
                common.log('ERROR: Input BED file {} contains non-matching entries for sample {} and region {}'.format(
                    self._filename,
                    sample if sample else '#{}'.format(sample_id),
                    region.to_str(genome) if genome else region))
                return None, n_regions
            else:
                pscn.append(cn_regions[0][1])
        return tuple(pscn), n_unknown


def parse_args(prog_name, in_argv, is_new):
    assert prog_name is not None
    usage = ('{prog} {model}(-i <bam> [...] | -I <bam-list>) -t <table> -f <fasta> '
        '{regions}-d <bg-depth> [...] -o <dir> [arguments]').format(prog=prog_name, model='' if is_new else '<model> ',
        regions='(-r <region> [...] | -R <bed>) ' if is_new else '')

    DEFAULT_PSCN_BOUND = (8, 500)
    parser = argparse.ArgumentParser(
        description='Find aggregate and paralog-specific copy number for given unique and duplicated regions.',
        formatter_class=common.SingleMetavar, add_help=False, usage=usage)
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
    io_args.add_argument('-d', '--depth', metavar='<file> [<file> ...]', required=True, nargs='+',
        help='Input files / directories with background read depth.\n'
            'Should be created using "parascopy depth".')
    io_args.add_argument('-o', '--output', metavar='<dir>', required=True,
        help='Output directory.')

    reg_args = parser.add_argument_group('Region arguments')
    if is_new:
        reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region> [<region> ...]',
            help='Region(s) in format "chr" or "chr:start-end".\n'
                'Start and end are 1-based inclusive. Commas are ignored.\n'
                'Optionally, you can provide region name using the format "region::name"\n'
                'For example "-r chr5:70,925,087-70,953,015::SMN1".')
        reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file> [<file> ...]',
            help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).\n'
                'Optional fourth column will be used as a region name.')
    reg_args.add_argument('--force-agcn', metavar='<bed>',
        help='Instead of calculating aggregate copy numbers, use provided bed file.\n'
            'Columns: chrom, start, end, samples, copy_num. Fourth column can be a single sample name;\n'
            'list of sample names separated by ","; or "*" to indicate all samples.')
    if is_new:
        reg_args.add_argument('--modify-ref', metavar='<bed>',
            help='Modify reference copy number using bed file with the same format as `--force-agcn`.\n'
                'Provided values are used for paralog-specific copy number detection.')
    reg_args.add_argument('--regions-subset', nargs='+', metavar='<str> [<str> ...]',
        help='Additionally filter input regions: only use regions with names that are in this list.\n'
            'If the first argument is "!", only use regions not in this list.')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    if is_new:
        filt_args.add_argument('-e', '--exclude', metavar='<expr>',
            default='length < 500 && seq_sim < 0.97',
            help='Exclude duplications for which the expression is true\n[default: %(default)s].')
        filt_args.add_argument('--short', type=int, metavar='<int>', default=500,
            help='Skip regions with short duplications (shorter than <int> bp),\n'
                'not excluded in the -e/--exclude argument [default: %(default)s].')
        filt_args.add_argument('--max-ref-cn', type=int, metavar='<int>', default=10,
            help='Skip regions with reference copy number higher than <int> [default: %(default)s].')
    filt_args.add_argument('--unknown-seq', type=float, metavar='<float>', default=0.1,
        help='At most this fraction of region sequence can be unknown (N) [default: %(default)s].')
    filt_args.add_argument('--skip-unique', action='store_true',
        help='Skip regions without any duplications in the reference genome.')

    aggr_det_args = parser.add_argument_group('Aggregate copy number (agCN) detection arguments')
    if is_new:
        aggr_det_args.add_argument('--min-samples', type=int, metavar='<int>', default=50,
            help='Use multi-sample information if there are at least <int> samples present\n'
                'for the region/PSV [default: %(default)s].')
        aggr_det_args.add_argument('--min-windows', type=int, metavar='<int>', default=5,
            help='Predict aggregate and paralog copy number only in regions with at\n'
                'least <int> windows [default: %(default)s].')
        aggr_det_args.add_argument('--region-dist', type=int, metavar='<int>', default=1000,
            help='Jointly calculate copy number for nearby duplications with equal reference copy number,\n'
                'if the distance between them does not exceed <int> [default: %(default)s].')
        aggr_det_args.add_argument('--window-filtering', type=float, metavar='<float>', default=1,
            help='Modify window filtering: by default window filtering is the same as in the background\n'
                'read depth calculation [default: %(default)s].\n'
                'Values < 1 - discard more windows, > 1 - keep more windows.')
        aggr_det_args.add_argument('--agcn-range', type=int, metavar='<int> <int>', nargs=2, default=(5, 7),
            help='Detect aggregate copy number in a range around the reference copy number [default: 5 7].\n'
                'For example, for a duplication with copy number 8 copy numbers 3-15 can be detected.')
        aggr_det_args.add_argument('--strict-agcn-range', action='store_true',
            help='Detect aggregate copy number strictly within the --agcn-range, even if there are\n'
                'samples with bigger/smaller copy number values.')
        aggr_det_args.add_argument('--agcn-jump', type=int, metavar='<int>', default=6,
            help='Maximal jump in the aggregate copy number between two consecutive windows [default: %(default)s].')
        aggr_det_args.add_argument('--transition-prob', type=float, metavar='<float>', default=-5,
            help='Log10 transition probability for the aggregate copy number HMM [default: %(default)s].')
    aggr_det_args.add_argument('--uniform-initial', action='store_true',
        help='Copy number HMM: use uniform initial distribution and do not update initial probabilities.')
    aggr_det_args.add_argument('--no-multipliers', action='store_false', dest='use_multipliers',
        help='Do not estimate or use read depth multipliers.')
    aggr_det_args.add_argument('--update-agcn', type=float, metavar='<float>', default=40,
        help='Update agCN using psCN probabilities when agCN quality is less than <float> [default: %(default)s].')

    par_det_args = parser.add_argument_group('Paralog-specific copy number (psCN) detection arguments')
    if is_new:
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
    else:
        par_det_args.add_argument('--reliable-threshold', type=float, metavar='<float> <float>', nargs=2,
            help='PSV-reliability thresholds (reliable PSV has all f-values over the threshold).\n'
                'First value is used for gene conversion detection,\n'
                'second value is used to estimate paralog-specific CN.\n'
                'Default: use reliable thresholds from <model>.')

    exec_args = parser.add_argument_group('Execution parameters')
    exec_args.add_argument('--rerun', choices=('full', 'partial', 'none'), metavar='full|partial|none', default='none',
        help='Rerun CN analysis for all loci:\n'
            '    full:    complete rerun,\n'
            '    partial: use pooled reads from a previous run,\n'
            '    none:    skip successfully finished loci [default].')
    exec_args.add_argument('--skip-cn', action='store_true',
        help='Do not calculate agCN and psCN profiles. If this option is set, Parascopy still\n'
            'calculates read depth for duplicated windows and PSV-allelic read depth.')
    exec_args.add_argument('--skip-pscn', action='store_true',
        help='Do not calculate psCN profiles.')
    exec_args.add_argument('-@', '--threads', type=int, metavar='<int>', default=4,
        help='Number of available threads [default: %(default)s].')
    exec_args.add_argument('--samtools', metavar='<path>|none', default='samtools',
        help='Path to samtools executable [default: %(default)s].\n'
            'Use python wrapper if "none", can lead to errors.')
    exec_args.add_argument('--tabix', metavar='<path>', default='tabix',
        help='Path to "tabix" executable [default: %(default)s].\n'
            'Use "none" to skip indexing output files.')

    #####
    vmr_args = parser.add_argument_group('VMR arguments')
    vmr_args.add_argument('--run-vmr', action='store_true')

    thresh_args = parser.add_argument_group('Threshold arguments')
    thresh_mutex = thresh_args.add_mutually_exclusive_group(required=False)
    thresh_mutex.add_argument("--threshold-value", nargs='?', default=1.15, 
        help="set max threshold for vmr using values")
    thresh_mutex.add_argument("--threshold-percentile", nargs='?', 
        help="set max threshold for vmr using percentiles")
    #####

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')
    args = parser.parse_args(in_argv)

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
    else:
        args.modify_ref = None
        args.strict_agcn_range = True
    return args


def _check_number_of_samples(n, min_samples):
    if n < min_samples:
        s = ('\nWARN: Too few input samples ({} < {}):\n'.format(n, min_samples) +
            '    * Paralog-specific copy number will not be estimated,\n'
            '    * Aggregate copy number will be partially estimated.\n'
            'Consider:\n'
            '    * using more samples,\n'
            '    * setting --min-samples,\n'
            '    * using existing model parameters (parascopy cn-using).\n')
        common.log(s)
    elif n < 30:
        common.log('WARN: Few input samples ({}), results may be unstable.'.format(n))


def main(prog_name=None, in_argv=None, is_new=None):
    args = parse_args(prog_name, in_argv, is_new)
    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

    data = DataStructures(args)
    data.prepare()
    genome = data.genome
    genome.compare_with_other(ChromNames.from_table(data.table), args.table)

    directory = args.output
    common.log('Using output directory "{}"'.format(directory))
    common.mkdir(directory)
    common.mkdir(os.path.join(directory, 'loci'))
    write_command(os.path.join(directory, 'command.txt'))

    regions, loaded_models = get_regions(args, genome, load_models=not args.is_new)
    if loaded_models:
        args.min_samples = None
    common.mkdir_clear(os.path.join(directory, 'model'), args.rerun == 'full')
    regions, loaded_models = filter_regions(regions, loaded_models, args.regions_subset)
    
    #####
    vmr_data = (args.depth, (args.threshold_value, args.threshold_percentile))
    bam_wrappers, samples = pool_reads.load_bam_files(args.input, args.input_list, genome, 
                                                      run_vmr=args.run_vmr, vmr_data=vmr_data)
    
    if args.is_new:
        _check_number_of_samples(len(bam_wrappers), args.min_samples)

    data.set_bam_wrappers(bam_wrappers)
    depth_.check_duplicated_samples(bam_wrappers)
    force_agcn = None if args.force_agcn is None else InCopyNums(args.force_agcn, genome, samples)
    modified_ref_cns = None if args.modify_ref is None else InCopyNums(args.modify_ref, genome, samples)

    if loaded_models:
        bg_depth = depth_.Depth.from_filenames(args.depth, samples)
    else:
        bg_depth = depth_.Depth.from_filenames(args.depth, samples, window_filtering_mult=args.window_filtering)
        common.log(bg_depth.params.describe() + '    ============')

    run(regions, data, samples, bg_depth, loaded_models, force_agcn, modified_ref_cns)
    data.close()


if __name__ == '__main__':
    main()
