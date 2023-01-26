#!/usr/bin/env python3

import argparse
import os
import sys
from glob import glob
from time import perf_counter
import operator
import multiprocessing
import pysam
import shutil
import numpy as np
import itertools
import traceback

from .inner import common
from .inner.genome import Genome, Interval
from .inner.duplication import Duplication
from .inner import paralog_cn
from .inner import bam_file as bam_file_
from .inner import variants as variants_
from .inner import itree
from .inner.cn_tools import OutputFiles
from . import long_version
from . import detect_cn
from . import pool_reads


def _write_calling_regions(cn_profiles, samples, genome, assume_cn, max_agcn, filenames):
    pooled_entries = []
    paralog_entries = []

    for sample_id, sample in enumerate(samples):
        for entry in cn_profiles.sample_profiles(sample_id):
            regions = [entry.region1]
            if entry.sample_const_region.regions2 is not None:
                regions.extend(region for region, _strand in entry.sample_const_region.regions2)
            assert entry.n_copies == len(regions)
            ref_agcn = entry.n_copies * 2
            pooled_entry = None

            if assume_cn:
                assumed_pscn, n_unknown = assume_cn.from_regions(regions, sample_id, sample, genome, ploidy=None)
                if n_unknown == 0:
                    pooled_entry = paralog_cn.PooledEntry(entry, artificial_cn=sum(assumed_pscn))
                    pooled_entries.append(pooled_entry)
                    for region_ix, (region, cn) in enumerate(zip(regions, assumed_pscn)):
                        paralog_entry = paralog_cn.ParalogEntry.from_pooled_entry(pooled_entry, region_ix, cn)
                        paralog_entries.append(paralog_entry)
                        assert paralog_entry.region1 == region
                elif n_unknown < len(regions):
                    common.log(
                        'ERROR: Sample {}: Input copy number from "{}" is not available for all repeat copies: {}'
                            .format(sample, assume_cn.filename, ', '.join(region.to_str(genome) for region in regions)))
                    continue

            if pooled_entry is None and entry.pred_cn is not None:
                pooled_entry = paralog_cn.PooledEntry(entry)
                pooled_entries.append(pooled_entry)
                for region_ix, (region, cn, cn_qual) in enumerate(zip(regions, entry.paralog_cn, entry.paralog_qual)):
                    if cn is not None:
                        paralog_entry = paralog_cn.ParalogEntry.from_pooled_entry(pooled_entry, region_ix, cn,
                            cn_filter=entry.paralog_filter, cn_qual=cn_qual)
                        paralog_entries.append(paralog_entry)
                        assert paralog_entry.region1 == region

    pooled_entries.sort()
    with common.open_possible_gzip(filenames.pooled_bed, 'w', bgzip=True) as pooled_bed, \
            open(filenames.cnv_map, 'w') as cnv_map:
        pooled_bed.write('## {}\n'.format(common.command_to_str()))
        pooled_bed.write('#chrom\tstart\tend\tsample\tfilter\tCN\tqual\thomologous_regions\n')
        for entry in pooled_entries:
            if entry.cn <= max_agcn:
                pooled_bed.write(entry.to_str(genome, samples))
                cnv_map.write(entry.to_str_short(genome, samples))
            else:
                cnv_map.write(entry.to_str_zero(genome, samples))

    paralog_entries.sort()
    with common.open_possible_gzip(filenames.paralog_bed, 'w', bgzip=True) as paralog_bed:
        paralog_bed.write('## {}\n'.format(common.command_to_str()))
        paralog_bed.write('#chrom\tstart\tend\tsample\tfilter\tCN\tqual\t'
            'agCN_filter\tagCN\tagCN_qual\thomologous_regions\n')
        for entry in paralog_entries:
            if entry.agcn <= max_agcn:
                paralog_bed.write(entry.to_str(genome, samples))


def _run_freebayes(locus, genome, args, filenames):
    if args.rerun != 'full' and os.path.isfile(filenames.freebayes) and common.non_empty_file(filenames.read_allele):
        return

    fb_executables = (
        args.freebayes,
        '_parascopy_freebayes',
        os.path.join(os.path.dirname(__file__), '../../freebayes/build/freebayes'))
    for path in fb_executables:
        if path is not None and shutil.which(path):
            freebayes = path
            break
    else:
        raise RuntimeError('Cannot run Freebayes, no executable found')

    common.log('[{}] Running Freebayes'.format(locus.name))
    command = [
        freebayes,
        '-f', args.fasta_ref,
        '-@', filenames.psvs,
        '-r', locus.to_str0(genome),
        '-C', args.alternate_count,
        '-F', args.alternate_fraction,
        '-n', args.n_alleles,
        '-p', 2, # Default ploidy, do we need a parameter for that?
        '-A', filenames.cnv_map,
        '--read-allele-obs', filenames.read_allele,
        '-v', filenames.freebayes + '.tmp',
        filenames.pooled,
    ]
    if not common.Process(command).finish():
        raise RuntimeError('Freebayes finished with non-zero code for locus {}'.format(locus.name))
    os.rename(filenames.freebayes + '.tmp', filenames.freebayes)
    common.log('[{}] Freebayes finished'.format(locus.name))


def _create_complement_dupl_tree(duplications, table, genome, padding):
    query_regions = []
    for dupl in duplications:
        region1 = dupl.region1.add_padding(padding)
        region1.trim(genome)
        query_regions.append(region1)

        region2 = dupl.region2.add_padding(padding)
        region2.trim(genome)
        query_regions.append(region2)

    query_regions = Interval.combine_overlapping(query_regions, max_dist=padding)
    dupl_regions = []
    for q_region in query_regions:
        for tup in table.fetch(q_region.chrom_name(genome), q_region.start, q_region.end):
            region1 = Interval(genome.chrom_id(tup[0]), int(tup[1]), int(tup[2]))
            dupl_regions.append(region1)
    dupl_regions = Interval.combine_overlapping(dupl_regions)
    unique_regions = genome.complement_intervals(dupl_regions, include_full_chroms=False)
    return itree.MultiNonOverlTree(unique_regions)


def _write_record_coordinates(filenames, samples, locus, duplications, table, genome, args):
    ix_filename = bam_file_.CoordinatesIndex.index_name(filenames.read_coord)
    if args.rerun != 'full' and os.path.isfile(filenames.read_coord) and common.non_empty_file(ix_filename):
        return

    common.log('[{}] Writing read coordinates'.format(locus.name))
    with pysam.AlignmentFile(filenames.pooled) as in_bam:
        comment_dict = bam_file_.get_comment_items(in_bam)
        max_mate_dist = int(comment_dict['max_mate_dist'])
        unique_tree = _create_complement_dupl_tree(duplications, table, genome, padding=max_mate_dist * 2)
        bam_file_.write_record_coordinates(in_bam, samples, unique_tree, genome, filenames.read_coord, comment_dict)
    common.log('[{}] Finished writing read coordinates'.format(locus.name))


def _analyze_sample(locus, sample_id, sample, all_read_allele_obs, coord_index, cn_profiles, assume_cn,
        genome, varcall_params, out):
    common.log('[{}] Analyzing sample {}'.format(locus.name, sample))
    read_collection = variants_.ReadCollection(sample_id, sample, coord_index)
    mean_read_len = read_collection.mean_len
    skip_paralog_gts = varcall_params.skip_paralog_gts

    debug_out = out.get('debug')
    all_genotype_predictions = []
    potential_psv_gts = []
    for variant_obs in all_read_allele_obs:
        gt_pred = variants_.VariantGenotypePred(sample_id, sample, variant_obs, cn_profiles, assume_cn)
        gt_pred.init_read_counts(read_collection.read_positions, varcall_params, out.get('debug_obs'))
        gt_pred.update_read_locations()
        gt_pred.init_genotypes(varcall_params, mean_read_len, out.genotypes, only_pooled=skip_paralog_gts)
        all_genotype_predictions.append(gt_pred)

        if skip_paralog_gts:
            variant_obs.update_vcf_records(gt_pred, genome)
            continue

        for psv_obs in variant_obs.psv_observations:
            psv_gt_pred = variants_.VariantGenotypePred(sample_id, sample, psv_obs, cn_profiles, assume_cn)
            psv_gt_pred.init_read_counts(read_collection.read_positions, varcall_params, None)
            psv_gt_pred.init_genotypes(varcall_params, mean_read_len, None, only_pooled=True)

            if psv_obs.get_psv_usability(psv_gt_pred, varcall_params, out.psv_use, debug_out):
                potential_psv_gts.append(psv_gt_pred)
    if skip_paralog_gts:
        return

    read_collection.find_possible_locations()
    n_psvs1 = len(potential_psv_gts)
    common.log('[{}] Sample {}: {} potentially informative PSVs'.format(locus.name, sample, n_psvs1))

    if n_psvs1:
        if debug_out:
            debug_out.write('{}: Examining PSV conflicts\n'.format(sample))
        informative_psv_gts = variants_.VariantGenotypePred.select_non_conflicing_psvs(potential_psv_gts,
            coord_index.max_mate_dist, varcall_params.error_rate[0], debug_out)
        common.log('[{}] Sample {}: selected {} informative PSVs'.format(locus.name, sample, len(informative_psv_gts)))
    else:
        informative_psv_gts = potential_psv_gts
    read_collection.add_psv_observations(informative_psv_gts, coord_index.max_mate_dist, varcall_params.no_mate_penalty)

    if debug_out is not None:
        read_collection.debug_read_probs(genome, debug_out)
    assert len(all_read_allele_obs) == len(all_genotype_predictions)
    for variant_obs, gt_pred in zip(all_read_allele_obs, all_genotype_predictions):
        gt_pred.utilize_reads(varcall_params, out.genotypes)
        variant_obs.update_vcf_records(gt_pred, genome)


def _debug_write_read_hashes(bam_filename, out):
    common.log('DEBUG: Get read hashes')
    with pysam.AlignmentFile(bam_filename) as in_bam:
        for record in in_bam.fetch():
            out.write('{}\t{:x}\n'.format(record.query_name,
                bam_file_.string_hash_fnv1(record.query_name, record.is_read1)))


def analyze_locus(locus, model_params, data, samples, assume_cn):
    genome = data.genome
    table = data.table
    args = data.args
    filenames = argparse.Namespace()
    filenames.par_dir = os.path.join(args.parascopy, locus.os_name)
    filenames.out_dir = os.path.join(args.output, locus.os_name)
    common.mkdir(filenames.out_dir)

    filenames.cn_res = os.path.join(filenames.par_dir, 'res.samples.bed.gz')
    filenames.psvs = os.path.join(filenames.par_dir, 'psvs.vcf.gz')
    if not os.path.isfile(filenames.psvs):
        raise FileNotFoundError('Could not read PSVs from "{}"'.format(filenames.psvs))

    duplications = model_params.get_duplications(table, locus, genome)
    duplications = [detect_cn.transform_duplication(dupl, locus, genome) for dupl in duplications]
    if data.bam_wrappers is None:
        filenames.pooled = os.path.join(filenames.par_dir, 'pooled_reads.bam')
        if not os.path.isfile(filenames.pooled):
            raise FileNotFoundError('Could not read pooled reads from "{}"'.format(filenames.pooled))
    else:
        filenames.pooled = os.path.join(filenames.out_dir, 'pooled_reads.bam')
        if not os.path.isfile(filenames.pooled) or args.rerun == 'full':
            pool_reads.pool(data.bam_wrappers, filenames.pooled, locus, duplications, genome,
                samtools=args.samtools, verbose=True)

    common.log('Analyzing locus [{}]'.format(locus.name))
    filenames.subdir = os.path.join(filenames.out_dir, 'var_extra')
    common.mkdir_clear(filenames.subdir, rewrite=args.rerun == 'full')
    filenames.success = os.path.join(filenames.subdir, 'success')
    if os.path.isfile(filenames.success):
        if args.rerun == 'none':
            common.log('Skipping locus {}'.format(locus.name))
            return
        os.remove(filenames.success)

    with pysam.VariantFile(filenames.psvs) as psvs_vcf:
        psv_records = list(psvs_vcf)
    varcall_params = variants_.VarCallParameters(args, samples)

    filenames.read_coord = os.path.join(filenames.subdir, 'read_coordinates.bin')
    _write_record_coordinates(filenames, samples, locus, duplications, table, genome, args)

    cn_profiles = paralog_cn.CopyNumProfiles(filenames.cn_res, genome, samples, locus.chrom_id)
    filenames.read_allele = os.path.join(filenames.subdir, 'read_allele_obs.bin')
    filenames.freebayes = os.path.join(filenames.subdir, 'freebayes.vcf')
    filenames.cnv_map = os.path.join(filenames.subdir, 'cnv_map.bed')
    filenames.pooled_bed = os.path.join(filenames.out_dir, 'variants_pooled.bed.gz')
    filenames.paralog_bed = os.path.join(filenames.out_dir, 'variants.bed.gz')
    _write_calling_regions(cn_profiles, samples, genome, assume_cn, args.max_agcn, filenames)

    _run_freebayes(locus, genome, args, filenames)

    dupl_pos_finder = variants_.DuplPositionFinder(locus.chrom_id, duplications)
    with open(filenames.read_allele, 'rb') as ra_inp, pysam.VariantFile(filenames.freebayes) as vcf_file:
        all_read_allele_obs = variants_.read_freebayes_results(ra_inp, samples, vcf_file, dupl_pos_finder)

    all_read_allele_obs = variants_.add_psv_variants(locus, all_read_allele_obs, psv_records, genome, varcall_params)
    vcf_headers = variants_.VariantReadObservations.create_vcf_headers(genome, sys.argv, samples)
    for read_allele_obs in all_read_allele_obs:
        read_allele_obs.init_vcf_records(genome, vcf_headers)

    extra_files = dict(genotypes='genotypes.csv', psv_use='psv_use.csv')
    if args.debug:
        extra_files.update(dict(debug_reads='debug_reads.log', debug_obs='debug_obs.log', debug='debug.log'))
    with bam_file_.CoordinatesIndex(filenames.read_coord, samples, genome) as coord_index, \
            OutputFiles(filenames.subdir, extra_files) as out:
        if args.debug:
            out.debug_reads.write('name\thash\n')
            _debug_write_read_hashes(filenames.pooled, out.debug_reads)
            out.debug_obs.write('variant\tread_hash\tread_mate\tobs_allele\tbasequal\n')

        out.genotypes.write('# Format: genotype=-log10(prob).\n')
        out.genotypes.write('sample\tpos\ttype\tgenotype_probs\n')
        out.psv_use.write('sample\tpsv\tpsv_status\tvar_to_ext_copy\text_pscn\t'
            'read_counts\tpooled_gt\text_alleles\tparalog_gt\tref_gt_qual\tuse\treason\n')
        for sample_id, sample in enumerate(samples):
            _analyze_sample(locus, sample_id, sample, all_read_allele_obs, coord_index, cn_profiles, assume_cn,
                genome, varcall_params, out)

    common.log('[{}] Writing output VCF file'.format(locus.name))
    filenames.out_vcf = os.path.join(filenames.out_dir, 'variants.vcf.gz')
    filenames.out_pooled_vcf = os.path.join(filenames.out_dir, 'variants_pooled.vcf.gz')
    variants_.write_vcf_file(filenames, vcf_headers, all_read_allele_obs, genome, args.tabix)
    os.mknod(filenames.success)
    common.log('[{}] Success'.format(locus.name))


def _analyze_locus_wrapper(locus_ix, locus, model_params, data, samples, assume_cn):
    try:
        data.prepare()
        res = analyze_locus(locus, model_params, data, samples, assume_cn)
        assert res is None
        return locus_ix, res
    except Exception as exc:
        trace = traceback.format_exc().strip().split('\n')
        trace = '\n'.join(' ' * 11 + s for s in trace)
        common.log('Error: region {}:\n{}'.format(locus.full_name(data.genome), trace))
        exc_str = '{}: {}'.format(type(exc).__name__, exc)
        return locus_ix, exc_str
    finally:
        data.close()


def run(loci, loaded_models, data, samples, assume_cn):
    n_loci = len(loci)
    threads = max(1, min(n_loci, data.args.threads))
    results = []
    if threads == 1:
        common.log('Using 1 thread')
        for locus_ix, (locus, model_params) in enumerate(zip(loci, loaded_models)):
            results.append(_analyze_locus_wrapper(locus_ix, locus, model_params, data, samples, assume_cn))

    else:
        def callback(res):
            results.append(res)

        def err_callback(exc):
            common.log('Thread finished with an exception:\n{}'.format(exc))
            os._exit(1)

        common.log('Using {} threads'.format(threads))
        pool = multiprocessing.Pool(threads)
        for locus_ix, (locus, model_params) in enumerate(zip(loci, loaded_models)):
            pool.apply_async(_analyze_locus_wrapper,
            args=(locus_ix, locus, model_params, data.copy(), samples, assume_cn),
            callback=callback, error_callback=err_callback)
        pool.close()
        pool.join()

    results.sort()
    assert len(results) == n_loci and all(i == locus_ix for i, (locus_ix, _) in enumerate(results))
    successful = [exc is None for _, exc in results]

    args = data.args
    out_dir = args.output
    genome = data.genome
    tabix = args.tabix
    successful_loci = list(itertools.compress(loci, successful))
    common.log('Merging output files')
    detect_cn.join_vcf_files([os.path.join(out_dir, region.os_name, 'variants.vcf.gz') for region in successful_loci],
        os.path.join(out_dir, 'variants.vcf.gz'), genome, tabix, merge_duplicates=True)
    detect_cn.join_vcf_files(
        [os.path.join(out_dir, region.os_name, 'variants_pooled.vcf.gz') for region in successful_loci],
        os.path.join(out_dir, 'variants_pooled.vcf.gz'), genome, tabix, merge_duplicates=True)
    detect_cn.join_bed_files(
        [os.path.join(out_dir, region.os_name, 'variants.bed.gz') for region in successful_loci],
        os.path.join(out_dir, 'variants.bed.gz'), genome, tabix)
    detect_cn.join_bed_files(
        [os.path.join(out_dir, region.os_name, 'variants_pooled.bed.gz') for region in successful_loci],
        os.path.join(out_dir, 'variants_pooled.bed.gz'), genome, tabix)

    n_successes = sum(successful)
    if n_successes < n_loci:
        common.log('==============================================')
        common.log('ERROR: Could not finish the following loci:')
        i = 0
        for locus, (_, exc) in zip(loci, results):
            if exc is not None:
                common.log('    {}: {}'.format(locus.full_name(genome), exc))
                i += 1
                if i >= 10:
                    break
        if i < n_loci - n_successes:
            common.log('    ...')

    if n_successes == 0:
        common.log('Failure! No loci were analyzed successfully.')
    else:
        common.log('Success [{} loci out of {}]'.format(n_successes, n_loci))


def _check_sample_subset(samples, sample_subset, missing_where):
    sample_set_diff = set(sample_subset) - set(samples)
    if sample_set_diff:
        first_sample = next(iter(sample_set_diff))
        if len(sample_set_diff) > 1:
            err = 'ERROR: {} samples (e.g. {}) is present in the sample subset (-s/--samples), but not in the input {}.'\
                .format(len(sample_set_diff), first_sample, missing_where)
        else:
            err = 'ERROR: Sample {} is present in the sample subset (-s/--samples), but not in the input {}.'.format(
                first_sample, missing_where)
        common.log(err)
        exit(1)


def main(prog_name=None, in_argv=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Call variants in duplicated regions.',
        formatter_class=argparse.RawTextHelpFormatter, add_help=False,
        usage='{} -p <dir> [-i <bam> ... | -I <bam-list>] -t <table> -f <fasta> [-o <dir>]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-p', '--parascopy', metavar='<dir>', required=True,
        help='Input directory with Parascopy copy number estimates.\n'
            'By default, pooled reads from the --parascopy directory are taken,\n'
            'however, one may supply alignment files using -i and -I arguments.')

    inp_me = io_args.add_mutually_exclusive_group(required=False)
    inp_me.add_argument('-i', '--input', metavar='<file>', nargs='+',
        help='Optional: Input indexed BAM/CRAM files.\n'
            'All entries should follow the format "filename[::sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input-list.')
    inp_me.add_argument('-I', '--input-list', metavar='<file>',
        help='Optional: A file containing a list of input BAM/CRAM files.\n'
            'All lines should follow the format "filename[ sample]"\n'
            'If sample name is not set, all reads in a corresponding file should have a read group (@RG).\n'
            'Mutually exclusive with --input.\n\n')

    io_args.add_argument('-t', '--table', metavar='<file>', required=True,
        help='Input indexed bed table with information about segmental duplications.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file>', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<dir>', required=False,
        help='Output directory. Required if -i or -I arguments were used.')

    fb_args = parser.add_argument_group('Freebayes parameters')
    fb_args.add_argument('--freebayes', metavar='<path>', required=False,
        help='Optional: path to the modified Freebayes executable.')
    fb_args.add_argument('--alternate-count', type=int, metavar='<int>', default=4,
        help='Minimum alternate allele read count (in at least one sample),\n'
            'corresponds to freebayes parameter -C <int> [default: %(default)s].')
    fb_args.add_argument('--alternate-fraction', type=float, metavar='<float>', default=0,
        help='Minimum alternate allele read fraction (in at least one sample),\n'
            'corresponds to freebayes parameter -F <float> [default: %(default)s].')
    fb_args.add_argument('--n-alleles', type=int, metavar='<int>', default=3,
        help='Use at most <int> best alleles (set 0 to all),\n'
             'corresponds to freebayes parameter -n <int> [default: %(default)s].')

    call_args = parser.add_argument_group('Variant calling parameters')
    call_args.add_argument('--skip-paralog', action='store_true',
        help='Do not calculate paralog-specific genotypes.')
    call_args.add_argument('--limit-qual', type=float, metavar='<float>', default=1,
        help='Skip SNVs that do not overlap PSVs and have Freebayes quality\n'
            'under <float> [default: %(default)s].')
    call_args.add_argument('--assume-cn', metavar='<bed>',
        help='Instead of using Parascopy paralog-specific copy number values, use copy number from\n'
            'the input file with columns "chrom start end samples copy_num".\n'
            'Fourth input column can be a single sample name; list of sample names separated by ",";\n'
            'or "*" to indicate all samples.')
    call_args.add_argument('--limit-pooled', type=float, metavar='<float>', default=-5,
        help='Based solely on allelic read depth, ignore pooled genotypes with probabilities\n'
            'under 10^<float> [default: %(default)s].')
    call_args.add_argument('--mutation-rate', type=float, metavar='<float>', default=-3,
        help='Log10 mutation rate (used for calculating genotype priors) [default: %(default)s].')
    call_args.add_argument('--error-rate', nargs=2, type=float, metavar='<float> <float>', default=(0.01, 0.01),
        help='Two error rates: first for SNPs, second for indels [default: 0.01 0.01].')
    call_args.add_argument('--base-qual', nargs=2, type=int, metavar='<int> <int>', default=(10, 10),
        help='Ignore observations with low base quality (first for SNPs, second for indels) [default: 10 10].')
    # call_args.add_argument('--use-af', metavar='yes|no|over-N', default='over-20',
    #     help='Use alternate fraction (AF) for calculating genotype priors:\n'
    #         'yes[y], no[n], over-N: use AF if there are at least N samples [default: %(default)s].')
    call_args.add_argument('--no-mate-penalty', type=float, metavar='<float>', default=-5,
        help='Penalize possible paired-read alignment positions in case they do not match\n'
            'second read alignment position (log10 penalty) [default: %(default)s].')
    call_args.add_argument('--psv-ref-gt', type=float, metavar='<float>', default=20,
        help='Use all PSVs (even unreliable) if they have a reference paralog-specific\n'
            'genotype (genotype quality >= <float>) [default: %(default)s].')
    call_args.add_argument('--limit-depth', nargs=2, type=int, metavar='<int> <int>', default=(3, 2000),
        help='Min and max variant read depth [default: 3, 2000].')
    call_args.add_argument('--max-strand-bias', type=float, metavar='<float>', default=30,
        help='Maximum strand bias (Phred p-value score) [default: %(default)s].')
    call_args.add_argument('--max-agcn', type=int, metavar='<int>', default=10,
        help='Maximum aggregate copy number [default: %(default)s].')

    exec_args = parser.add_argument_group('Execution parameters')
    exec_args.add_argument('--rerun', choices=('full', 'partial', 'none'), metavar='full|partial|none', default='none',
        help='Rerun analysis for all loci:\n'
            '    full:    complete rerun,\n'
            '    partial: use already calculated read-allele observations,\n'
            '    none:    skip successfully finished loci [default].')
    exec_args.add_argument('-s', '--samples', metavar='(<file>|<name>) ...', nargs='+',
        help='Limit the analysis to the provided sample names.\n'
            'Input may consist of sample names and files with sample names (filename should contain "/").')
    exec_args.add_argument('-@', '--threads', type=int, metavar='<int>', default=4,
        help='Number of available threads [default: %(default)s].')
    exec_args.add_argument('--regions-subset', nargs='+', metavar='<str> [<str> ...]',
        help='Additionally filter input regions: only use regions with names that are in this list.\n'
            'If the first argument is "!", only use regions not in this list.')
    exec_args.add_argument('--samtools', metavar='<path>', default='samtools',
        help='Path to "samtools" executable [default: %(default)s].')
    exec_args.add_argument('--tabix', metavar='<path>', default='tabix',
        help='Path to "tabix" executable [default: %(default)s].\n'
            'Use "none" to skip indexing output files.')
    exec_args.add_argument('--debug', action='store_true',
        help='Output additional debug information.')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version', version=long_version(), help='Show version.')

    args = parser.parse_args(in_argv)
    common.check_executable(args.samtools)
    if args.tabix != 'none':
        common.check_executable(args.tabix)
    if not os.path.isdir(args.parascopy):
        common.log('Input directory "{}" does not exist'.format(args.parascopy))
        exit(1)

    if args.output is None:
        args.output = args.parascopy
        separate_output = False
    else:
        assert args.output != args.parascopy
        common.mkdir(args.output)
        separate_output = True
    detect_cn.write_command(os.path.join(args.output, 'command_call.txt'))

    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)
    data = detect_cn.DataStructures(args)
    data.prepare()
    genome = data.genome

    with pysam.VariantFile(os.path.join(args.parascopy, 'psvs.vcf.gz')) as psvs_vcf:
        samples = bam_file_.Samples(psvs_vcf.header.samples)
    aln_samples = None
    if args.input is not None or args.input_list is not None:
        if not separate_output:
            common.log('ERROR: -o/--output argument is required if -i/--input or -I/--input-list arguments are used!')
            exit(1)
        bam_wrappers, aln_samples = pool_reads.load_bam_files(args.input, args.input_list, genome)
        data.set_bam_wrappers(bam_wrappers)
        from .depth import check_duplicated_samples
        check_duplicated_samples(bam_wrappers)

    if args.samples is not None:
        sample_subset = common.file_or_str_list(args.samples)
        sample_superset = samples if aln_samples is None else set(samples) & set(aln_samples)
        _check_sample_subset(sample_superset, sample_subset, 'alignment files or parascopy directory')
        samples = bam_file_.Samples(sample_subset)
    elif aln_samples is not None:
        _check_sample_subset(samples, aln_samples, 'parascopy directory')
        samples = aln_samples
    if len(samples) == 0:
        common.log('ERROR: Zero samples to analyze!')
        exit(1)

    assume_cn = None if args.assume_cn is None else detect_cn.InCopyNums(args.assume_cn, genome, samples)
    args.model = (os.path.join(args.parascopy, 'model'),)
    regions, loaded_models = detect_cn.get_regions(args, genome, load_models=True)
    regions, loaded_models = detect_cn.filter_regions(regions, loaded_models, args.regions_subset)

    run(regions, loaded_models, data, samples, assume_cn)
    data.close()


if __name__ == '__main__':
    main()
