from scipy import stats
from scipy import optimize
from scipy.special import logsumexp
from scipy.cluster import hierarchy
import scipy
import warnings
import numpy as np
import sys
import collections
import operator
import itertools
from time import perf_counter
from datetime import timedelta
from enum import Enum

from . import common
from . import variants as variants_
from . import cn_tools
from . import cn_hmm
from . import itree
from .genome import Interval
from .. import __pkg_name__, __version__


_BETA_DISTR_A = 5.0
_BETA_DISTR_B = 1.0

def _get_f_prior(f_values):
    EPSILON = 1e-6
    max_f_value = np.max(f_values)
    beta_cdfs = stats.beta.logcdf((max_f_value, max_f_value - EPSILON), _BETA_DISTR_A, _BETA_DISTR_B)
    return logsumexp(beta_cdfs, b=(1, -1))


def _e_step(psv_infos, psv_f_values, genotype_priors, n_samples):
    """
    Returns
        - total ln likelihood,
        - matrix of P(sample_genotype | allele_counts, psvs), size = (n_samples, n_sample_genotypes).
    """
    n_psvs = len(psv_infos)
    n_sample_genotypes = len(genotype_priors)
    prob_matrix = np.zeros((n_samples, n_sample_genotypes))
    usable_samples = np.zeros(n_samples, dtype=np.bool)
    total_f_prior = 0

    for psv_info in psv_infos:
        if not psv_info.in_em:
            continue
        sample_ids = psv_info.sample_ids
        usable_samples[sample_ids] = True

        precomp_data = psv_info.precomp_data_ref_cn
        psv_gt_probs = psv_info.em_psv_gt_probs
        exponent = psv_info.info_content

        psv_ix = psv_info.psv_ix
        f_values = psv_f_values[psv_ix]
        total_f_prior += _get_f_prior(f_values)
        f_pow_combs = precomp_data.f_power_combinations(f_values)
        all_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)

        for sample_gt_ix, psv_gt_coefs in enumerate(all_psv_gt_coefs):
            curr_probs = exponent * logsumexp(psv_gt_coefs + psv_gt_probs, axis=1)
            prob_matrix[sample_ids, sample_gt_ix] += curr_probs

    with np.errstate(invalid='ignore'):
        prob_matrix += genotype_priors[np.newaxis, :]
        prob_matrix[~usable_samples] = np.nan
        row_sums = logsumexp(prob_matrix, axis=1)
        total_lik = np.sum(row_sums[usable_samples]) + total_f_prior
        prob_matrix -= row_sums[:, np.newaxis]
    return total_lik, prob_matrix


def _minus_lik_fn(gt_probs_regular, precomp_data, psv_gt_probs):
    n_samples, n_genotypes = gt_probs_regular.shape

    def fn(f_values):
        beta_prior = -_get_f_prior(f_values)
        if np.isinf(beta_prior):
            return beta_prior

        inner_term = np.full((n_samples, n_genotypes), np.nan)
        f_pow_combs = precomp_data.f_power_combinations(f_values)
        all_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)

        for sample_gt_ix, psv_gt_coefs in enumerate(all_psv_gt_coefs):
            inner_term[:, sample_gt_ix] = logsumexp(psv_gt_coefs + psv_gt_probs, axis=1)
        return -np.sum(inner_term * gt_probs_regular) + beta_prior
    return fn


def _m_step(sample_gt_probs, psv_infos, psv_ixs, prev_psv_f_values):
    """
    Returns
        - matrix with f-values (n_psvs x n_copies),
        - total_likelihood.
    """
    n_psvs = len(psv_infos)
    n_samples, n_genotypes = sample_gt_probs.shape
    n_copies = prev_psv_f_values.shape[1]
    psv_f_values = np.full_like(prev_psv_f_values, np.nan)

    OPTS = dict(maxiter=50)
    METHOD = 'L-BFGS-B'
    # If we use (0, 1) here, the optimization will not work properly because of the inifinite prior at the boundaries.
    bounds = ((1e-6, 1 - 1e-6),) * n_copies

    sample_gt_probs_regular = np.exp(sample_gt_probs)
    total_lik = 0
    for psv_ix in psv_ixs:
        psv_info = psv_infos[psv_ix]
        sample_ids = psv_info.sample_ids
        assert len(sample_ids) > 0

        sample_gt_probs_subset = sample_gt_probs_regular[sample_ids]
        precomp_data = psv_info.precomp_data_ref_cn
        psv_gt_probs = psv_info.em_psv_gt_probs
        prev_f_values = prev_psv_f_values[psv_ix]

        minus_lik = _minus_lik_fn(sample_gt_probs_subset, precomp_data, psv_gt_probs)
        sol = optimize.minimize(minus_lik, x0=prev_f_values, bounds=bounds, options=OPTS, method=METHOD)

        total_lik -= sol.fun
        psv_f_values[psv_ix] = sol.x
    return psv_f_values, total_lik


def _match_psv_alleles(psv, regions2, genome):
    pos2 = psv.info['pos2']
    if len(pos2) == 1:
        return (0, 1)

    res = [None] * (len(regions2) + 1)
    res[0] = 0
    for entry in pos2:
        chrom, pos, strand, allele = entry.split(':')
        chrom_id = genome.chrom_id(chrom)
        pos = int(pos) - 1
        strand = strand == '+'
        allele = int(allele)
        if not strand:
            pos += len(psv.alleles[allele]) - 1
        for i, (region2, strand2) in enumerate(regions2):
            if res[i + 1] is None and strand == strand2 and region2.contains_point(chrom_id, pos):
                res[i + 1] = allele
                break
        else:
            s = 'Cannot match PSV alleles:\n'
            for i, entry2 in enumerate(pos2, start=1):
                s += '    {} {}{}\n'.format(i, entry2, '  !!!' if entry == entry2 else '')
            s += 'With regions:\n    '
            s += cn_tools.regions2_str(regions2, genome, use_comma=True, sep='\n    ')
            common.log(s)
            raise RuntimeError('Cannot match PSV alleles')
    return tuple(res)


def _select_psv_sample_pairs(region_group_extra, samples, outp, min_samples):
    group = region_group_extra.region_group
    psv_infos = region_group_extra.psv_infos
    sample_reliable_regions = region_group_extra.sample_reliable_regions
    group_name = group.name
    n_psvs = len(psv_infos)
    n_samples = len(samples)

    outp.write('# Group {}. Sample regions with agCN = refCN = {}:\n'.format(group_name, group.cn))
    reliable_regions = [None] * n_samples
    n_reliable = 0
    for sample_id, sample in enumerate(samples):
        sample_region = sample_reliable_regions[sample_id]
        if sample_region is None:
            continue
        outp.write('#    {}: {:,}-{:,}\n'.format(sample, sample_region.start + 1, sample_region.end))
        n_reliable += 1

    if n_reliable < min_samples:
        outp.write('# Too few samples ({} < {}).\n'.format(n_reliable, min_samples))
        return

    for psv_ix, psv_info in enumerate(psv_infos):
        info_str = ''
        use_samples = np.zeros(n_samples, dtype=np.bool)
        for sample_id, sample_region in enumerate(sample_reliable_regions):
            good_obs = psv_info.psv_gt_probs[sample_id] is not None
            is_ref = sample_region is not None \
                and sample_region.start <= psv_info.start and psv_info.end <= sample_region.end
            info_str += '\t{}{}'.format('+' if good_obs else '-', '+' if is_ref else '-')
            use_samples[sample_id] = good_obs and is_ref

        psv_info.set_use_samples(use_samples)
        psv_info.in_em = psv_info.n_used_samples >= min_samples
        outp.write('{}\t{}:{}\t{}{}\n'.format(group_name, psv_info.chrom, psv_info.start + 1,
            psv_info.n_used_samples, info_str))


class _PsvInfo:
    def __init__(self, psv_ix, psv, region_group, n_samples, genome):
        self.psv_ix = psv_ix
        self.psv = psv
        self.chrom = psv.chrom
        self.start = psv.start
        self.end = psv.start + len(psv.ref)
        self.is_indel = len(set(map(len, psv.alleles))) != 1
        self.allele_corresp = _match_psv_alleles(psv, region_group.regions2, genome)
        self.ref_cn = region_group.cn

        self.info_content = np.nan
        self.n_used_samples = 0
        self.in_em = False
        self.sample_ids = None

        # keys: agCN values.
        self.precomp_datas = {}
        self.sample_cns = [None] * n_samples
        self.psv_gt_probs = [None] * n_samples
        self.em_psv_gt_probs = None
        self.precomp_data_ref_cn = None
        self.support_matrix = None

    def set_use_samples(self, use_samples):
        self.sample_ids = np.where(use_samples)[0]
        self.n_used_samples = len(self.sample_ids)

    def distance(self, other):
        if other.psv_ix < self.psv_ix:
            return self.start - other.end
        return other.start - self.end

    def __str__(self):
        return '{}:{}'.format(self.chrom, self.start + 1)

    def str_ext(self):
        return '{} ({}  information content {:.4f})'.format(
            self, 'indel,' if self.is_indel else 'snp,  ', self.info_content)

    def __lt__(self, other):
        """
        Used in sort, so better PSV would have lower key.
        """
        if not other.in_em:
            return True
        if not self.in_em:
            return False
        if (self.info_content < 0.9 or other.info_content < 0.9) \
                and (abs(self.info_content - other.info_content) >= 0.01):
            return self.info_content > other.info_content
        if self.is_indel != other.is_indel:
            return other.is_indel
        return self.start < other.start

    def check_complicated_pos(self, region, region_seq, homopolymer_len=5, distance_to_edge=10):
        if self.start - region.start < distance_to_edge or region.end - self.end < distance_to_edge:
            self.in_em = False
            return
        if self.end == region.end:
            return
        after_psv = region_seq[self.end - region.start : self.end + homopolymer_len - region.start]
        if len(set(after_psv)) == 1:
            self.in_em = False

    def create_em_psv_gt_probs(self):
        if not self.n_used_samples:
            return
        self.precomp_data_ref_cn = self.precomp_datas[self.ref_cn]
        self.em_psv_gt_probs = np.zeros((self.n_used_samples, self.precomp_data_ref_cn.n_psv_genotypes))
        for i, sample_id in enumerate(self.sample_ids):
            assert self.psv_gt_probs[sample_id] is not None
            self.em_psv_gt_probs[i] = self.psv_gt_probs[sample_id]


def _calculate_psv_info_content(group_name, psv_infos, min_samples, outp):
    outp.write('Region group {}\n'.format(group_name))
    for psv_info in psv_infos:
        if not psv_info.n_used_samples:
            outp.write('{}   no applicable samples, skipping.\n'.format(psv_info))
            continue

        precomp_data = psv_info.precomp_data_ref_cn
        psv_gt_probs = psv_info.em_psv_gt_probs
        n_alleles = precomp_data.n_alleles
        gt_mult = np.fromiter((n_alleles - gt.count(0) - 1 for gt in precomp_data.psv_genotypes),
            np.int16, len(precomp_data.psv_genotypes))

        sum_info_content = 0.0
        inform_samples = 0
        for i in range(psv_info.n_used_samples):
            curr_info_content = np.sum(np.exp(psv_gt_probs[i]) * gt_mult)
            sum_info_content += curr_info_content
            if curr_info_content >= 0.8:
                inform_samples += 1
        outp.write('{}   informative: {}/{} samples. '.format(psv_info, inform_samples, psv_info.n_used_samples))

        psv_info.info_content = sum_info_content / psv_info.n_used_samples / (n_alleles - 1)
        outp.write('Information content: {:.4f}\n'.format(psv_info.info_content))
        if psv_info.info_content < 1e-6:
            psv_info.in_em = False


def _filter_close_psvs(psv_infos, outp, close_psv_dist):
    outp.write('\nFiltering closeby PSVs\n')
    n_psvs = len(psv_infos)
    removed_a = sum(not psv_info.in_em for psv_info in psv_infos)

    for psv_info in sorted(psv_infos):
        psv_ix = psv_info.psv_ix
        if not psv_info.in_em:
            continue

        for step in (-1, 1):
            for j in itertools.count(psv_ix + step, step):
                if j < 0 or j >= n_psvs:
                    break
                oth_info = psv_infos[j]
                dist = oth_info.distance(psv_info)
                if dist > close_psv_dist:
                    break
                if oth_info.in_em:
                    oth_info.in_em = False
                    outp.write('Removing PSV {} - close to PSV {}, distance {}\n'
                        .format(oth_info.str_ext(), psv_info.str_ext(), dist))

    removed_b = sum(not psv_info.in_em for psv_info in psv_infos)
    outp.write('Total {:5} PSVs\n'.format(n_psvs))
    outp.write('    - {:5} uninformative PSVs\n'.format(removed_a))
    outp.write('    - {:5} closeby PSVs\n'.format(removed_b - removed_a))
    outp.write('    = {:5} retained PSVs\n'.format(n_psvs - removed_b))


def _define_sample_gt_priors(n_copies, sample_genotypes):
    # Sample priors are distributed by distance to the reference genotypes.
    # Minimal prior is for (4,0) or (6,0,0) ..., and it is 1e-6.
    MIN_PRIOR = -6 * np.log(10)

    ref_gt = np.full(n_copies, 2)
    max_dist = 2 * (n_copies - 1)
    priors = np.full(len(sample_genotypes), np.nan)
    for i, gt in enumerate(sample_genotypes):
        dist_to_ref = np.sum(np.abs(ref_gt - gt)) // 2
        priors[i] = MIN_PRIOR * dist_to_ref / max_dist
    priors -= logsumexp(priors)
    return priors


def _cluster_psvs(psv_infos, psv_counts, n_samples):
    n_psvs = len(psv_infos)

    ref_fractions = np.full((n_psvs, n_samples), np.nan)
    for psv_info in psv_infos:
        allele_corresp = psv_info.allele_corresp
        mult = len(allele_corresp) / allele_corresp.count(0)
        for sample_id in psv_info.sample_ids:
            allele_counts = psv_counts[psv_info.psv_ix][sample_id].allele_counts
            ref_fractions[psv_info.psv_ix, sample_id] = allele_counts[0] / sum(allele_counts) * mult

    cor_matrix = np.full((n_psvs, n_psvs), np.nan)
    np.fill_diagonal(cor_matrix, 1.0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=stats.PearsonRConstantInputWarning)
        for psv_i in range(n_psvs):
            for psv_j in range(psv_i + 1, n_psvs):
                # Samples with present observations for both PSVs.
                mask = ~np.logical_or(np.isnan(ref_fractions[psv_i]), np.isnan(ref_fractions[psv_j]))
                if np.sum(mask) < 5:
                    continue
                cor, _ = stats.pearsonr(ref_fractions[psv_i, mask], ref_fractions[psv_j, mask])
                cor_matrix[psv_i, psv_j] = cor_matrix[psv_j, psv_i] = cor

    dist_matrix = np.full((n_psvs, n_psvs), np.nan)
    np.fill_diagonal(dist_matrix, 0)
    use_psvs = np.ones(n_psvs, dtype=np.bool)
    for psv_i in range(n_psvs):
        if not use_psvs[psv_i]:
            continue
        for psv_j in range(psv_i + 1, n_psvs):
            if not use_psvs[psv_j]:
                continue
            mask_i = ~np.isnan(cor_matrix[psv_i])
            mask_j = ~np.isnan(cor_matrix[psv_j])
            mask = np.logical_and(mask_i, mask_j)
            mask_size = np.sum(mask)
            if not mask_size:
                if np.sum(mask_i) > np.sum(mask_j):
                    use_psvs[psv_j] = False
                else:
                    use_psvs[psv_i] = False
                    break
            else:
                dist_matrix[psv_i, psv_j] = dist_matrix[psv_j, psv_i] = \
                    scipy.spatial.distance.pdist((cor_matrix[psv_i, mask], cor_matrix[psv_j, mask])) / mask_size
    all_usable = np.array([psv_info.psv_ix for psv_info in psv_infos if psv_info.in_em])
    N_CLUSTERS = 2
    MIN_CLUSTER_SIZE = 5

    psv_ixs = np.where(use_psvs)[0]
    if len(psv_ixs) < MIN_CLUSTER_SIZE * 2:
        return [all_usable]

    condensed_dist = scipy.spatial.distance.squareform(dist_matrix[psv_ixs[:, None], psv_ixs])
    linkage = hierarchy.linkage(condensed_dist, method='complete')

    clusters = hierarchy.fcluster(linkage, 2, criterion='maxclust')
    cluster_sizes = np.bincount(clusters)

    res = [all_usable]
    if len(cluster_sizes) <= 1 or np.max(cluster_sizes) < MIN_CLUSTER_SIZE:
        return res
    added_all = False
    for cluster, count in enumerate(cluster_sizes):
        if count < MIN_CLUSTER_SIZE:
            continue
        res.append(psv_ixs[clusters == cluster])
        if len(clusters) - count < MIN_CLUSTER_SIZE:
            res[0] = None
    if res[0] is None:
        del res[0]
    return res


_BestCluster = collections.namedtuple('_BestCluster',
    'cluster_i n_reliable info_content likelihood psv_f_values sample_gt_probs')

_SMALLEST_COPY_NUM = 2


def write_headers(out, samples, args):
    samples_str = '\t'.join(samples) + '\n'
    out.checked_write('use_psv_sample',
        '# Each element stores two booleans:  first shows if the PSV has good observations at the sample,\n',
        '# Second shows if the sample has the reference copy number at the PSV.\n',
        '# If PSV has less than {} "good" samples, it is not used.\n'.format(args.min_samples),
        'region_group\tpsv\tgood\t' + samples_str)

    max_ref_cn = min(args.max_ref_cn, args.pscn_bound[0])
    col_copies = '\t'.join(map('copy{}'.format, range(1, max_ref_cn // 2 + 1))) + '\n'
    out.checked_write('interm_psv_f_values', 'region_group\tcluster\titeration\tpsv\tinfo_content\t' + col_copies)
    out.checked_write('psv_f_values', 'region_group\tpsv\tn_samples\tuse_in_em\tinfo_content\t' + col_copies)

    out.checked_write('em_likelihoods',
        'region_group\tcluster\titeration\ttime\tlikelihood\tn_reliable\treliable_info\n')
    out.checked_write('em_sample_gts', 'region_group\tcluster\titeration\tgenotype\tprior\t' + samples_str)

    out.checked_write('paralog_cn', 'region_group\tsample\tregion1\tgenotypes\tmarginal_probs\n')
    out.checked_write('gene_conversion',
        '#chrom\tstart\tend\tsample\tregion_group\tmain_gt\treplacement_gt\tqual\tn_psvs\n')


def create_psv_infos(psvs, region_group, n_samples, genome):
    return [_PsvInfo(psv_ix, psv, region_group, n_samples, genome) for psv_ix, psv in enumerate(psvs)]


def find_reliable_psvs(region_group_extra, samples, genome, out, min_samples,
        reliable_threshold, max_agcn):
    # ===== Setting up variables =====
    psvs = region_group_extra.psvs
    n_psvs = len(psvs)
    region_group = region_group_extra.region_group
    group_name = region_group.name
    cn = region_group.cn
    n_copies = cn // 2
    n_samples = len(samples)
    if not n_psvs or n_copies < _SMALLEST_COPY_NUM or n_copies > max_agcn // 2:
        return

    # ===== Selecting a set of PSVs used in the EM algorithm =====
    _select_psv_sample_pairs(region_group_extra, samples, out.use_psv_sample, min_samples)

    psv_infos = region_group_extra.psv_infos
    region_sequence = region_group.region1.get_sequence(genome)
    for psv_info in psv_infos:
        psv_info.check_complicated_pos(region_group.region1, region_sequence)
        psv_info.create_em_psv_gt_probs()
    if not any(psv_info.in_em for psv_info in psv_infos):
        return

    timer_start = perf_counter()
    _calculate_psv_info_content(group_name, psv_infos, min_samples, out.psv_filtering)
    _filter_close_psvs(psv_infos, out.psv_filtering, close_psv_dist=100)
    em_psv_ixs = np.array([psv_info.psv_ix for psv_info in psv_infos if psv_info.in_em])
    if not len(em_psv_ixs):
        return

    common.log('    Searching for reliable PSVs')
    sample_genotypes = variants_.all_gt_counts(n_copies, cn)
    sample_genotypes_str = [','.join(map(str, gt)) for gt in sample_genotypes]
    sample_genotype_priors = _define_sample_gt_priors(n_copies, sample_genotypes)

    # ===== EM iterations, try several clusters =====
    best_cluster = None
    psv_clusters = _cluster_psvs(psv_infos, region_group_extra.psv_read_counts, n_samples)
    for cluster_i, cluster in enumerate(psv_clusters, start=1):
        psv_f_values = np.full((n_psvs, n_copies), 0.5)
        psv_f_values[cluster] = 0.9

        total_lik = -np.inf
        for iteration in range(1, 101):
            if iteration > 1:
                psv_f_values, _ = _m_step(sample_gt_probs, psv_infos, em_psv_ixs, psv_f_values)
            for psv_ix in em_psv_ixs:
                psv_info = psv_infos[psv_ix]
                out.interm_psv_f_values.write('{}\t{}\t{}\t{}:{}\t{:.6f}\t{}\n'.format(group_name, cluster_i,
                    iteration, psv_info.chrom, psv_info.start + 1, psv_info.info_content,
                    '\t'.join(map('{:.6f}'.format, psv_f_values[psv_ix]))))

            old_lik = total_lik
            total_lik, sample_gt_probs = _e_step(psv_infos, psv_f_values, sample_genotype_priors, n_samples)
            for gt_i, gt in enumerate(sample_genotypes_str):
                out.em_sample_gts.write('{}\t{}\t{}\t{}\t{:.3f}\t'.format(group_name, cluster_i, iteration, gt,
                    sample_genotype_priors[gt_i] / common.LOG10))
                out.em_sample_gts.write('\t'.join(map('{:.3f}'.format, sample_gt_probs[:, gt_i] / common.LOG10)))
                out.em_sample_gts.write('\n')

            common.log('        Cluster {}, iteration {:2}, EM likelihood: {:,.3f}'.format(
                cluster_i, iteration, total_lik / common.LOG10))

            with np.errstate(invalid='ignore'):
                curr_reliable = np.where(np.all(psv_f_values >= reliable_threshold, axis=1))[0]

            n_reliable = len(curr_reliable)
            mean_info_content = np.mean([psv_infos[i].info_content for i in curr_reliable]) if n_reliable else 0.0
            out.em_likelihoods.write('{}\t{}\t{}\t{}\t{:.7f}\t{}\t{:.6f}\n'.format(group_name, cluster_i, iteration,
                str(timedelta(seconds=perf_counter() - timer_start))[:-5], total_lik / common.LOG10,
                n_reliable, mean_info_content))
            if total_lik < old_lik + 0.01:
                break
        if best_cluster is None or best_cluster.likelihood < total_lik:
            best_cluster = _BestCluster(cluster_i, n_reliable, mean_info_content, total_lik,
                psv_f_values, sample_gt_probs)

    # ===== Save results from the last cluster =====
    psv_f_values = best_cluster.psv_f_values
    sample_gt_probs = best_cluster.sample_gt_probs
    if len(psv_clusters) > 1:
        common.log('    === Best cluster {}: likelihood {:.3f},  {} reliable PSVs with mean information content {:.3f}'
            .format(best_cluster.cluster_i, best_cluster.likelihood / common.LOG10,
            best_cluster.n_reliable, best_cluster.info_content))
    if best_cluster.n_reliable and best_cluster.info_content < 0.8:
        common.log('WARN: Many reliable PSVs have low information content.')

    discarded_psvs = np.array([psv_info.psv_ix for psv_info in psv_infos
        if not psv_info.in_em and psv_info.n_used_samples > 0])
    if len(discarded_psvs):
        oth_f_values, _ = _m_step(sample_gt_probs, psv_infos, discarded_psvs, np.full((n_psvs, n_copies), 0.5))
        psv_f_values[discarded_psvs, :] = oth_f_values[discarded_psvs, :]
    for psv_info in psv_infos:
        out.psv_f_values.write('{}\t{}:{}\t{}\t{}\t{:.6f}\t{}\n'.format(group_name, psv_info.chrom,
            psv_info.start + 1, psv_info.n_used_samples, 'T' if psv_info.in_em else 'F', psv_info.info_content,
            '\t'.join(map('{:.6f}'.format, psv_f_values[psv_info.psv_ix]))))
    region_group_extra.set_f_values(psv_f_values)


def _single_sample_e_step(sample_id, sample_cn, psv_infos, reliable_psv_ixs):
    """
    Returns sample genotype probabilities.
    """
    n_sample_genotypes = len(psv_infos[reliable_psv_ixs[0]].support_matrix[sample_id])
    res = np.zeros(n_sample_genotypes)
    for psv_ix in reliable_psv_ixs:
        psv_info = psv_infos[psv_ix]
        assert sample_cn == psv_info.sample_cns[sample_id]
        res += psv_info.support_matrix[sample_id]
    return res


def calculate_marginal_probs(genotypes, gt_probs, n_copies, cn):
    """
    Returns
        - marginal probabilities (n_copies x cn + 1), matrix[x, y] represents log probability of the CN x at copy y,
        - paralog-specific CN      (n_copies),
        - paralog-specific CN qual (n_copies).
    """
    marginal_probs = np.full((n_copies, cn + 1), -np.nan)
    gt_probs -= logsumexp(gt_probs)
    for copy in range(n_copies):
        for curr_copy_cn in range(cn + 1):
            ixs = [i for i, gt in enumerate(genotypes) if gt[copy] == curr_copy_cn]
            marginal_probs[copy, curr_copy_cn] = logsumexp(gt_probs[ixs])

    paralog_cn = np.zeros(n_copies, dtype=np.int8)
    paralog_qual = np.zeros(n_copies)
    for copy in range(n_copies):
        best_cn = np.argmax(marginal_probs[copy])
        paralog_cn[copy] = best_cn
        paralog_qual[copy] = common.phred_qual(marginal_probs[copy], best_cn)
    return marginal_probs, paralog_cn, paralog_qual


def paralog_cn_str(paralog_cn, paralog_qual, min_qual_value=5):
    """
    Returns
        - paralog CN: string,
        - paralog qual: tuple of integers,
        - any_known: bool (any of the values over the threshold).
    If paralog quality is less than min_qual_value, corresponding CN is replaced with '?' and quality
    is replaced with 0. Additionally, quality is rounded down to integers.
    """
    paralog_cn_str = []
    new_paralog_qual = []
    any_known = False
    for cn, qual in zip(paralog_cn, paralog_qual):
        if qual < min_qual_value:
            paralog_cn_str.append('?')
            new_paralog_qual.append(0)
        else:
            paralog_cn_str.append(str(cn))
            new_paralog_qual.append(int(qual))
            any_known = True
    return ','.join(paralog_cn_str), tuple(new_paralog_qual), any_known


def _add_paralog_filter(results, filt):
    for res in results:
        res.paralog_filter.add(filt)


class GeneConversionHmm(cn_hmm.HmmModel):
    def __init__(self, best_gt, n_gts, n_observations, stay_prob=0.99, initial_best_prob=0.5):
        n_samples = 1
        super().__init__(n_samples, n_gts, n_observations, max_state_dist=n_gts * 2)

        transition = np.full((n_gts, n_gts), -np.inf)
        stay_log = np.log(stay_prob)
        single_trans = np.log1p(-stay_prob)
        mult_trans = np.log((1 - stay_prob) / (n_gts - 1))
        for state in range(n_gts):
            if state == best_gt:
                transition[state] = mult_trans
            else:
                transition[state, best_gt] = single_trans
            transition[state, state] = stay_log
        self.set_transition(transition)

        initial = np.full(n_gts, np.log((1 - initial_best_prob) / (n_gts - 1)))
        initial[best_gt] = np.log(initial_best_prob)
        self.set_initial(initial)


GeneConversion = collections.namedtuple('GeneConversion', 'start end main_gt replacement_gt qual n_psvs')


def _detect_gene_conversion(sample_id, genotypes_str, sample_gt_probs, psv_infos, semirel_psv_ixs):
    n_psvs = len(semirel_psv_ixs)
    n_genotypes = len(genotypes_str)

    best_gt = np.argmax(sample_gt_probs)
    model = GeneConversionHmm(best_gt, n_genotypes, n_psvs)

    emission_matrix = np.zeros((1, n_genotypes, n_psvs))
    HMM_SAMPLE_ID = 0
    for i, psv_ix in enumerate(semirel_psv_ixs):
        emission_matrix[HMM_SAMPLE_ID, :, i] = psv_infos[psv_ix].support_matrix[sample_id]
    model.set_emission_matrices(emission_matrix)
    prob, states_vec = model.viterbi(HMM_SAMPLE_ID)
    model.run_forward_backward()

    res = []
    for segment in cn_hmm.get_simple_path(states_vec):
        if segment.state == best_gt or segment.end_ix == segment.start_ix + 1:
            continue
        segment0 = cn_hmm.SimpleSegment(segment.start_ix, segment.end_ix, best_gt)
        probs = np.array((
            model.path_likelihood(HMM_SAMPLE_ID, (segment0,)),
            model.path_likelihood(HMM_SAMPLE_ID, (segment,))))
        probs -= logsumexp(probs)
        qual = common.phred_qual(probs, best_ix=1)

        start_psv = psv_infos[semirel_psv_ixs[segment.start_ix]].psv
        end_psv = psv_infos[semirel_psv_ixs[segment.end_ix - 1]].psv
        res.append(GeneConversion(start_psv.start, end_psv.start + len(end_psv.ref),
            genotypes_str[best_gt], genotypes_str[segment.state], qual, segment.end_ix - segment.start_ix))
    return res


def _create_sample_results_from_agcn(sample_id, region_group_extra):
    sample_results = []
    linked_ranges = []
    group_name = region_group_extra.region_group.name

    for sample_const_region in region_group_extra.sample_const_regions[sample_id]:
        entry = ResultEntry(sample_id, sample_const_region)
        entry.info['group'] = group_name
        entry.info.update(sample_const_region.info)

        reg_start = sample_const_region.region1.start
        reg_end = sample_const_region.region1.end
        entry.info['n_windows'] = region_group_extra.group_windows_searcher.overlap_size(reg_start, reg_end)
        entry.info['hmm_windows'] = region_group_extra.hmm_windows_searcher.overlap_size(reg_start, reg_end)

        psv_start_ix, psv_end_ix = region_group_extra.psv_searcher.contained_ixs(reg_start, reg_end)
        entry.info['n_psvs'] = psv_end_ix - psv_start_ix
        entry.info['rel_psvs'] = np.sum(region_group_extra.psv_is_reliable[psv_start_ix : psv_end_ix])

        curr_res_ix = len(sample_results)
        if sample_results and sample_results[-1].pred_cn == entry.pred_cn:
            linked_ranges[-1][1] = curr_res_ix + 1
        else:
            linked_ranges.append([curr_res_ix, curr_res_ix + 1])
        sample_results.append(entry)
    return sample_results, linked_ranges


def _genotypes_str(sample_genotypes, genotypes_str_cache):
    """
    Returns
        - string representations of sample genotypes,
        - string representations of various marginal probabilities in form (0??, 1??, ..., ?0?, ...).
    """
    n_copies = len(sample_genotypes[0])
    sample_cn = sum(sample_genotypes[0])
    if sample_cn in genotypes_str_cache:
        return genotypes_str_cache[sample_cn]

    sample_genotypes_str = [','.join(map(str, gt)) for gt in sample_genotypes]
    marginal_str = []
    if n_copies > 2:
        gt_str = ['?'] * n_copies
        for copy in range(n_copies):
            for curr_copy_cn in range(sample_cn + 1):
                gt_str[copy] = str(curr_copy_cn)
                marginal_str.append(''.join(gt_str))
            gt_str[copy] = '?'
    res = (sample_genotypes_str, marginal_str)
    genotypes_str_cache[sample_cn] = res
    return res


def _single_sample_pscn(sample_id, sample_name, sample_results, linked_ranges, region_group_extra, genome,
        out, genotypes_str_cache, max_genotypes):
    # ====== Defining useful variables ======
    psv_infos = region_group_extra.psv_infos
    n_psvs = len(psv_infos)
    region_group = region_group_extra.region_group
    group_name = region_group.name
    n_copies = region_group.cn // 2
    outp = out.paralog_cn
    region_chrom = region_group.region1.chrom_name(genome)
    psv_searcher = region_group_extra.psv_searcher

    # ====== Calculate psCN for a set of consecutive regions with the same agCN ======
    for link_ix, (start_ix, end_ix) in enumerate(linked_ranges):
        # ===== Check if psCN can be calculated =====
        curr_results = sample_results[start_ix:end_ix]
        if not curr_results[0].sample_const_region.cn_is_known:
            _add_paralog_filter(curr_results, Filter.UncertainCN)
            continue
        psv_ixs = []
        for subresults in curr_results:
            for psv_ix in range(*psv_searcher.contained_ixs(subresults.region1.start, subresults.region1.end)):
                if psv_infos[psv_ix].psv_gt_probs[sample_id] is not None:
                    psv_ixs.append(psv_ix)
        sample_cn = curr_results[0].pred_cn
        if sample_cn == 0:
            continue
        psv_ixs = np.array(psv_ixs)
        if len(psv_ixs) == 0:
            _add_paralog_filter(curr_results, Filter.NoPSVs)
            continue
        reliable_psv_ixs = psv_ixs[region_group_extra.psv_is_reliable[psv_ixs]]
        if len(reliable_psv_ixs) == 0:
            _add_paralog_filter(curr_results, Filter.NoReliable)
            continue
        sample_genotypes = variants_.all_gt_counts(n_copies, sample_cn)
        if len(sample_genotypes) > max_genotypes:
            _add_paralog_filter(curr_results, Filter.HighCN)
            continue

        # ===== Run E-step once again to calculate psCN =====
        sample_gt_probs = _single_sample_e_step(sample_id, sample_cn, psv_infos, reliable_psv_ixs)
        assert len(sample_gt_probs) == len(sample_genotypes)
        marginal_probs, paralog_cn, paralog_qual = calculate_marginal_probs(sample_genotypes, sample_gt_probs,
            n_copies, sample_cn)
        sample_genotypes_str, marginal_str = _genotypes_str(sample_genotypes, genotypes_str_cache)

        # ===== Detect gene conversion =====
        GENE_CONV_QUAL_THRESHOLD = 20
        MIN_SEMIREL_PSVS = 3
        semirel_psv_ixs = psv_ixs[region_group_extra.psv_is_semirel[psv_ixs]]

        if np.all(paralog_qual >= GENE_CONV_QUAL_THRESHOLD) and len(semirel_psv_ixs) >= MIN_SEMIREL_PSVS:
            gene_conv = _detect_gene_conversion(sample_id, sample_genotypes_str, sample_gt_probs,
                psv_infos, semirel_psv_ixs)
            for entry in gene_conv:
                out.gene_conversion.write('{}\t{}\t{}\t'.format(region_chrom, entry.start, entry.end))
                out.gene_conversion.write('{}\t{}\t{}\t{}\t{:.1f}\t{}\n'.format(sample_name, group_name,
                    entry.main_gt, entry.replacement_gt, entry.qual, entry.n_psvs))
        else:
            gene_conv = None

        region1 = Interval(curr_results[0].region1.chrom_id,
            curr_results[0].region1.start, curr_results[-1].region1.end).to_str(genome)
        outp.write('{}\t{}\t{}\t'.format(group_name, sample_name, region1))
        outp.write('  '.join(map('%s=%.1f'.__mod__,
            zip(sample_genotypes_str, np.abs(sample_gt_probs / common.LOG10)))))
        outp.write('\t')
        if n_copies > 2:
            outp.write('  '.join(map('%s=%.1f'.__mod__,
                zip(marginal_str, np.abs(marginal_probs.flatten() / common.LOG10)))))
        else:
            outp.write('*')
        outp.write('\n')

        mean_info = np.mean([psv_infos[psv_ix].info_content for psv_ix in reliable_psv_ixs])
        max_f_value = np.max(np.min(region_group_extra.psv_f_values[reliable_psv_ixs], axis=1))
        if mean_info < 0.9:
            _add_paralog_filter(curr_results, Filter.LowInfoContent)
        if max_f_value < 0.99:
            _add_paralog_filter(curr_results, Filter.NoComplReliable)
        if len(reliable_psv_ixs) < 3:
            _add_paralog_filter(curr_results, Filter.FewReliable)

        info_update = dict(
            n_psvs=len(psv_ixs),
            rel_psvs=len(reliable_psv_ixs),
            semirel_psvs=len(semirel_psv_ixs) - len(reliable_psv_ixs),
            psv_info='{:.3f}'.format(mean_info),
            max_f_value='{:.3f}'.format(max_f_value),
            gene_conv='T' if gene_conv else 'F')
        if end_ix > start_ix + 1:
            info_update['link'] = link_ix

        for res_entry in curr_results:
            res_entry.paralog_cn = paralog_cn
            res_entry.paralog_qual = paralog_qual
            res_entry.info.update(info_update)


def estimate_paralog_cn(region_group_extra, samples, genome, out, *, max_agcn, max_genotypes):
    common.log('    Calculating paralog-specific copy number profiles')
    # ===== Defining useful variables =====
    region_group = region_group_extra.region_group
    n_copies = region_group.cn // 2
    genotypes_str_cache = {}
    has_reliable_psvs = np.sum(region_group_extra.psv_is_reliable) > 0

    results = []
    for sample_id in range(len(samples)):
        sample_results, linked_ranges = _create_sample_results_from_agcn(sample_id, region_group_extra)
        results.extend(sample_results)
        if not has_reliable_psvs:
            if n_copies > max_agcn // 2:
                _add_paralog_filter(sample_results, Filter.HighCN)
            elif n_copies > 1:
                _add_paralog_filter(sample_results, Filter.NoReliable)
            continue

        _single_sample_pscn(sample_id, samples[sample_id], sample_results, linked_ranges, region_group_extra,
            genome, out, genotypes_str_cache, max_genotypes)
    return results


class Filter(Enum):
    Pass = 0
    HighCN = 11

    NoReliable = 20
    FewReliable = 21
    NoComplReliable = 22
    LowInfoContent = 23
    UncertainCN = 24
    NoPSVs = 25

    def __str__(self):
        if self == Filter.Pass:
            return 'PASS'
        if self == Filter.HighCN:
            return 'HighCN'
        if self == Filter.NoReliable:
            return 'NoReliable'
        if self == Filter.FewReliable:
            return 'FewReliable'
        if self == Filter.NoComplReliable:
            return 'NoComplReliable'
        if self == Filter.LowInfoContent:
            return 'LowInfoCont'
        if self == Filter.UncertainCN:
            return 'UncertainCN'
        if self == Filter.NoPSVs:
            return 'NoPSVs'

    @classmethod
    def from_str(cls, s):
        if s == 'PASS':
            return Filter.Pass
        if s == 'HighCN':
            return Filter.HighCN
        if s == 'NoReliable':
            return Filter.NoReliable
        if s == 'FewReliable':
            return Filter.FewReliable
        if s == 'NoComplReliable':
            return Filter.NoComplReliable
        if s == 'LowInfoCont':
            return Filter.LowInfoContent
        if s == 'UncertainCN':
            return Filter.UncertainCN
        if s == 'NoPSVs':
            return Filter.NoPSVs


class Filters:
    def __init__(self):
        self.filters = None

    def add(self, filt):
        if self.filters is None:
            self.filters = set()
        self.filters.add(filt)

    def to_str(self, value_defined):
        if not self.filters:
            return 'PASS' if value_defined else '*'
        return ';'.join(map(str, self.filters))

    def to_tuple(self):
        if not self.filters:
            return ('PASS',)
        return tuple(map(str, self.filters))

    def __len__(self):
        return 0 if self.filters is None else len(self.filters)

    def __bool__(self):
        return bool(self.filters)

    def copy(self):
        res = Filters()
        if self.filters:
            res.filters = self.filters.copy()
        return res


class ResultEntry:
    def __init__(self, sample_id, sample_const_region):
        self.sample_id = sample_id
        self.sample_const_region = sample_const_region
        self.n_copies = sample_const_region.cn // 2

        self.agcn_filter = Filters()
        self.paralog_filter = Filters()
        self.paralog_cn = None
        self.paralog_qual = None
        self.info = {}

    @property
    def region1(self):
        return self.sample_const_region.region1

    @property
    def pred_cn(self):
        return self.sample_const_region.pred_cn

    def copy_num_to_str(self):
        if self.pred_cn is None:
            return (self.agcn_filter.to_str(False), '?', '*')
        return (self.agcn_filter.to_str(True), self.sample_const_region.pred_cn_str,
            '{:.2f}'.format(self.sample_const_region.qual))

    def paralog_to_str(self):
        if self.paralog_cn is None:
            if self.n_copies == 1:
                self.paralog_cn = (self.sample_const_region.pred_cn_str,)
                self.paralog_qual = (self.sample_const_region.qual,)
            elif self.pred_cn == 0:
                self.paralog_cn = (0,) * self.n_copies
                self.paralog_qual = (self.sample_const_region.qual,) * self.n_copies
        paralog_filter = self.paralog_filter.to_str(self.paralog_cn is not None)

        if self.paralog_cn is None:
            return paralog_filter, ','.join('?' * self.n_copies), ','.join('0' * self.n_copies)

        pscn, pscn_qual, _ = paralog_cn_str(self.paralog_cn, self.paralog_qual)
        return paralog_filter, pscn, ','.join(map(str, pscn_qual))

    def to_str(self, region_name, genome, samples):
        res = '{}\t{}\t{}\t'.format(self.sample_const_region.region1.to_bed(genome), region_name,
            samples[self.sample_id])
        res += '{}\t{}\t{}\t'.format(*self.copy_num_to_str())
        res += '{}\t{}\t{}\t'.format(*self.paralog_to_str())

        if self.info:
            res += ';'.join(map('%s=%s'.__mod__, self.info.items()))
        else:
            res += '*'
        res += '\t'
        res += self.sample_const_region.regions2_str(genome)
        return res

    def __lt__(self, other):
        return self.region1.__lt__(other.region1)

    @classmethod
    def from_dict(cls, row, genome, samples):
        sample_id = samples.id(row['sample'])
        region1 = Interval(genome.chrom_id(row['chrom']), int(row['start']), int(row['end']))

        hom_regions = row['homologous_regions']
        regions2 = None
        if hom_regions != '*':
            regions2 = []
            for entry in hom_regions.split(','):
                regions2.append(Interval.parse_with_strand(entry, genome))
        pred_cn_str = row['agCN']
        pred_cn = int(pred_cn_str) if pred_cn_str.isdigit() else None
        pred_cn_qual = float(row['agCN_qual'])
        sample_const_region = cn_tools.CopyNumPrediction(region1, regions2, pred_cn, pred_cn_str, pred_cn_qual)

        res = cls(sample_id, sample_const_region)
        pscn = row['psCN'].split(',')
        pscn_qual = row['psCN_qual'].split(',')
        for i in range(len(pscn)):
            pscn[i] = None if pscn[i] == '?' else int(pscn[i])
            pscn_qual[i] = int(pscn_qual[i])
        res.paralog_cn = pscn
        res.paralog_qual = pscn_qual

        for entry in row['agCN_filter'].split(';'):
            res.agcn_filter.add(Filter.from_str(entry))
        for entry in row['psCN_filter'].split(';'):
            res.paralog_filter.add(Filter.from_str(entry))

        info = res.info
        row_info = row['info']
        if row_info != '*':
            for entry in row_info.split(';'):
                key, val = entry.split('=')
                info[key] = val
        return res


def _process_sample_entries(start, end, by_sample, searchers):
    # Order of ResultEntry.copy_num_to_str() + entry.paralog_to_str()
    CNF = 0
    CN = 1
    CNQ = 2
    PCNF = 3
    PCN = 4
    PCNQ = 5

    all_copy_nums = collections.Counter()
    all_paralog_cns = collections.Counter()
    sample_results = ''
    for sample_entries, searcher in zip(by_sample, searchers):
        start_ix, end_ix = searcher.overlap_ixs(start, end)
        sample_results += '\t'
        if end_ix <= start_ix:
            sample_results += '*'
            continue

        info_sets = [set() for _ in range(6)]
        coverage = 0
        # Convert to int because islice does not work with numpy.int.
        for entry in itertools.islice(sample_entries, int(start_ix), int(end_ix)):
            curr_info = entry.copy_num_to_str() + entry.paralog_to_str()
            for i, value in enumerate(curr_info):
                info_sets[i].add(value)
            curr_cov = min(end, entry.region1.end) - max(start, entry.region1.start)

            all_copy_nums[curr_info[CN]] += curr_cov
            all_paralog_cns[curr_info[PCN]] += curr_cov
            coverage += curr_cov

        coverage = 100 * coverage / (end - start)
        if len(info_sets[1]) != 1:
            sample_results += '! ! ! ! ! ! {:.1f}'.format(coverage)
            continue
        # Replace with constants.
        for i in (CN, CNF, CNQ, PCN, PCNF, PCNQ):
            if len(info_sets[i]) == 1:
                sample_results += '{} '.format(info_sets[i].pop())
            else:
                sample_results += '! '
        sample_results += '{:.1f}'.format(coverage)
    return all_copy_nums, all_paralog_cns, sample_results


def write_matrix_summary(results, region_name, genome, samples, out):
    if not results:
        return

    out.write('## {}\n'.format(' '.join(sys.argv)))
    out.write('## {} {}\n'.format(__pkg_name__, __version__))
    out.write('## For each sample 7 values are stored: agCN, agCN_filter, agCN_qual; '
        'psCN, psCN_filter, psCN_qual; overlap.\n')
    out.write('## overlap - percentage of the region covered by the sample entries.\n')
    out.write('## Entries for sample can contain "!", that means that several entries '
        'cover the region and have different values.\n')
    out.write('#chrom\tstart\tend\tlocus\trefCN\tagCN_freq\tpsCN_freq\tinfo\thomologous_regions\t')
    out.write('\t'.join(samples))
    out.write('\n')

    by_sample = [[] for _ in range(len(samples))]
    unique_events = collections.defaultdict(list)
    for entry in results:
        by_sample[entry.sample_id].append(entry)
        key = (entry.region1.start, entry.region1.end)
        unique_events[key].append(entry)

    searchers = []
    for sample_entries in by_sample:
        searchers.append(itree.NonOverlTree(sample_entries, itree.region1_start, itree.region1_end))

    for (start, end) in sorted(unique_events.keys()):
        templates = unique_events[(start, end)]
        template = templates[0]
        out.write('{}\t{}\t{}\t'.format(template.region1.to_bed(genome), region_name,
            template.sample_const_region.cn))

        all_copy_nums, all_paralog_cns, sample_results = _process_sample_entries(start, end, by_sample, searchers)
        copy_num_freqs = []
        for copy_num, freq in all_copy_nums.items():
            if copy_num.isdigit():
                # Store sorting key, copy number, and frequency.
                copy_num_freqs.append((int(copy_num), copy_num, freq))
            elif copy_num.startswith('<'):
                copy_num_freqs.append((-1, copy_num, freq))
            elif copy_num.startswith('>'):
                copy_num_freqs.append((1000, copy_num, freq))
            # else: ignore
        copy_num_freqs.sort()
        copy_num_freq_sum = sum(map(operator.itemgetter(2), copy_num_freqs))
        out.write(' '.join('{}={:.5g}'.format(copy_num, freq / copy_num_freq_sum)
            for _, copy_num, freq in copy_num_freqs))
        out.write('\t')

        paralog_freq_sum = sum(all_paralog_cns.values())
        out.write(' '.join('{}={:.5g}'.format(paralog, freq / paralog_freq_sum)
            for paralog, freq in all_paralog_cns.most_common()))

        info = 'len={:.1f}kb;samples={}:{}{}'.format((end - start) / 1000, len(templates),
            ','.join(samples[entry.sample_id] for entry in itertools.islice(templates, 0, 10)),
            ',...' if len(templates) > 10 else '')
        out.write('\t{}\t'.format(info))
        out.write(template.sample_const_region.regions2_str(genome))

        out.write(sample_results)
        out.write('\n')
