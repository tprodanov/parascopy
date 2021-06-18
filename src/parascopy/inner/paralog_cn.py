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
from functools import lru_cache
import itertools
from time import perf_counter
from datetime import timedelta
from enum import Enum

from . import common
from . import variants as variants_
from . import polynomials
from . import cn_tools
from . import cn_hmm
from .genome import Interval
from .. import __pkg_name__, __version__


_LOG10 = np.log(10)

@lru_cache(maxsize=None)
def _all_f_combinations(n_copies, cn):
    f_combinations = [(0,) * n_copies]
    for i in range(1, cn + 1):
        f_combinations.extend(variants_.all_gt_counts(n_copies, i))
    f_comb_ixs = { gt: i for i, gt in enumerate(f_combinations) }
    return np.array(f_combinations), f_comb_ixs


@lru_cache(maxsize=None)
class _PrecomputedData:
    def __init__(self, alleles, cn):
        self.n_copies = len(alleles)
        self.n_alleles = max(alleles) + 1
        self.cn = cn

        self.sample_genotypes = variants_.all_gt_counts(self.n_copies, cn)
        self.psv_genotypes = variants_.all_gt_counts(self.n_alleles, cn)
        self.n_psv_genotypes = len(self.psv_genotypes)
        self.f_combinations, f_comb_ixs = _all_f_combinations(self.n_copies, cn)

        self.poly_matrices = []
        self.f_ix_converts = []
        for sample_gt in self.sample_genotypes:
            f_powers, x_powers, res_poly = polynomials.multiply_polynomials_f_values(alleles, sample_gt)
            f_modulo = x_powers[0]

            poly_matrix = np.zeros((self.n_psv_genotypes, f_modulo))
            for i, psv_gt in enumerate(self.psv_genotypes):
                power = sum((x_powers[allele] - f_modulo) * count for allele, count in enumerate(psv_gt))
                res_poly.get_slice(power, poly_matrix[i])
            f_ix_convert = np.fromiter(
                (f_comb_ixs.get(gt, 0) for gt in itertools.product(*(range(count + 1) for count in sample_gt))),
                dtype=np.int16)
            self.poly_matrices.append(poly_matrix)
            self.f_ix_converts.append(f_ix_convert)

    def f_power_combinations(self, f_values):
        f_pow = f_values[:, np.newaxis] ** range(0, self.cn + 1)
        f_pow_comb = f_pow[range(self.n_copies), self.f_combinations]
        return np.product(f_pow_comb, axis=1)

    def eval_poly_matrices(self, f_power_combinations):
        res = []
        with np.errstate(divide='ignore'):
            for poly_matrix, f_ix_convert in zip(self.poly_matrices, self.f_ix_converts):
                res_matrix = np.sum(poly_matrix * f_power_combinations[f_ix_convert], axis=1)
                res.append(np.log(np.clip(res_matrix, 0, 1)))
        return res


def _precompute_probabilities(psv_infos, psv_counts, cn, n_samples):
    for psv_info in psv_infos:
        if not psv_info.n_used_samples:
            continue

        precomp_data = _PrecomputedData(psv_info.allele_corresp, cn)
        genotypes = precomp_data.psv_genotypes
        probs_matrix = np.full((n_samples, precomp_data.n_psv_genotypes), np.nan)

        for sample_id in np.where(psv_info.use_samples)[0]:
            counts = psv_counts[psv_info.psv_ix][sample_id]
            _, probs = variants_.genotype_likelihoods(cn, counts.allele_counts, gt_counts=precomp_data.psv_genotypes)
            probs -= logsumexp(probs)
            probs_matrix[sample_id] = probs
        psv_info.set_psv_gt_probs(precomp_data, probs_matrix)


_BETA_DISTR_A = 5.0
_BETA_DISTR_B = 1.0

def _get_f_prior(f_values):
    EPSILON = 1e-6
    max_f_value = np.max(f_values)
    beta_cdfs = stats.beta.logcdf((max_f_value, max_f_value - EPSILON), _BETA_DISTR_A, _BETA_DISTR_B)
    return logsumexp(beta_cdfs, b=(1, -1))


def _e_step(psv_infos, psv_f_values, sample_genotypes, genotype_priors, n_samples):
    """
    Returns
        - matrix of P(sample_genotype | allele_counts, psvs), size = (n_samples, n_genotypes),
        - total ln likelihood,
        - 3d matrix (n_psvs x n_samples x n_genotypes) with individual (psv,sample,genotype) probabilities.
    """
    n_genotypes = len(sample_genotypes)
    n_psvs = len(psv_infos)
    res = np.zeros((n_samples, n_genotypes))
    usable_samples = np.zeros(n_samples, dtype=np.bool)
    individual_probs = np.full((n_psvs, n_samples, n_genotypes), np.nan)
    total_f_prior = 0

    for psv_info in psv_infos:
        if not psv_info.in_em:
            continue
        sample_ixs = np.where(psv_info.use_samples)[0]
        usable_samples[sample_ixs] = True

        precomp_data = psv_info.precomp_data
        psv_gt_probs = psv_info.psv_gt_probs
        psv_gt_probs = psv_gt_probs[sample_ixs]
        exponent = psv_info.info_content

        psv_ix = psv_info.psv_ix
        f_values = psv_f_values[psv_ix]
        total_f_prior += _get_f_prior(f_values)
        f_pow_combs = precomp_data.f_power_combinations(f_values)
        all_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)

        for sample_gt_ix, psv_gt_coefs in enumerate(all_psv_gt_coefs):
            curr_probs = exponent * logsumexp(psv_gt_coefs + psv_gt_probs, axis=1)
            individual_probs[psv_ix, sample_ixs, sample_gt_ix] = curr_probs
            res[sample_ixs, sample_gt_ix] += curr_probs

    res += genotype_priors[np.newaxis, :]
    res[~usable_samples] = np.nan
    row_sums = logsumexp(res, axis=1)
    total_lik = np.sum(row_sums[usable_samples]) + total_f_prior
    res -= row_sums[:, np.newaxis]
    return res, total_lik, individual_probs


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
        sample_ixs = np.where(psv_info.use_samples)[0]
        assert len(sample_ixs) > 0

        sample_gt_probs_subset = sample_gt_probs_regular[sample_ixs]
        precomp_data = psv_info.precomp_data
        psv_gt_probs = psv_info.psv_gt_probs
        psv_gt_probs = psv_gt_probs[sample_ixs]
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
            for i, entry2 in enumerate(pos2, 1):
                s += '    {} {}{}\n'.format(i, entry2, '  !!!' if entry == entry2 else '')
            s += 'With regions:\n    '
            s += cn_tools.regions2_str(regions2, genome, use_comma=True, sep='\n    ')
            common.log(s)
            raise RuntimeError('Cannot match PSV alleles')
    return tuple(res)


def _select_psv_sample_pairs(group, psv_infos, samples, psv_counts, sample_reliable_regions, outp, min_samples):
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
            counts = psv_counts[psv_ix][sample_id]
            good_obs = not counts.skip
            is_ref = sample_region is not None \
                and sample_region.start <= psv_info.start and psv_info.end <= sample_region.end
            info_str += '\t{}{}'.format('+' if good_obs else '-', '+' if is_ref else '-')
            use_samples[sample_id] = good_obs and is_ref

        psv_info.set_use_samples(use_samples, min_samples)
        outp.write('{}\t{}:{}\t{}{}\n'.format(group_name, psv_info.chrom, psv_info.start + 1,
            psv_info.n_used_samples, info_str))


class _PsvInfo:
    def __init__(self, psv_ix, psv, region_group, genome):
        self.psv_ix = psv_ix
        self.chrom = psv.chrom
        self.start = psv.start
        self.end = psv.start + len(psv.ref)
        self.is_indel = len(set(map(len, psv.alleles))) != 1
        self.allele_corresp = _match_psv_alleles(psv, region_group, genome)

        self.info_content = np.nan
        self.use_samples = None
        self.n_used_samples = 0
        self.in_em = False

        self.precomp_data = None
        self.psv_gt_probs = None

    def set_use_samples(self, use_samples, min_samples):
        self.use_samples = use_samples
        self.n_used_samples = np.sum(use_samples)
        self.in_em = self.n_used_samples >= min_samples

    def set_psv_gt_probs(self, precomp_data, psv_gt_probs):
        self.precomp_data = precomp_data
        self.psv_gt_probs = psv_gt_probs

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


def _calculate_psv_info_content(group_name, psv_infos, min_samples, outp):
    outp.write('Region group {}\n'.format(group_name))
    for psv_info in psv_infos:
        if not psv_info.n_used_samples:
            outp.write('{}   no applicable samples, skipping.\n'.format(psv_info))
            continue

        precomp_data = psv_info.precomp_data
        psv_gt_probs = psv_info.psv_gt_probs
        n_alleles = precomp_data.n_alleles
        gt_mult = np.fromiter((n_alleles - gt.count(0) - 1 for gt in precomp_data.psv_genotypes),
            np.int16, len(precomp_data.psv_genotypes))

        sample_ixs = np.where(psv_info.use_samples)[0]
        sum_info_content = 0.0
        inform_samples = 0
        for sample_id in sample_ixs:
            curr_info_content = np.sum(np.exp(psv_gt_probs[sample_id]) * gt_mult)
            sum_info_content += curr_info_content
            if curr_info_content >= 0.8:
                inform_samples += 1
        outp.write('{}   informative: {}/{} samples. '.format(psv_info, inform_samples, len(sample_ixs)))

        psv_info.info_content = sum_info_content / len(sample_ixs) / (n_alleles - 1)
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
        for sample_id in np.where(psv_info.use_samples)[0]:
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
    'cluster_i n_reliable info_content likelihood psv_f_values sample_gt_probs individual_probs')

_SMALLEST_COPY_NUM = 2


def write_headers(out, samples, args):
    samples_str = '\t'.join(samples) + '\n'
    out.checked_write('use_psv_sample',
        '# Each element stores two booleans:  first shows if the PSV has good observations at the sample,\n',
        '# Second shows if the sample has the reference copy number at the PSV.\n',
        '# If PSV has less than {} "good" samples, it is not used.\n'.format(args.min_samples),
        'region_group\tpsv\tgood\t' + samples_str)

    max_copy_num = min(args.max_copy_num, args.copy_num_bound[0])
    col_copies = '\t'.join(map('copy{}'.format, range(1, max_copy_num // 2 + 1))) + '\n'
    out.checked_write('interm_psv_f_values', 'region_group\tcluster\titeration\tpsv\tinfo_content\t' + col_copies)
    out.checked_write('psv_f_values', 'region_group\tpsv\tn_samples\tuse_in_em\tinfo_content\t' + col_copies)

    out.checked_write('em_likelihoods',
        'region_group\tcluster\titeration\ttime\tlikelihood\tn_reliable\treliable_info\n')
    out.checked_write('em_sample_gts', 'region_group\tcluster\titeration\tgenotype\tprior\t' + samples_str)
    out.checked_write('em_sample_psv_support', 'region_group\tpsv\tgenotype\t' + samples_str)
    out.checked_write('em_sample_psv_gts', 'region_group\tpsv\t' + samples_str)

    out.checked_write('paralog_cn', 'region_group\tsample\tregion1\tgenotypes\tmarginal_probs\n')
    out.checked_write('gene_conversion',
        '#chrom\tstart\tend\tsample\tregion_group\tmain_gt\treplacement_gt\tqual\tn_psvs\n')


def create_psv_infos(psvs, region_group, genome):
    return [_PsvInfo(psv_ix, psv, region_group.regions2, genome) for psv_ix, psv in enumerate(psvs)]


def find_reliable_psvs(region_group_extra, all_psv_counts, samples, genome, out, min_samples,
        reliable_threshold, paralog_max_copy_num):
    psvs = region_group_extra.psvs
    n_psvs = len(psvs)
    region_group = region_group_extra.region_group
    group_name = region_group.name
    cn = region_group.cn
    n_copies = cn // 2
    n_samples = len(samples)
    if not n_psvs or not (_SMALLEST_COPY_NUM <= n_copies <= paralog_max_copy_num // 2):
        return

    psv_counts = [all_psv_counts[i] for i in region_group.psv_ixs]
    psv_infos = region_group_extra.psv_infos

    _select_psv_sample_pairs(region_group, psv_infos, samples, psv_counts, region_group_extra.sample_reliable_regions,
        out.use_psv_sample, min_samples)

    region_sequence = region_group.region1.get_sequence(genome)
    for psv_info in psv_infos:
        psv_info.check_complicated_pos(region_group.region1, region_sequence)
    if not any(psv_info.in_em for psv_info in psv_infos):
        return

    timer_start = perf_counter()
    _precompute_probabilities(psv_infos, psv_counts, cn, n_samples)
    _calculate_psv_info_content(group_name, psv_infos, min_samples, out.psv_filtering)
    _filter_close_psvs(psv_infos, out.psv_filtering, close_psv_dist=100)
    em_psv_ixs = np.array([psv_info.psv_ix for psv_info in psv_infos if psv_info.in_em])
    if not len(em_psv_ixs):
        return

    common.log('    Searching for reliable PSVs')
    sample_genotypes = variants_.all_gt_counts(n_copies, cn)
    sample_genotypes_str = [','.join(map(str, gt)) for gt in sample_genotypes]
    sample_genotype_priors = _define_sample_gt_priors(n_copies, sample_genotypes)

    _, _, individual_probs = _e_step(psv_infos, np.full((n_psvs, n_copies), 0.99),
        sample_genotypes, sample_genotype_priors, n_samples)
    for psv_info in psv_infos:
        if not psv_info.in_em:
            continue
        out.em_sample_psv_gts.write('{}\t{}:{}'.format(group_name, psv_info.chrom, psv_info.start + 1))
        for sample_id, sample_row in enumerate(individual_probs[psv_info.psv_ix]):
            if psv_info.use_samples[sample_id]:
                gt_ix = np.argmax(sample_row)
                out.em_sample_psv_gts.write('\t{}'.format(sample_genotypes_str[gt_ix]))
            else:
                out.em_sample_psv_gts.write('\tNA')
        out.em_sample_psv_gts.write('\n')

    best_cluster = None
    psv_clusters = _cluster_psvs(psv_infos, psv_counts, n_samples)
    for cluster_i, cluster in enumerate(psv_clusters, 1):
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
            sample_gt_probs, total_lik, individual_probs = _e_step(psv_infos, psv_f_values,
                sample_genotypes, sample_genotype_priors, n_samples)
            for gt_i, gt in enumerate(sample_genotypes_str):
                out.em_sample_gts.write('{}\t{}\t{}\t{}\t{:.3f}\t'.format(group_name, cluster_i, iteration, gt,
                    sample_genotype_priors[gt_i] / _LOG10))
                out.em_sample_gts.write('\t'.join(map('{:.3f}'.format, sample_gt_probs[:, gt_i] / _LOG10)))
                out.em_sample_gts.write('\n')

            common.log('        Cluster {}, iteration {:2}, EM likelihood: {:,.3f}'.format(
                cluster_i, iteration, total_lik / _LOG10))

            with np.errstate(invalid='ignore'):
                curr_reliable = np.where(np.all(psv_f_values >= reliable_threshold, axis=1))[0]

            n_reliable = len(curr_reliable)
            mean_info_content = np.mean([psv_infos[i].info_content for i in curr_reliable]) if n_reliable else 0.0
            out.em_likelihoods.write('{}\t{}\t{}\t{}\t{:.7f}\t{}\t{:.6f}\n'.format(group_name, cluster_i, iteration,
                str(timedelta(seconds=perf_counter() - timer_start))[:-5], total_lik / _LOG10,
                n_reliable, mean_info_content))
            if total_lik < old_lik + 0.01:
                break
        if best_cluster is None or best_cluster.likelihood < total_lik:
            best_cluster = _BestCluster(cluster_i, n_reliable, mean_info_content, total_lik,
                psv_f_values, sample_gt_probs, individual_probs)

    psv_f_values = best_cluster.psv_f_values
    sample_gt_probs = best_cluster.sample_gt_probs
    individual_probs = best_cluster.individual_probs
    if len(psv_clusters) > 1:
        common.log('    === Best cluster {}: likelihood {:.3f},  {} reliable PSVs with mean information content {:.3f}'
            .format(best_cluster.cluster_i, best_cluster.likelihood / _LOG10,
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

    for psv_ix, psv in enumerate(psvs):
        prefix = '{}\t{}:{}\t'.format(group_name, psv.chrom, psv.pos)
        psv_ind_probs = individual_probs[psv_ix]
        for gt_ix, gt in enumerate(sample_genotypes_str):
            # out.em_sample_psv_support.write('region_group\tpsv\tgenotype\t{}\n'.format(samples_str))
            out.em_sample_psv_support.write('{}{}\t'.format(prefix, gt))
            out.em_sample_psv_support.write('\t'.join(map('{:.4f}'.format, psv_ind_probs[:, gt_ix] / _LOG10)))
            out.em_sample_psv_support.write('\n')


def _precompute_probabilities_single_sample(sample_id, sample_cn, reliable_psv_ixs, psv_counts, psv_infos):
    res = []
    for psv_ix in reliable_psv_ixs:
        psv_info = psv_infos[psv_ix]
        if psv_info.use_samples[sample_id]:
            precomp_data = psv_info.precomp_data
            psv_gt_probs = psv_info.psv_gt_probs[sample_id]
            assert precomp_data.cn == sample_cn
            res.append((precomp_data, psv_gt_probs))
            continue

        counts = psv_counts[psv_ix][sample_id]
        if counts.skip:
            res.append(None)
            continue

        precomp_data = _PrecomputedData(psv_info.allele_corresp, sample_cn)
        psv_genotypes = precomp_data.psv_genotypes
        _, probs = variants_.genotype_likelihoods(sample_cn, counts.allele_counts, gt_counts=psv_genotypes)
        probs -= logsumexp(probs)
        res.append((precomp_data, probs))
    return res


def _single_sample_e_step(sample_genotypes, psv_f_values, psv_exponents, reliable_psv_ixs, psvs,
        sample_psv_gt_probs, f_pow_combs_cache):
    """
    Returns
        - sample_genotype_probabilities (n_sample_genotypes),
        - used_psvs,
        - support matrix (len(used_psvs), n_sample_genotypes).
    """
    n_genotypes = len(sample_genotypes)
    n_psvs = len(reliable_psv_ixs)
    res = np.zeros(n_genotypes)

    used_psvs = []
    support_matrix = []
    for psv_ix, entry in zip(reliable_psv_ixs, sample_psv_gt_probs):
        if entry is None:
            continue
        precomp_data, psv_gt_probs = entry

        exponent = psv_exponents[psv_ix]
        f_pow_combs = f_pow_combs_cache[psv_ix].get(precomp_data.cn)
        if f_pow_combs is None:
            f_pow_combs = precomp_data.f_power_combinations(psv_f_values[psv_ix])
            f_pow_combs_cache[psv_ix][precomp_data.cn] = f_pow_combs

        all_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)
        support_row = np.zeros(n_genotypes)
        for sample_gt_ix, psv_gt_coefs in enumerate(all_psv_gt_coefs):
            support_row[sample_gt_ix] = exponent * logsumexp(psv_gt_coefs + psv_gt_probs)
        support_matrix.append(support_row)
        used_psvs.append(psvs[psv_ix])
        res += support_row
    return res, used_psvs, np.array(support_matrix)


def _calc_marginal_probs(genotypes, gt_probs, n_copies, cn):
    """
    Returns matrix (n_copies x cn + 1), where matrix[x, y] represents log probability of the CN x at the copy y.
    """
    res = np.full((n_copies, cn + 1), -np.nan)
    gt_probs -= logsumexp(gt_probs)
    for copy in range(n_copies):
        for curr_copy_cn in range(cn + 1):
            ixs = [i for i, gt in enumerate(genotypes) if gt[copy] == curr_copy_cn]
            res[copy, curr_copy_cn] = logsumexp(gt_probs[ixs])
    return res


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


def _detect_gene_conversion(genotypes_str, sample_gt_probs, support_psvs, support_matrix):
    """
    Support_matrix: matrix (n_psvs, n_genotypes).
    """
    n_psvs = len(support_psvs)
    n_genotypes = len(genotypes_str)
    if n_psvs <= 2:
        return ()

    best_gt = np.argmax(sample_gt_probs)
    model = GeneConversionHmm(best_gt, n_genotypes, n_psvs)

    emission_matrix = support_matrix.T
    model.set_emission_matrices(emission_matrix[np.newaxis, :, :])
    sample_id = 0
    prob, states_vec = model.viterbi(sample_id)

    res = []
    for segment in cn_hmm.get_simple_path(states_vec):
        if segment.state == best_gt or segment.end_ix == segment.start_ix + 1:
            continue
        segment0 = cn_hmm.SimpleSegment(segment.start_ix, segment.end_ix, best_gt)
        prob0 = model.path_likelihood(sample_id, (segment0,), states_vec)
        prob1 = model.path_likelihood(sample_id, (segment,), states_vec)
        prob0 -= logsumexp((prob0, prob1))
        qual = -10 * prob0 / _LOG10

        start_psv = support_psvs[segment.start_ix]
        end_psv = support_psvs[segment.end_ix - 1]
        res.append(GeneConversion(start_psv.start, end_psv.start + len(end_psv.ref),
            genotypes_str[best_gt], genotypes_str[segment.state], qual, segment.end_ix - segment.start_ix))
    return res


def estimate_paralog_cn(region_group_extra, all_psv_counts, samples, genome, out,
        reliable_thresholds, paralog_max_copy_num, max_genotypes):
    common.log('    Calculating paralog copy number profiles')
    gene_conv_threshold, reliable_threshold = reliable_thresholds
    n_samples = len(samples)
    psvs = region_group_extra.psvs
    psv_infos = region_group_extra.psv_infos
    n_psvs = len(psvs)
    region_group = region_group_extra.region_group
    group_name = region_group.name
    cn = region_group.cn
    n_copies = cn // 2
    psv_counts = [all_psv_counts[i] for i in region_group.psv_ixs]
    outp = out.paralog_cn
    region_chrom = region_group.region1.chrom_name(genome)

    if region_group_extra.has_f_values:
        with np.errstate(invalid='ignore'):
            rel_psvs = np.all(region_group_extra.psv_f_values >= reliable_threshold, axis=1) \
                & [psv_info.in_em for psv_info in psv_infos]
            gene_conv_psvs = np.all(region_group_extra.psv_f_values >= gene_conv_threshold, axis=1) \
                & [psv_info.in_em for psv_info in psv_infos]
        psv_info_content = np.array([psv_info.info_content for psv_info in region_group_extra.psv_infos])
    else:
        rel_psvs = np.zeros(n_psvs, dtype=np.bool)
        gene_conv_psvs = np.zeros(n_psvs, dtype=np.bool)
        psv_info_content = None
    n_rel_psvs = np.sum(rel_psvs)
    psv_finder = common.NonOverlappingSet.from_variants(psvs)

    if n_rel_psvs:
        f_pow_combs_cache = [{} if rel_psvs[psv_ix] else None for psv_ix in range(n_psvs)]
        genotypes_str_cache = {}
    else:
        f_pow_combs_cache = genotypes_str_cache = None

    results = []
    for sample_id in range(n_samples):
        sample_results = []
        linked_ranges = []

        for sample_const_region in region_group_extra.sample_const_regions[sample_id]:
            entry = ResultEntry(sample_id, sample_const_region)
            entry.info['group'] = group_name
            entry.info['region_ix'] = sample_const_region.region_ix

            reg_start = sample_const_region.region1.start
            reg_end = sample_const_region.region1.end
            a, b = region_group_extra.group_windows_searcher.select(reg_start, reg_end)
            entry.info['n_windows'] = b - a
            a, b = region_group_extra.viterbi_windows_searcher.select(reg_start, reg_end)
            entry.info['hmm_windows'] = b - a

            psv_start_ix, psv_end_ix = psv_finder.select(reg_start, reg_end)
            entry.info['n_psvs'] = psv_end_ix - psv_start_ix
            curr_rel = rel_psvs[psv_start_ix : psv_end_ix]
            curr_n_rel = np.sum(curr_rel)
            entry.info['rel_psvs'] = curr_n_rel
            if curr_n_rel:
                entry.info['psv_info'] = '{:.3f}'.format(np.mean(psv_info_content[psv_start_ix : psv_end_ix][curr_rel]))

            curr_res_ix = len(sample_results)
            if sample_results and sample_results[-1].pred_cn == entry.pred_cn:
                linked_ranges[-1][1] = curr_res_ix + 1
            else:
                linked_ranges.append([curr_res_ix, curr_res_ix + 1])
            sample_results.append(entry)

        results.extend(sample_results)
        if not n_rel_psvs:
            if n_copies == 1:
                pass
            elif n_copies > paralog_max_copy_num // 2:
                _add_paralog_filter(sample_results, Filter.HighCN)
            else:
                _add_paralog_filter(sample_results, Filter.NoReliable)
            continue

        for link_ix, (start_ix, end_ix) in enumerate(linked_ranges):
            curr_results = sample_results[start_ix:end_ix]
            psv_ixs = []
            for subresults in curr_results:
                psv_ixs.extend(range(*psv_finder.select(subresults.region1.start, subresults.region1.end)))
            psv_ixs = np.array(psv_ixs)
            if len(psv_ixs) == 0:
                continue
            curr_rel_psvs = psv_ixs[rel_psvs[psv_ixs]]
            if len(curr_rel_psvs) == 0:
                _add_paralog_filter(curr_results, Filter.NoReliable)
                continue

            if not sample_results[start_ix].sample_const_region.cn_is_known:
                _add_paralog_filter(curr_results, Filter.UncertainCN)
                continue

            sample_cn = sample_results[start_ix].pred_cn
            if sample_cn == 0:
                continue
            sample_genotypes = variants_.all_gt_counts(n_copies, sample_cn)
            if len(sample_genotypes) > max_genotypes:
                _add_paralog_filter(curr_results, Filter.HighCN)
                continue

            curr_gene_conv_psvs = psv_ixs[gene_conv_psvs[psv_ixs]]
            sample_psv_count_probs = _precompute_probabilities_single_sample(sample_id, sample_cn,
                curr_gene_conv_psvs, psv_counts, psv_infos)
            gene_conv_is_reliable = rel_psvs[curr_gene_conv_psvs]
            rel_sample_psv_count_probs = list(itertools.compress(sample_psv_count_probs, gene_conv_is_reliable))
            if all(value is None for value in rel_sample_psv_count_probs):
                _add_paralog_filter(curr_results, Filter.NoReliable)
                continue

            sample_gt_probs, support_psvs, support_matrix = _single_sample_e_step(sample_genotypes,
                region_group_extra.psv_f_values, psv_info_content, curr_rel_psvs, psvs,
                rel_sample_psv_count_probs, f_pow_combs_cache)
            marginal_probs = _calc_marginal_probs(sample_genotypes, sample_gt_probs, n_copies, sample_cn)

            paralog_cn = np.zeros(n_copies, dtype=np.int)
            paralog_qual = np.zeros(n_copies)
            for copy in range(n_copies):
                best_cn = np.argmax(marginal_probs[copy])
                prob = np.exp(marginal_probs[copy, best_cn])
                if 1 - prob < 1e-16:
                    qual = np.inf
                else:
                    qual = -10 * np.log1p(-prob) / _LOG10
                paralog_cn[copy] = best_cn
                paralog_qual[copy] = min(qual, 1000)

            if sample_cn not in genotypes_str_cache:
                sample_genotypes_str = [','.join(map(str, gt)) for gt in sample_genotypes]
                marginal_str = []
                if n_copies > 2:
                    gt_str = ['?'] * n_copies
                    for copy in range(n_copies):
                        for curr_copy_cn in range(sample_cn + 1):
                            gt_str[copy] = str(curr_copy_cn)
                            marginal_str.append(''.join(gt_str))
                        gt_str[copy] = '?'
                genotypes_str_cache[sample_cn] = (sample_genotypes_str, marginal_str)
            else:
                sample_genotypes_str, marginal_str = genotypes_str_cache[sample_cn]

            GENE_CONV_THRESHOLD = 20
            if np.all(paralog_qual >= GENE_CONV_THRESHOLD):
                gene_conv = _detect_gene_conversion(sample_genotypes_str, sample_gt_probs, support_psvs, support_matrix)
                for entry in gene_conv:
                    out.gene_conversion.write('{}\t{}\t{}\t'.format(region_chrom, entry.start, entry.end))
                    out.gene_conversion.write('{}\t{}\t{}\t{}\t{:.1f}\t{}\n'.format(samples[sample_id], group_name,
                        entry.main_gt, entry.replacement_gt, entry.qual, entry.n_psvs))
            else:
                gene_conv = None

            region1 = Interval(sample_results[start_ix].region1.chrom_id, reg_start, reg_end).to_str(genome)
            outp.write('{}\t{}\t{}\t'.format(group_name, samples[sample_id], region1))
            outp.write('  '.join(map('%s=%.1f'.__mod__, zip(sample_genotypes_str, np.abs(sample_gt_probs / _LOG10)))))
            outp.write('\t')
            if n_copies > 2:
                outp.write('  '.join(map('%s=%.1f'.__mod__,
                    zip(marginal_str, np.abs(marginal_probs.flatten() / _LOG10)))))
            else:
                outp.write('*')
            outp.write('\n')

            mean_info = np.mean(psv_info_content[curr_rel_psvs])
            max_f_value = np.max(np.min(region_group_extra.psv_f_values[curr_rel_psvs], axis=1))
            if mean_info < 0.9:
                _add_paralog_filter(curr_results, Filter.LowInfoContent)
            if max_f_value < 0.99:
                _add_paralog_filter(curr_results, Filter.NoComplReliable)
            if len(curr_rel_psvs) < 3:
                _add_paralog_filter(curr_results, Filter.FewReliable)

            info_update = dict(max_f_value='{:.3f}'.format(max_f_value), gene_conv='T' if gene_conv else 'F',
                semirel_psvs=len(curr_gene_conv_psvs) - len(curr_rel_psvs))

            if end_ix > start_ix + 1:
                info_update['link'] = link_ix
                info_update['link_psvs'] = len(psv_ixs)
                info_update['link_rel_psvs'] = len(curr_rel_psvs)
                info_update['link_psv_info'] = '{:.3f}'.format(mean_info)

            for res_entry in curr_results:
                res_entry.paralog_cn = paralog_cn
                res_entry.paralog_qual = paralog_qual
                res_entry.info.update(info_update)
    return results


class Filter(Enum):
    Pass = 0
    HighCN = 11

    NoReliable = 20
    FewReliable = 21
    NoComplReliable = 22
    LowInfoContent = 23
    UncertainCN = 24

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


class ResultEntry:
    def __init__(self, sample_id, sample_const_region):
        self.sample_id = sample_id
        self.sample_const_region = sample_const_region
        self.n_copies = sample_const_region.cn // 2

        self.ploidy_filter = Filters()
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
            return (self.ploidy_filter.to_str(False), '?', '*')
        return (self.ploidy_filter.to_str(True), self.sample_const_region.pred_cn_str,
            '{:.2f}'.format(self.sample_const_region.qual))

    def paralog_to_str(self):
        if self.paralog_cn is None and self.n_copies == 1:
            self.paralog_cn = (self.sample_const_region.pred_cn_str,)
            self.paralog_qual = (self.sample_const_region.qual,)
        paralog_filter = self.paralog_filter.to_str(self.paralog_cn is not None)

        if self.paralog_cn is None:
            return paralog_filter, ','.join('?' * self.n_copies), '*'

        paralog_cn = []
        paralog_qual = []
        for copy, qual in zip(self.paralog_cn, self.paralog_qual):
            if qual < 5:
                paralog_cn.append('?')
                paralog_qual.append('*')
            else:
                paralog_cn.append(str(copy))
                paralog_qual.append('{:.2f}'.format(qual))
        return paralog_filter, ','.join(paralog_cn), ','.join(paralog_qual)

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
        start_ix, end_ix = searcher.select(start, end)
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
        start_ends = [(entry.region1.start, entry.region1.end) for entry in sample_entries]
        searchers.append(common.NonOverlappingSet.from_start_end_pairs(start_ends))

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
