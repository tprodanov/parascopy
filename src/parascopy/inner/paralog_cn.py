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
import csv
from time import perf_counter
from datetime import timedelta
from enum import Enum
import pysam

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


def _calculate_sample_pscn_probs(psv_gt_probs, psv_gt_coefs):
    # psv_gt_probs: n_samples x n_genotypes
    # psv_gt_coefs: n_pscns   x n_genotypes
    # ext_matrix:   n_samples x n_pscns x n_genotypes
    ext_matrix = psv_gt_probs[:, np.newaxis, :] + psv_gt_coefs[np.newaxis, :, :]
    return logsumexp(ext_matrix, axis=2)


def _e_step(psv_infos, psv_f_values, pscn_priors):
    """
    Returns
        - total ln likelihood,
        - list of lists of P(sample_genotype | allele_counts, psvs),
            outer size = n_samples; inner size for sample_id is len(pscn_priors[sample_id]).
    """
    n_psvs = len(psv_infos)
    prob_matrix = [None] * len(pscn_priors)
    for sample_id, genotype_priors in enumerate(pscn_priors):
        if genotype_priors is not None:
            prob_matrix[sample_id] = genotype_priors.copy()

    total_lik = 0
    for psv_info in psv_infos:
        if not psv_info.in_em:
            continue

        exponent = psv_info.info_content
        f_values = psv_f_values[psv_info.psv_ix]
        total_lik += _get_f_prior(f_values)

        for i, agcn in enumerate(psv_info.em_agcns):
            precomp_data = psv_info.em_precomp_datas[i]
            # psv_gt_probs: n_samples x n_genotypes
            psv_gt_probs = psv_info.em_psv_gt_probs[i]
            # psv_gt_coefs: n_pscns   x n_genotypes
            psv_gt_coefs = precomp_data.fval_to_gt_coefs(f_values)
            # ext_matrix:   n_samples x n_pscns x n_genotypes
            ext_matrix = psv_gt_probs[:, np.newaxis, :] + psv_gt_coefs[np.newaxis, :, :]
            # sample_pscn_probs:  n_samples x n_pscns
            sample_pscn_probs = exponent * logsumexp(ext_matrix, axis=2)

            for sample_id, curr_sample_pscn_probs in zip(psv_info.em_sample_ids[i], sample_pscn_probs):
                prob_matrix[sample_id] += curr_sample_pscn_probs

    for sample_row in prob_matrix:
        if sample_row is not None:
            s = logsumexp(sample_row)
            total_lik += s
            sample_row -= s
    return total_lik, prob_matrix


def _minus_lik_fn(psv_info, sample_pscn_probs):
    reg_pscn_probs = []
    for i, sample_ids in enumerate(psv_info.em_sample_ids):
        n_samples = len(sample_ids)
        n_pscns = psv_info.em_precomp_datas[i].n_poss_pscns
        curr_pscn_probs = np.zeros((n_samples, n_pscns))
        for j, sample_id in enumerate(sample_ids):
            curr_pscn_probs[j] = np.exp(sample_pscn_probs[sample_id])
        reg_pscn_probs.append(curr_pscn_probs)
    n_agcns = len(psv_info.em_agcns)

    def fn(f_values):
        min_lik = -_get_f_prior(f_values)
        assert np.isfinite(min_lik)

        for i in range(n_agcns):
            precomp_data = psv_info.em_precomp_datas[i]

            # psv_gt_probs: n_samples x n_genotypes
            psv_gt_probs = psv_info.em_psv_gt_probs[i]
            # psv_gt_coefs: n_pscns   x n_genotypes
            psv_gt_coefs = precomp_data.fval_to_gt_coefs(f_values)
            # ext_matrix:   n_samples x n_pscns x n_genotypes
            ext_matrix = psv_gt_probs[:, np.newaxis, :] + psv_gt_coefs[np.newaxis, :, :]
            # sample_pscn_probs:  n_samples x n_pscns
            sample_pscn_probs = logsumexp(ext_matrix, axis=2)
            min_lik -= np.sum(sample_pscn_probs * reg_pscn_probs[i])
        return min_lik
    return fn


def _m_step(sample_pscn_probs, psv_infos, psv_ixs, prev_psv_f_values):
    """
    Returns
        - matrix with f-values (n_psvs x n_copies),
        - total_likelihood.
    """
    n_psvs = len(psv_infos)
    n_samples = len(sample_pscn_probs)
    n_copies = prev_psv_f_values.shape[1]
    psv_f_values = np.full_like(prev_psv_f_values, np.nan)

    OPTS = dict(maxiter=50)
    METHOD = 'L-BFGS-B'
    # If the number of samples is small -- cannot estimate f-values very close to 0 or 1.
    min_bound = np.clip(1 / n_samples, 1e-6, 0.03)
    bounds = ((min_bound, 1 - min_bound),) * n_copies

    total_lik = 0
    for psv_ix in psv_ixs:
        psv_info = psv_infos[psv_ix]
        prev_f_values = prev_psv_f_values[psv_ix]

        minus_lik = _minus_lik_fn(psv_info, sample_pscn_probs)
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
        chrom, pos, strand, allele = entry.rsplit(':', 3)
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


def _get_ref_region(sample_regions, ref_agcn):
    if ref_agcn is None:
        return None
    best_region = None
    best_sum_len = 0
    curr_region = None
    curr_sum_len = 0

    for sample_region in sample_regions:
        if sample_region.pred_cn != ref_agcn:
            if curr_sum_len > best_sum_len:
                best_region = curr_region
                best_sum_len = curr_sum_len
            curr_region = None
            curr_sum_len = 0
        elif curr_region is None:
            curr_region = sample_region.region1
            curr_sum_len = len(sample_region.region1)
        else:
            curr_region = curr_region.combine(sample_region.region1)
            curr_sum_len += len(sample_region.region1)

    if curr_sum_len > best_sum_len:
        best_region = curr_region
        best_sum_len = curr_sum_len
    return best_region


def _select_psv_sample_pairs(region_group_extra, samples, out, min_samples, sample_ref_agcns, sample_ref_pscns):
    group = region_group_extra.region_group
    group_name = group.name
    psv_infos = region_group_extra.psv_infos
    sample_const_regions = region_group_extra.sample_const_regions
    n_psvs = len(psv_infos)
    n_samples = len(samples)

    out.write('# Group {}:\n'.format(group_name))
    reliable_regions = []
    n_reliable = 0
    for sample_id, sample in enumerate(samples):
        sample_region = _get_ref_region(sample_const_regions[sample_id], sample_ref_agcns[sample_id])
        reliable_regions.append(sample_region)
        if sample_region is None:
            sample_ref_agcns[sample_id] = sample_ref_pscns[sample_id] = None
        else:
            out.write('#    {}: {:,}-{:,}   agCN {}\n'.format(
                sample, sample_region.start + 1, sample_region.end, sample_ref_agcns[sample_id]))
            n_reliable += 1

    if n_reliable < min_samples:
        out.write('# Too few samples ({} < {}).\n'.format(n_reliable, min_samples))
        return

    for psv_ix, psv_info in enumerate(psv_infos):
        info_str = ''
        use_samples = np.zeros(n_samples, dtype=np.bool_)
        for sample_id, sample_region in enumerate(reliable_regions):
            good_obs = psv_info.sample_infos[sample_id] is not None
            is_ref = sample_region is not None \
                and sample_region.start <= psv_info.start and psv_info.end <= sample_region.end
            info_str += '\t{}{}'.format('+' if good_obs else '-', '+' if is_ref else '-')
            use_samples[sample_id] = good_obs and is_ref

        psv_info.em_prepare(use_samples, sample_ref_agcns)
        psv_info.in_em = psv_info.total_used_samples >= min_samples
        out.write('{}\t{}:{}\t{}{}\n'.format(group_name, psv_info.chrom, psv_info.start + 1,
            psv_info.total_used_samples, info_str))


class SamplePsvInfo:
    def __init__(self, best_cn):
        self.best_cn = best_cn
        self.psv_gt_probs = {}
        self.support_rows = {}


class _PsvInfo:
    def __init__(self, psv_ix, psv, region_group, n_samples, genome):
        self.psv_ix = psv_ix
        self.psv = psv
        self.chrom = psv.chrom
        self.start = psv.start
        self.end = psv.start + len(psv.ref)
        self.is_indel = len(set(map(len, psv.alleles))) != 1
        self.n_alleles = len(psv.alleles)
        self.allele_corresp = _match_psv_alleles(psv, region_group.regions2, genome)

        self.info_content = np.nan
        self.in_em = False
        # Tuple of integers. Denote number of agCN values as N.
        self.em_agcns = None
        # Tuple of sample indices lists (length: N).
        self.em_sample_ids = None
        self.total_used_samples = None
        # One precomputed data for each agCN value (length: N).
        self.em_precomp_datas = None
        # N matrices (samples x genotypes), em_psv_gt_probs[i].shape[0] == len(em_sample_ids[i]).
        self.em_psv_gt_probs = None

        # All precomputed datas (not only used in the EM).
        # keys: agCN values.
        self.precomp_datas = {}
        self.sample_infos = [None] * n_samples

    def em_prepare(self, use_samples, agcn_values):
        self.total_used_samples = np.sum(use_samples)
        if self.total_used_samples == 0:
            return

        self.em_agcns = tuple(sorted(set(itertools.compress(agcn_values, use_samples))))
        self.em_sample_ids = []
        self.em_precomp_datas = []
        self.em_psv_gt_probs = []
        for agcn in self.em_agcns:
            # Cannot use np.where because agcn_values is not a numpy array and may contain Nones.
            curr_sample_ids = [sample_id for sample_id, sample_agcn in enumerate(agcn_values)
                if use_samples[sample_id] and sample_agcn == agcn]
            curr_n_samples = len(curr_sample_ids)
            assert curr_n_samples
            self.em_sample_ids.append(curr_sample_ids)

            precomp_data = self.precomp_datas[agcn]
            self.em_precomp_datas.append(precomp_data)
            n_psv_genotypes = precomp_data.n_psv_genotypes

            psv_gt_probs = np.zeros((curr_n_samples, n_psv_genotypes))
            for i, sample_id in enumerate(curr_sample_ids):
                psv_gt_probs[i] = self.sample_infos[sample_id].psv_gt_probs[agcn]
            self.em_psv_gt_probs.append(psv_gt_probs)

        self.em_sample_ids = tuple(self.em_sample_ids)
        self.em_precomp_datas = tuple(self.em_precomp_datas)
        self.em_psv_gt_probs = tuple(self.em_psv_gt_probs)

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


def _calculate_psv_info_content(group_name, psv_infos, min_samples, out):
    out.write('Region group {}\n'.format(group_name))
    for psv_info in psv_infos:
        if not psv_info.total_used_samples:
            out.write('{}   no applicable samples, skipping.\n'.format(psv_info))
            continue

        # Cache of genotype multipliers based on the aggregate CN.
        n_alleles_m1 = psv_info.n_alleles - 1
        rn_alleles_m1 = 1 / n_alleles_m1
        threshold = 0.8 * rn_alleles_m1

        sum_info_content = 0.0
        inform_samples = 0
        for i, agcn in enumerate(psv_info.em_agcns):
            psv_genotypes = psv_info.em_precomp_datas[i].psv_genotypes
            gt_mults = rn_alleles_m1 * \
                np.fromiter((n_alleles_m1 - gt.count(0) for gt in psv_genotypes),np.float64, len(psv_genotypes))

            for psv_gt_probs in psv_info.em_psv_gt_probs[i]:
                curr_info_content = np.sum(np.exp(psv_gt_probs) * gt_mults)
                sum_info_content += curr_info_content
                if curr_info_content >= threshold:
                    inform_samples += 1

        psv_info.info_content = sum_info_content / psv_info.total_used_samples
        out.write('{}   informative: {}/{} samples. Information content: {:.4f}\n'.format(
            psv_info, inform_samples, psv_info.total_used_samples, psv_info.info_content))
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


def _define_sample_pscn_priors(n_copies, ref_agcns, ref_pscns):
    """
    Returns two lists of size = n_samples.
        First list  = possible sample psCN values for each sample as strings.
        Second list = sample sample psCN priors for each sample.
    """
    # Sample priors are distributed by distance to the reference paralog-specific copy numbers.
    # psCN (4, 0) has distance 4 from psCN (2, 2), therefore will have prior 1e-6.
    DIST_MULT = -1.5 * common.LOG10
    cache = {}
    all_poss_pscns = []
    all_priors = []
    for sample_id, agcn in enumerate(ref_agcns):
        if agcn is None:
            all_poss_pscns.append(None)
            all_priors.append(None)
            continue

        pscn = ref_pscns[sample_id]
        poss_pscns_str, priors = cache.get(pscn, (None, None))
        if priors is None:
            poss_pscns = variants_.all_gt_counts(n_copies, agcn)
            ref_pscn = np.array(pscn)
            priors = np.full(len(poss_pscns), np.nan)
            poss_pscns_str = []
            for i, curr_pscn in enumerate(poss_pscns):
                poss_pscns_str.append(','.join(map(str, curr_pscn)))
                priors[i] = np.sum(np.abs(ref_pscn - curr_pscn)) * DIST_MULT
            priors -= logsumexp(priors)
            cache[pscn] = (poss_pscns_str, priors)

        all_poss_pscns.append(poss_pscns_str)
        all_priors.append(priors)
    return all_poss_pscns, all_priors


def _cluster_psvs(psv_infos, psv_counts, n_samples):
    n_psvs = len(psv_infos)

    ref_fractions = np.full((n_psvs, n_samples), np.nan)
    for psv_info in psv_infos:
        if not psv_info.total_used_samples:
            continue
        allele_corresp = psv_info.allele_corresp
        mult = len(allele_corresp) / allele_corresp.count(0)
        for sample_id in itertools.chain(*psv_info.em_sample_ids):
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
    use_psvs = np.ones(n_psvs, dtype=np.bool_)
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
    'cluster_i n_reliable info_content likelihood psv_f_values poss_pscn_probs')


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

    out.checked_write('em_pscn', '# Format for each sample: "psCN:qual"\n')
    out.checked_write('em_pscn',
        'region_group\tcluster\titeration\ttime\tlikelihood\tn_reliable\treliable_info\t' + samples_str)

    out.checked_write('paralog_cn', 'region_group\tsample\tregion1\tgenotypes\tmarginal_probs\n')
    out.checked_write('gene_conversion',
        '#chrom\tstart\tend\tsample\tregion_group\tmain_gt\treplacement_gt\tqual\tn_psvs\n')


def create_psv_infos(psvs, region_group, n_samples, genome):
    return [_PsvInfo(psv_ix, psv, region_group, n_samples, genome) for psv_ix, psv in enumerate(psvs)]


def _get_ref_pscns(samples, genome, region_group, const_regions, modified_ref_cns, max_agcn):
    """
    Looks up modified reference CNs and returns two lists
       - reference agCNs for all samples,
       - reference psCNs for all samples.
    If for some reason the sample is not appropriate for the EM-algorithm, both entries would be None.
    """
    n_samples = len(samples)
    n_copies = 1 + len(region_group.regions2)
    def_agcn = 2 * n_copies
    def_pscn = (2,) * n_copies

    if modified_ref_cns is None:
        if def_agcn > max_agcn:
            return ([None] * n_samples,) * 2
        return [def_agcn] * n_samples, [def_pscn] * n_samples

    undefined = object()
    ref_agcns = [undefined] * n_samples
    ref_pscns = [undefined] * n_samples

    for i, region_ix in enumerate(region_group.region_ixs):
        const_region = const_regions[region_ix]
        regions = [const_region.region1]
        for region2, _ in const_region.regions2:
            regions.append(region2)

        for sample_id, sample in enumerate(samples):
            if ref_pscns[sample_id] is None:
                continue

            pscn, _ = modified_ref_cns.from_regions(regions, sample_id, sample, genome)
            if pscn is None:
                ref_agcns[sample_id] = ref_pscns[sample_id] = None
                continue

            if i > 0:
                if ref_pscns[sample_id] != pscn:
                    common.log(
                        'ERROR: Input BED file {} contains non-matching entries for sample {} and region group {}'
                            .format(modified_ref_cns.filename, samples[sample_id], region_group.name))
                    ref_agcns[sample_id] = ref_pscns[sample_id] = None
                continue

            agcn = sum(pscn)
            if agcn < 2 or agcn > max_agcn or sum(map(bool, pscn)) <= 1:
                ref_agcns[sample_id] = ref_pscns[sample_id] = None
            else:
                ref_agcns[sample_id] = agcn
                ref_pscns[sample_id] = pscn
    return ref_agcns, ref_pscns


def find_reliable_psvs(region_group_extra, samples, genome, modified_ref_cns, out, *,
        min_samples, reliable_threshold, max_agcn):
    # ===== Setting up variables =====
    psvs = region_group_extra.psvs
    n_psvs = len(psvs)
    region_group = region_group_extra.region_group
    const_regions = region_group_extra.dupl_hierarchy.const_regions
    group_name = region_group.name
    n_copies = 1 + len(region_group.regions2)
    n_samples = len(samples)
    if not n_psvs or n_copies <= 1:
        return

    # ===== Selecting a set of PSVs used in the EM algorithm =====
    ref_agcns, ref_pscns = _get_ref_pscns(samples, genome, region_group, const_regions, modified_ref_cns, max_agcn)
    if not any(ref_agcns):
        return
    _select_psv_sample_pairs(region_group_extra, samples, out.use_psv_sample, min_samples, ref_agcns, ref_pscns)

    psv_infos = region_group_extra.psv_infos
    region_sequence = region_group.region1.get_sequence(genome)
    for psv_info in psv_infos:
        psv_info.check_complicated_pos(region_group.region1, region_sequence)
    if not any(psv_info.in_em for psv_info in psv_infos):
        return

    timer_start = perf_counter()
    _calculate_psv_info_content(group_name, psv_infos, min_samples, out.psv_filtering)
    _filter_close_psvs(psv_infos, out.psv_filtering, close_psv_dist=100)
    em_psv_ixs = np.array([psv_info.psv_ix for psv_info in psv_infos if psv_info.in_em])
    if not len(em_psv_ixs):
        return

    common.log('    Searching for reliable PSVs (out of total {} PSVs)'.format(len(em_psv_ixs)))
    # ===== EM iterations, try several clusters =====
    poss_pscns_str, poss_pscn_priors = _define_sample_pscn_priors(n_copies, ref_agcns, ref_pscns)
    best_cluster = None
    psv_clusters = _cluster_psvs(psv_infos, region_group_extra.psv_read_counts, n_samples)
    for cluster_i, cluster in enumerate(psv_clusters, start=1):
        psv_f_values = np.full((n_psvs, n_copies), 0.5)
        psv_f_values[cluster] = 0.9

        total_lik = -np.inf
        for iteration in range(1, 101):
            if iteration > 1:
                psv_f_values, _ = _m_step(poss_pscn_probs, psv_infos, em_psv_ixs, psv_f_values)
            for psv_ix in em_psv_ixs:
                psv_info = psv_infos[psv_ix]
                out.interm_psv_f_values.write('{}\t{}\t{}\t{}:{}\t{:.6f}\t{}\n'.format(group_name, cluster_i,
                    iteration, psv_info.chrom, psv_info.start + 1, psv_info.info_content,
                    '\t'.join(map('{:.6f}'.format, psv_f_values[psv_ix]))))

            old_lik = total_lik
            total_lik, poss_pscn_probs = _e_step(psv_infos, psv_f_values, poss_pscn_priors)
            common.log('        Cluster {}, iteration {:2}, EM likelihood: {:,.3f}'.format(
                cluster_i, iteration, total_lik / common.LOG10))
            with np.errstate(invalid='ignore'):
                curr_reliable = np.where(np.all(psv_f_values >= reliable_threshold, axis=1))[0]

            n_reliable = len(curr_reliable)
            mean_info_content = np.mean([psv_infos[i].info_content for i in curr_reliable]) if n_reliable else 0.0
            elapsed_time = str(timedelta(seconds=perf_counter() - timer_start))[:-5]
            out.em_pscn.write('{}\t{}\t{}\t{}\t{:.7f}\t{}\t{:.6f}'.format(group_name, cluster_i, iteration,
                elapsed_time, total_lik / common.LOG10, n_reliable, mean_info_content))
            for sample_id, curr_pscn_probs in enumerate(poss_pscn_probs):
                if curr_pscn_probs is None:
                    out.em_pscn.write('\t*')
                    continue
                best_pscn_i = np.argmax(curr_pscn_probs)
                pscn_qual = common.phred_qual(curr_pscn_probs, best_pscn_i)
                out.em_pscn.write('\t{}:{:.0f}'.format(poss_pscns_str[sample_id][best_pscn_i], pscn_qual))
            out.em_pscn.write('\n')

            if total_lik < old_lik + 0.01:
                break
        if best_cluster is None or best_cluster.likelihood < total_lik:
            best_cluster = _BestCluster(cluster_i, n_reliable, mean_info_content, total_lik,
                psv_f_values, poss_pscn_probs)

    # ===== Save results from the last cluster =====
    psv_f_values = best_cluster.psv_f_values
    poss_pscn_probs = best_cluster.poss_pscn_probs
    if len(psv_clusters) > 1:
        common.log('    === Best cluster {}: likelihood {:.3f},  {} reliable PSVs with mean information content {:.3f}'
            .format(best_cluster.cluster_i, best_cluster.likelihood / common.LOG10,
            best_cluster.n_reliable, best_cluster.info_content))
    if best_cluster.n_reliable and best_cluster.info_content < 0.8:
        common.log('WARN: Many reliable PSVs have low information content.')

    discarded_psvs = np.array([psv_info.psv_ix for psv_info in psv_infos
        if not psv_info.in_em and psv_info.total_used_samples > 0])
    if len(discarded_psvs):
        oth_f_values, _ = _m_step(poss_pscn_probs, psv_infos, discarded_psvs, np.full((n_psvs, n_copies), 0.5))
        psv_f_values[discarded_psvs, :] = oth_f_values[discarded_psvs, :]
    for psv_info in psv_infos:
        out.psv_f_values.write('{}\t{}:{}\t{}\t{}\t{:.6f}\t{}\n'.format(group_name, psv_info.chrom,
            psv_info.start + 1, psv_info.total_used_samples, 'T' if psv_info.in_em else 'F', psv_info.info_content,
            '\t'.join(map('{:.6f}'.format, psv_f_values[psv_info.psv_ix]))))
    region_group_extra.set_f_values(psv_f_values)


def _create_psv_weights(psv_infos, neighb_radius):
    i = 0
    k = 0
    n = len(psv_infos)
    neighb_size = np.zeros(n)
    for j, psv_info in enumerate(psv_infos):
        while i <= j and psv_info.start - psv_infos[i].start > neighb_radius:
            i += 1
        while k < n and psv_infos[k].start - psv_info.start <= neighb_radius:
            k += 1
        neighb_size[j] = k - i
    return 1.0 / neighb_size


def _single_sample_e_step(sample_id, agcn, psv_infos, psv_weights=None):
    """
    Returns sample genotypes & sample genotype probabilities.
    If it is impossible to estimate genotype probabilities (one of the PSV does not have necessary values),
        return (None, None).
    """
    # NOTE: Should we set paralog-specific priors?
    precomp_data = psv_infos[0].precomp_datas.get(agcn)
    if precomp_data is None:
        return None, None
    gt_probs = np.zeros(precomp_data.n_poss_pscns)

    if psv_weights is None:
        psv_weights = itertools.repeat(1.0)
    for psv_info, psv_weight in zip(psv_infos, psv_weights):
        support_row = psv_info.sample_infos[sample_id].support_rows.get(agcn)
        if support_row is None:
            return None, None
        gt_probs += support_row * psv_weight
    return precomp_data.poss_pscns, gt_probs


def _identify_best_agcn(sample_id, probable_cns, remaining_prob, psv_infos):
    """
    Finds best AggregateCN based on ParalogCN estimates. Possible AggregateCN values are taken from probable_cns.
    Returns pair (AggregateCN, quality).
    """
    n_cns = len(probable_cns)
    assert n_cns > 1
    psv_weights = _create_psv_weights(psv_infos, neighb_radius=500)
    agcn_probs = np.full(n_cns, -np.inf)
    initial_probs = []
    for i, weighted_cn in enumerate(probable_cns):
        if weighted_cn.agcn is not None:
            _, gt_probs = _single_sample_e_step(sample_id, weighted_cn.agcn, psv_infos, psv_weights)
            if gt_probs is not None:
                agcn_probs[i] = logsumexp(gt_probs) + weighted_cn.prob
                initial_probs.append(weighted_cn.prob)
    assert np.isfinite(agcn_probs[0])

    # Total sum will be the same as the initial sum of weights.
    agcn_probs -= logsumexp(agcn_probs)
    best = np.argmax(agcn_probs)
    remaining_prob = max(remaining_prob, common.log1minus(initial_probs))
    qual = common.extended_phred_qual(agcn_probs, best, rem_prob=remaining_prob, max_value=1000)

    agcn = probable_cns[best].agcn
    assert str(agcn) == probable_cns[best].agcn_str
    return agcn, qual


def _get_psv_subset(results, psv_searcher, psv_infos, sample_id, agcn):
    psv_ixs = []
    for subresults in results:
        for psv_ix in range(*psv_searcher.contained_ixs(subresults.region1.start, subresults.region1.end)):
            sample_info = psv_infos[psv_ix].sample_infos[sample_id]
            if sample_info is not None:
                sample_info.best_cn = agcn
                if agcn in sample_info.psv_gt_probs:
                    psv_ixs.append(psv_ix)
    return np.array(psv_ixs)


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
        - any_known: bool (any of the quality values over the threshold).
    If paralog quality is less than min_qual_value, corresponding CN is replaced with '?' and quality
    is replaced with 0. Additionally, quality is rounded down to integers.
    """
    paralog_cn_str = []
    new_paralog_qual = []
    agcn = 0
    sum_hq_pscn = 0
    n_hq = 0
    for cn, qual in zip(paralog_cn, paralog_qual):
        agcn += cn
        if qual >= min_qual_value:
            paralog_cn_str.append(str(cn))
            new_paralog_qual.append(int(qual))
            n_hq += 1
            sum_hq_pscn += cn
        else:
            paralog_cn_str.append('?')
            new_paralog_qual.append(0)

    paralog_cn_str = ','.join(paralog_cn_str)
    if sum_hq_pscn == agcn and n_hq < len(paralog_cn):
        paralog_cn_str = paralog_cn_str.replace('?', '0')
    return paralog_cn_str, tuple(new_paralog_qual), bool(n_hq)


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


def _detect_gene_conversion(sample_id, sample_agcn, genotypes_str, poss_pscn_probs, psv_infos, semirel_psv_ixs):
    n_psvs = len(semirel_psv_ixs)
    n_genotypes = len(genotypes_str)

    best_gt = np.argmax(poss_pscn_probs)
    model = GeneConversionHmm(best_gt, n_genotypes, n_psvs)

    emission_matrix = np.zeros((1, n_genotypes, n_psvs))
    HMM_SAMPLE_ID = 0
    for i, psv_ix in enumerate(semirel_psv_ixs):
        sample_info = psv_infos[psv_ix].sample_infos[sample_id]
        assert sample_info.best_cn == sample_agcn
        emission_matrix[HMM_SAMPLE_ID, :, i] = sample_info.support_rows[sample_info.best_cn]
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
    group_name = region_group_extra.region_group.name
    for sample_const_region in region_group_extra.sample_const_regions[sample_id]:
        entry = ResultEntry(sample_id, sample_const_region)
        entry.info['group'] = group_name

        reg_start = sample_const_region.region1.start
        reg_end = sample_const_region.region1.end
        entry.info['n_windows'] = region_group_extra.group_windows_searcher.n_overlaps(reg_start, reg_end)
        entry.info['hmm_windows'] = region_group_extra.hmm_windows_searcher.n_overlaps(reg_start, reg_end)

        psv_start_ix, psv_end_ix = region_group_extra.psv_searcher.contained_ixs(reg_start, reg_end)
        entry.info['n_psvs'] = psv_end_ix - psv_start_ix
        entry.info['indep_psvs'] = np.sum(region_group_extra.psv_in_em[psv_start_ix : psv_end_ix])
        entry.info['rel_psvs'] = np.sum(region_group_extra.psv_is_reliable[psv_start_ix : psv_end_ix])
        entry.info['semirel_psvs'] = np.sum(region_group_extra.psv_is_reliable[psv_start_ix : psv_end_ix])
        sample_results.append(entry)
    return sample_results


def _genotypes_str(poss_pscns, genotypes_str_cache):
    """
    Returns
        - string representations of sample genotypes,
        - string representations of various marginal probabilities in form (0??, 1??, ..., ?0?, ...).
    """
    n_copies = len(poss_pscns[0])
    sample_agcn = sum(poss_pscns[0])
    if sample_agcn in genotypes_str_cache:
        return genotypes_str_cache[sample_agcn]

    poss_pscns_str = [','.join(map(str, gt)) for gt in poss_pscns]
    marginal_str = []
    if n_copies > 2:
        gt_str = ['?'] * n_copies
        for copy in range(n_copies):
            for curr_copy_cn in range(sample_agcn + 1):
                gt_str[copy] = str(curr_copy_cn)
                marginal_str.append(''.join(gt_str))
            gt_str[copy] = '?'
    res = (poss_pscns_str, marginal_str)
    genotypes_str_cache[sample_agcn] = res
    return res


def _refine_agcn_probs(sample_id, sample_results, region_group_extra):
    psv_searcher = region_group_extra.psv_searcher
    psv_infos = region_group_extra.psv_infos

    for entry in sample_results:
        const_region = entry.sample_const_region
        probable_cns = const_region.probable_cns
        if len(probable_cns) < 2:
            continue

        agcn1 = probable_cns[0].agcn
        psv_ixs = _get_psv_subset((entry,), psv_searcher, psv_infos, sample_id, agcn1)
        if len(psv_ixs) == 0:
            continue
        reliable_psv_ixs = psv_ixs[region_group_extra.psv_is_reliable[psv_ixs]]
        n_rel = len(reliable_psv_ixs)
        if n_rel == 0:
            continue
        rel_psv_infos = [psv_infos[psv_ix] for psv_ix in reliable_psv_ixs]
        agcn2, agcn2_qual = _identify_best_agcn(sample_id, probable_cns, const_region.remaining_prob, rel_psv_infos)
        const_region.update_pred_cn(agcn2, agcn2_qual)
        entry.info['agCN_psvs'] = n_rel


def _link_sample_results(sample_results):
    linked_ranges = []
    prev_entry = None
    for entry_ix, entry in enumerate(sample_results):
        if prev_entry is not None and prev_entry.pred_cn == entry.pred_cn:
            linked_ranges[-1][1] = entry_ix + 1
        else:
            linked_ranges.append([entry_ix, entry_ix + 1])
        prev_entry = entry
    return linked_ranges


def _single_sample_pscn(sample_id, sample_name, sample_results, region_group_extra, genome, out, genotypes_str_cache):
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
    linked_ranges = _link_sample_results(sample_results)
    for link_ix, (start_ix, end_ix) in enumerate(linked_ranges):
        curr_results = sample_results[start_ix:end_ix]
        sample_agcn = curr_results[0].pred_cn
        if sample_agcn == 0:
            continue
        if not curr_results[0].sample_const_region.cn_is_known:
            _add_paralog_filter(curr_results, Filter.UncertainCN)
            continue

        psv_ixs = _get_psv_subset(curr_results, psv_searcher, psv_infos, sample_id, sample_agcn)
        if len(psv_ixs) == 0:
            _add_paralog_filter(curr_results, Filter.NoGoodPSVs)
            continue
        reliable_psv_ixs = psv_ixs[region_group_extra.psv_is_reliable[psv_ixs]]
        if len(reliable_psv_ixs) == 0:
            _add_paralog_filter(curr_results, Filter.NoReliable)
            continue
        rel_psv_infos = [psv_infos[psv_ix] for psv_ix in reliable_psv_ixs]

        # ===== Run E-step once again to calculate psCN =====
        poss_pscns, poss_pscn_probs = _single_sample_e_step(sample_id, sample_agcn, rel_psv_infos)
        marginal_probs, paralog_cn, paralog_qual = calculate_marginal_probs(poss_pscns, poss_pscn_probs,
            n_copies, sample_agcn)
        poss_pscns_str, marginal_str = _genotypes_str(poss_pscns, genotypes_str_cache)

        # ===== Detect gene conversion =====
        GENE_CONV_QUAL_THRESHOLD = 20
        MIN_SEMIREL_PSVS = 3
        semirel_psv_ixs = psv_ixs[region_group_extra.psv_is_semirel[psv_ixs]]

        if np.all(paralog_qual >= GENE_CONV_QUAL_THRESHOLD) and len(semirel_psv_ixs) >= MIN_SEMIREL_PSVS:
            gene_conv = _detect_gene_conversion(sample_id, sample_agcn, poss_pscns_str, poss_pscn_probs,
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
            zip(poss_pscns_str, np.abs(poss_pscn_probs / common.LOG10)))))
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
            indep_psvs=np.sum(region_group_extra.psv_in_em[psv_ixs]),
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


def estimate_paralog_cn(region_group_extra, samples, genome, out, max_agcn):
    common.log('    Calculating paralog-specific copy number profiles')
    # ===== Defining useful variables =====
    region_group = region_group_extra.region_group
    n_copies = region_group.cn // 2
    genotypes_str_cache = {}
    has_reliable_psvs = np.sum(region_group_extra.psv_is_reliable) > 0

    results = []
    for sample_id in range(len(samples)):
        sample_results = _create_sample_results_from_agcn(sample_id, region_group_extra)
        results.extend(sample_results)
        if not has_reliable_psvs:
            if n_copies > max_agcn // 2:
                _add_paralog_filter(sample_results, Filter.HighCN)
            elif n_copies > 1:
                _add_paralog_filter(sample_results, Filter.NoReliable)
            continue

        _refine_agcn_probs(sample_id, sample_results, region_group_extra)
        _single_sample_pscn(sample_id, samples[sample_id], sample_results, region_group_extra, genome, out,
            genotypes_str_cache)
    return results


class Filter(Enum):
    Pass = 0
    HighCN = 11

    NoReliable = 20
    FewReliable = 21
    NoComplReliable = 22
    LowInfoContent = 23
    UncertainCN = 24
    NoGoodPSVs = 25

    def __str__(self):
        if self == Filter.Pass:
            return 'PASS'
        return self.name

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
        if s == 'LowInfoCont' or s == 'LowInfoContent':
            return Filter.LowInfoContent
        if s == 'UncertainCN':
            return Filter.UncertainCN
        if s == 'NoGoodPSVs':
            return Filter.NoGoodPSVs


class Filters:
    def __init__(self):
        self._filters = None

    @classmethod
    def from_str(cls, subcls, s):
        self = Filters()
        return self.update_str(subcls, s)

    def update_str(self, subcls, s):
        if s == '*' or s == 'PASS':
            return self
        for filt in s.split(';'):
            self.add(subcls.from_str(filt))
        return self

    def add(self, filt):
        if self._filters is None:
            self._filters = set()
        self._filters.add(filt)

    def map(self, fn):
        if self._filters is None:
            return ()
        return map(fn, self._filters)

    def union(self, other):
        new = Filters()
        if self._filters is None and other._filters is None:
            new._filters = None
        elif self._filters is None:
            new._filters = other._filters.copy()
        elif other._filters is None:
            new._filters = self._filters.copy()
        else:
            new._filters = self._filters | other._filters
        return new

    def to_str(self, default_pass=True):
        if not self._filters:
            return 'PASS' if default_pass else '*'
        return ';'.join(map(str, self._filters))

    def to_tuple(self):
        if not self._filters:
            return ('PASS',)
        return tuple(map(str, self._filters))

    def __len__(self):
        return 0 if self._filters is None else len(self._filters)

    def __bool__(self):
        return bool(self._filters)

    def copy(self):
        res = Filters()
        if self._filters:
            res._filters = self._filters.copy()
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
        return (self.agcn_filter.to_str(), self.sample_const_region.pred_cn_str,
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

        info = { **self.info, **self.sample_const_region.info }
        if info:
            res += ';'.join(map('%s=%s'.__mod__, info.items()))
        else:
            res += '*'
        res += '\t'
        res += self.sample_const_region.regions2_str(genome)
        return res

    def __lt__(self, other):
        return self.region1.__lt__(other.region1)

    @classmethod
    def from_dict(cls, row, genome, samples):
        sample_id = samples.id_or_none(row['sample'])
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

    out.write('## {}\n'.format(common.command_to_str()))
    out.write('## {} v{}\n'.format(__pkg_name__, __version__))
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


class CopyNumProfiles:
    def __init__(self, filename, genome, samples, search_chrom_id=None):
        n_samples = len(samples)
        self._cn_profiles = [[] for _ in range(n_samples)]

        with common.open_possible_gzip(filename) as inp:
            fieldnames = None
            for line in inp:
                if line.startswith('##'):
                    continue
                assert line.startswith('#')
                fieldnames = line[1:].strip().split('\t')
                break
            assert fieldnames is not None

            reader = csv.DictReader(inp, fieldnames, delimiter='\t')
            for row in reader:
                entry = ResultEntry.from_dict(row, genome, samples)
                if entry.sample_id is not None:
                    self._cn_profiles[entry.sample_id].append(entry)

        self._search_chrom_id = search_chrom_id
        self._searchers = None
        if search_chrom_id is not None:
            self._searchers = []
            for sample_entries in self._cn_profiles:
                curr_sample_entries = [entry for entry in sample_entries if entry.region1.chrom_id == search_chrom_id]
                self._searchers.append(itree.NonOverlTree(curr_sample_entries, itree.region1_start, itree.region1_end))

    @property
    def search_chrom_id(self):
        return self._search_chrom_id

    @property
    def n_samples(self):
        return len(self._cn_profiles)

    def sample_profiles(self, sample_id):
        return self._cn_profiles[sample_id]

    def cn_estimates(self, sample_id, region):
        assert region.chrom_id == self._search_chrom_id
        return tuple(self._searchers[sample_id].overlap_iter(region.start, region.end))


class ParalogEntry:
    @classmethod
    def create(cls, entry, region_ix, store_hom_regions=False):
        if entry.paralog_cn[region_ix] is None:
            return None
        self = cls.__new__(cls)
        self.region1 = entry.region1 if region_ix == 0 else entry.sample_const_region.regions2[region_ix - 1][0]
        self.sample_id = entry.sample_id
        self.pscn = entry.paralog_cn[region_ix]
        self.pscn_qual = entry.paralog_qual[region_ix]
        self.pscn_filter = entry.paralog_filter

        self.agcn = entry.sample_const_region.pred_cn
        self.agcn_qual = entry.sample_const_region.qual
        self.agcn_filter = entry.agcn_filter
        self.main_region = entry.region1

        self.regions2 = None
        if store_hom_regions:
            self.regions2 = []
            if region_ix == 0 and entry.sample_const_region.regions2 is not None:
                self.regions2.extend(region for region, _strand in entry.sample_const_region.regions2)
            else:
                self.regions2.append(entry.region1)
                self.regions2.extend(region for i, (region, _strand)
                    in enumerate(entry.sample_const_region.regions2) if i + 1 != region_ix)
        return self

    @classmethod
    def from_pooled_entry(cls, pooled_entry, region_ix, cn, cn_filter=None, cn_qual=10000):
        self = cls.__new__(cls)
        self.region1 = pooled_entry.region1 if region_ix == 0 else pooled_entry.regions2[region_ix - 1]
        self.sample_id = pooled_entry.sample_id
        self.pscn = cn
        self.pscn_qual = cn_qual
        self.pscn_filter = Filters() if cn_filter is None else cn_filter

        self.agcn = pooled_entry.cn
        self.agcn_qual = pooled_entry.qual
        self.agcn_filter = pooled_entry.filter
        self.main_region = pooled_entry.region1
        if region_ix == 0:
            self.regions2 = list(pooled_entry.regions2)
        else:
            self.regions2 = [pooled_entry.region1]
            self.regions2.extend(region for i, region in enumerate(pooled_entry.regions2) if i + 1 != region_ix)
        return self

    @classmethod
    def parse(cls, tup, genome, samples):
        self = cls.__new__(cls)
        self.region1 = Interval(genome.chrom_id(tup[0]), int(tup[1]), int(tup[2]))
        self.sample_id = samples.id(tup[3])

        self.pscn_filter = Filters.from_str(Filter, tup[4])
        self.pscn = int(tup[5])
        self.pscn_qual = float(tup[6])

        self.agcn_filter = Filters.from_str(Filter, tup[7])
        self.agcn = int(tup[8])
        self.agcn_qual = float(tup[9])
        self.main_region = Interval.parse(tup[10], genome)
        self.regions2 = None
        if len(tup) >= 12:
            if tup[11] == '*':
                self.regions2 = ()
            else:
                self.regions2 = tuple(Interval.parse(s, genome) for s in tup[11].split(','))
        return self

    def copy(self, new_region=None):
        cls = self.__class__
        new = cls.__new__(cls)
        new.region1 = new_region or self.region1
        new.sample_id = self.sample_id

        new.pscn = self.pscn
        new.pscn_qual = self.pscn_qual
        new.pscn_filter = self.pscn_filter.copy()

        new.agcn = self.agcn
        new.agcn_qual = self.agcn_qual
        new.agcn_filter = self.agcn_filter.copy()
        new.main_region = self.main_region
        if new_region is None:
            new.regions2 = self.regions2
        else:
            # Cannot set new region if regions2 are non-empty.
            assert self.regions2 is None
            new.regions2 = None
        return new

    def __lt__(self, other):
        return self.region1.__lt__(other.region1)

    def compatible(self, other):
        return self.region1.chrom_id == other.region1.chrom_id and self.region1.end == other.region1.start \
            and self.sample_id == other.sample_id \
            and self.pscn == other.pscn and self.pscn_qual == self.pscn_qual

    @classmethod
    def extend_entries(cls, entries, new_subregion, paralog_entries):
        if len(paralog_entries) == 1:
            entry = paralog_entries[0]
        else:
            best_i = np.argmax([min(entry.pscn_qual, entry.agcn_qual) for entry in paralog_entries])
            entry = paralog_entries[best_i]
        if entry.region1 != new_subregion:
            entry = entry.copy(new_subregion)

        if entries and entries[-1].compatible(entry):
            prev = entries[-1]
            entries[-1] = prev.copy(prev.region1.combine(entry.region1))
        else:
            entries.append(entry)

    def to_str(self, genome, samples):
        if self.regions2 is None:
            reg_col = self.main_region.to_str(genome)
        elif not self.regions2:
            reg_col = '*'
        else:
            reg_col = ','.join(region.to_str(genome) for region in self.regions2)
        return '{}\t{}\t{}\t{}\t{:.0f}\t{}\t{}\t{:.0f}\t{}\n'.format(
            self.region1.to_bed(genome), samples[self.sample_id],
            self.pscn_filter.to_str(), self.pscn, self.pscn_qual,
            self.agcn_filter.to_str(), self.agcn, self.agcn_qual, reg_col)

    @classmethod
    def load(cls, filename, genome, samples):
        sample_entries = [[] for _ in range(len(samples))]
        with pysam.TabixFile(filename, parser=pysam.asTuple()) as inp:
            for entry in inp.fetch():
                par_entry = cls.parse(entry, genome, samples)
                sample_entries[par_entry.sample_id].append(par_entry)

        trees = []
        region_getter = operator.attrgetter('region')
        for par_entries in sample_entries:
            trees.append(itree.MultiNonOverlTree(par_entries, region_getter))
        return trees


def summary_to_paralog_bed(in_filename, out_filename, genome, samples, tabix):
    cn_profiles = CopyNumProfiles(in_filename, genome, samples)
    all_entries = []
    for sample_id in range(len(samples)):
        sample_entries = []
        for entry in cn_profiles.sample_profiles(sample_id):
            for i in range(entry.sample_const_region.cn // 2):
                par_entry = ParalogEntry.create(entry, i)
                if par_entry is not None:
                    sample_entries.append(par_entry)
        sample_entries.sort()

        subregions = Interval.get_disjoint_subregions(map(operator.attrgetter('region1'), sample_entries))
        for subregion, subregion_ixs in subregions:
            if len(subregion_ixs) > 1:
                ParalogEntry.extend_entries(all_entries, subregion, [sample_entries[i] for i in subregion_ixs])
            else:
                ParalogEntry.extend_entries(all_entries, subregion, (sample_entries[subregion_ixs[0]],))
    all_entries.sort()

    with common.open_possible_gzip(out_filename, 'w', bgzip=True) as out:
        out.write('## {}\n'.format(common.command_to_str()))
        out.write('## {} v{}\n'.format(__pkg_name__, __version__))
        out.write('#chrom\tstart\tend\tsample\tcn_filter\tcn\tcn_qual\t'
            'agcn_filter\tagcn\tagcn_qual\tmain_region\n')
        for entry in all_entries:
            out.write(entry.to_str(genome, samples))
    if out_filename.endswith('.gz') and tabix != 'none':
        common.Process([tabix, '-p', 'bed', out_filename]).finish()


class PooledEntry:
    def __init__(self, res_entry, artificial_cn=None):
        self.region1 = res_entry.region1
        self.sample_id = res_entry.sample_id
        if res_entry.sample_const_region.regions2 is None:
            self.regions2 = ()
        else:
            self.regions2 = tuple(map(operator.itemgetter(0), res_entry.sample_const_region.regions2))

        if artificial_cn is None:
            self.cn = res_entry.pred_cn
            self.filter = res_entry.agcn_filter
            self.qual = res_entry.sample_const_region.qual
        else:
            self.cn = artificial_cn
            self.filter = Filters()
            self.qual = 10000

    def __lt__(self, oth):
        return self.region1.__lt__(oth.region1)

    def to_str(self, genome, samples):
        return '{}\t{}\t{}\t{}\t{:.0f}\t{}\n'.format(self.region1.to_bed(genome), samples[self.sample_id],
            self.filter.to_str(), self.cn, self.qual,
            ','.join(region.to_str(genome) for region in self.regions2) if self.regions2 else '*')

    def to_str_short(self, genome, samples):
        return '{}\t{}\t{}\n'.format(self.region1.to_bed(genome), samples[self.sample_id], self.cn)

    def to_str_zero(self, genome, samples):
        return '{}\t{}\t0\n'.format(self.region1.to_bed(genome), samples[self.sample_id])

    @classmethod
    def parse(cls, tup, genome, samples):
        self = cls.__new__(cls)
        self.region1 = Interval(genome.chrom_id(tup[0]), int(tup[1]), int(tup[2]))
        self.sample_id = samples.id(tup[3])
        self.filter = Filters.from_str(Filter, tup[4])
        self.cn = int(tup[5])
        self.qual = float(tup[6])
        self.ref_cn = int(tup[7])
        if tup[8] == '*':
            self.regions2 = ()
        else:
            self.regions2 = tuple(Interval.parse(s, genome) for s in tup[8].split(','))
        return self
