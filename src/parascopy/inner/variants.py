import sys
import operator
import itertools
import pysam
import numpy as np
from enum import Enum
from scipy.stats import poisson, multinomial, fisher_exact
from scipy.special import logsumexp, gammaln
from functools import lru_cache
from collections import namedtuple, defaultdict

from . import duplication as duplication_
from .cigar import Cigar, Operation
from .genome import Interval
from .paralog_cn import Filters, SamplePsvInfo
from . import errors
from . import common
from . import itree
from . import polynomials


def copy_vcf_fields(source_rec, target_rec, old_to_new=None):
    """
    Copies format fields from source_rec to target_rec.
        Copies fields GQ, DP, GT, AD.
    old_to_new: None or list of indices, old_to_new[old_allele_index] = new_allele_index.
    """
    n_alleles = len(target_rec.alleles)
    for sample, source in source_rec.samples.items():
        target = target_rec.samples[sample]
        if 'GQ' in source:
            target['GQ'] = source['GQ']
        if 'DP' in source:
            target['DP'] = source['DP']

        if 'GT' in source:
            source_gt = source['GT']
            if source_gt is None or source_gt[0] is None:
                pass
            elif old_to_new is None:
                target['GT'] = source_gt
            else:
                target['GT'] = tuple(old_to_new[j] for j in source_gt)

        if 'AD' in source:
            source_ad = source['AD']
            if source_ad is None or source_ad[0] is None:
                pass
            elif old_to_new is None:
                target['AD'] = source_ad
            else:
                target_ad = [0] * n_alleles
                for i, j in enumerate(old_to_new):
                    target_ad[j] += source_ad[i]
                target['AD'] = target_allele_depth


MAX_SHIFT_VAR_SIZE = 50


def _move_left_gap(gap_start, gap_seq, chrom_seq, chrom_seq_shift, min_start):
    m = len(gap_seq)
    k = m - 1
    while gap_start > min_start and gap_seq[k] == chrom_seq[gap_start - chrom_seq_shift - 1]:
        gap_start -= 1
        k = k - 1 if k else m - 1
    return gap_start


def move_left(rec_start, ref, alts, chrom_seq, chrom_seq_shift, min_start=None, skip_alleles=False):
    """
    Returns pair (new_start, new_alleles) if moved, and None otherwise.
    if skip_alleles, returns only the new start.
    """
    if min_start is None:
        min_start = chrom_seq_shift
    else:
        assert min_start >= chrom_seq_shift and min_start < rec_start

    # List of tuples (gap_start, gap_seq, new_start, is_insertion: bool)
    gaps = []
    ref_len = len(ref)
    shift = sys.maxsize

    for alt in alts:
        if ref.startswith(alt):
            assert ref != alt
            # Deletion
            gap_start = rec_start + len(alt)
            gap_seq = ref[len(alt):]
            is_insertion = False
        elif alt.startswith(ref):
            # Insertion
            gap_start = rec_start + ref_len
            gap_seq = alt[ref_len:]
            is_insertion = True
        else:
            # Neither deletion nor insertion
            return None

        new_start = _move_left_gap(gap_start, gap_seq, chrom_seq, chrom_seq_shift, min_start)
        if new_start == gap_start:
            # Could not move the start
            return None
        gaps.append((gap_start, gap_seq, is_insertion))
        shift = min(shift, gap_start - new_start)

    new_start = max(min_start, rec_start - shift)
    if skip_alleles:
        return new_start

    new_ref = chrom_seq[new_start - chrom_seq_shift : new_start + ref_len - chrom_seq_shift]
    alleles = [new_ref]
    for alt, (gap_start, gap_seq, is_insertion) in zip(alts, gaps):
        # Common prefix
        alleles.append(new_ref[:gap_start - rec_start])
        if is_insertion:
            gap_shift = len(gap_seq) - shift % len(gap_seq)
            alleles[-1] += gap_seq[gap_shift:] + gap_seq[:gap_shift]
    return new_start, alleles


@lru_cache(maxsize=None)
def all_gt_counts(n_alleles, cn):
    """
    all_gt_counts(n_alleles=3, cn=2) -> (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)
    all_gt_counts(n_alleles=2, cn=3) -> (3, 0),    (2, 1),    (1, 2),    (0, 3)
    """
    if n_alleles == 1:
        return ((cn,),)

    genotype = [0] * n_alleles
    genotype[0] = cn
    res = [tuple(genotype)]
    ix = 0
    remaining = 0
    n_alleles_2 = n_alleles - 2

    while True:
        while not genotype[ix]:
            if not ix:
                return tuple(res)
            genotype[ix + 1] = 0
            ix -= 1
        genotype[ix] -= 1
        remaining += 1
        genotype[ix + 1] = remaining
        res.append(tuple(genotype))
        if ix < n_alleles_2:
            ix += 1
            remaining = 0


def genotype_likelihoods(cn, allele_counts, error_prob=0.001, gt_counts=None):
    """
    Returns two lists of the same size: [gt_counts: tuple of ints], [likelihood: float].
    Returned likelihoods are normalized.

    allele_counts stores counts of reads that cover each allele, and other_count shows number of reads that
    do not correspond to any allele.
    """
    # TODO: use other_count.

    n_alleles = len(allele_counts)
    total_reads = sum(allele_counts)
    gt_counts = gt_counts or all_gt_counts(n_alleles, cn)
    likelihoods = np.zeros(len(gt_counts))

    for i, gt in enumerate(gt_counts):
        # Element-wise maximum. Use it to replace 0 with error probability.
        probs = np.maximum(gt, error_prob * cn)
        probs /= np.sum(probs)
        likelihoods[i] = multinomial.logpmf(allele_counts, total_reads, probs)
    likelihoods -= logsumexp(likelihoods)
    return gt_counts, likelihoods


@lru_cache(maxsize=None)
def _all_f_combinations(n_copies, cn):
    f_combinations = [(0,) * n_copies]
    for i in range(1, cn + 1):
        f_combinations.extend(all_gt_counts(n_copies, i))
    f_comb_ixs = { gt: i for i, gt in enumerate(f_combinations) }
    return np.array(f_combinations), f_comb_ixs


@lru_cache(maxsize=None)
class _PrecomputedData:
    def __init__(self, alleles, cn):
        assert cn > 0
        self.n_copies = len(alleles)
        self.n_alleles = max(alleles) + 1
        self.cn = cn

        self.poss_pscns = all_gt_counts(self.n_copies, cn)
        self.psv_genotypes = all_gt_counts(self.n_alleles, cn)
        self.n_psv_genotypes = len(self.psv_genotypes)
        self.f_combinations, f_comb_ixs = _all_f_combinations(self.n_copies, cn)

        self.poly_matrices = []
        self.f_ix_converts = []
        for sample_gt in self.poss_pscns:
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


def _fill_psv_gts(sample_id, cns, psv_infos, psv_counts, psv_start_ix, psv_end_ix, max_genotypes):
    for psv_ix in range(psv_start_ix, psv_end_ix):
        counts = psv_counts[psv_ix][sample_id]
        if counts.skip:
            continue

        psv_info = psv_infos[psv_ix]
        if psv_info.sample_infos[sample_id] is not None:
            continue
        psv_info.sample_infos[sample_id] = sample_info = SamplePsvInfo(cns[0])

        for sample_cn in cns:
            precomp_data = psv_info.precomp_datas.get(sample_cn)
            if precomp_data is None:
                precomp_data = _PrecomputedData(psv_info.allele_corresp, sample_cn)
                psv_info.precomp_datas[sample_cn] = precomp_data
            if len(precomp_data.poss_pscns) <= max_genotypes:
                _, probs = genotype_likelihoods(sample_cn, counts.allele_counts, gt_counts=precomp_data.psv_genotypes)
                sample_info.psv_gt_probs[sample_cn] = probs


def calculate_all_psv_gt_probs(region_group_extra, max_agcn, max_genotypes):
    psv_ixs = region_group_extra.region_group.psv_ixs
    if len(psv_ixs) == 0:
        return
    common.log('    Calculating PSV genotype probabilities')
    psv_counts = region_group_extra.psv_read_counts
    psv_searcher = region_group_extra.psv_searcher
    psv_infos = region_group_extra.psv_infos
    ref_cn = region_group_extra.region_group.cn

    n_psvs = len(psv_counts)
    n_samples = len(psv_counts[0])
    for sample_id in range(n_samples):
        for sample_const_region in region_group_extra.sample_const_regions[sample_id]:
            reg_start = sample_const_region.region1.start
            reg_end = sample_const_region.region1.end
            if sample_const_region.pred_cn is None or sample_const_region.pred_cn == 0 \
                    or sample_const_region.pred_cn > max_agcn:
                continue
            psv_start_ix, psv_end_ix = psv_searcher.contained_ixs(reg_start, reg_end)
            if psv_start_ix >= psv_end_ix:
                continue

            cns = []
            for weighted_cn in sample_const_region.probable_cns:
                if weighted_cn.agcn is not None and 0 < weighted_cn.agcn <= max_agcn:
                    cns.append(weighted_cn.agcn)
            _fill_psv_gts(sample_id, cns, psv_infos, psv_counts, psv_start_ix, psv_end_ix, max_genotypes)


def calculate_support_matrix(region_group_extra):
    """
    Calculates probabilities of sample genotypes according to all individual PSVs (if they have f-values).
    """
    if not region_group_extra.has_f_values:
        return

    psv_searcher = region_group_extra.psv_searcher
    psv_infos = region_group_extra.psv_infos
    n_psvs = len(psv_infos)
    n_samples = region_group_extra.n_samples

    f_values = region_group_extra.psv_f_values
    psv_exponents = region_group_extra.psv_infos
    psv_gt_coefs_cache = [{} for _ in range(n_psvs)]

    for sample_id in range(n_samples):
        for psv_ix in range(n_psvs):
            psv_info = psv_infos[psv_ix]
            sample_info = psv_info.sample_infos[sample_id]
            if np.isnan(f_values[psv_ix, 0]) or sample_info is None:
                continue

            for sample_cn, psv_gt_probs in sample_info.psv_gt_probs.items():
                if sample_cn in psv_gt_coefs_cache[psv_ix]:
                    curr_psv_gt_coefs = psv_gt_coefs_cache[psv_ix][sample_cn]
                else:
                    precomp_data = psv_info.precomp_datas[sample_cn]
                    f_pow_combs = precomp_data.f_power_combinations(f_values[psv_ix])
                    curr_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)
                    psv_gt_coefs_cache[psv_ix][sample_cn] = curr_psv_gt_coefs

                sample_info.support_rows[sample_cn] = support_row = np.zeros(len(curr_psv_gt_coefs))
                for sample_gt_ix, psv_gt_coefs in enumerate(curr_psv_gt_coefs):
                    support_row[sample_gt_ix] = logsumexp(psv_gt_coefs + psv_gt_probs)
                support_row *= psv_exponents[psv_ix].info_content


def _read_string(reader):
    length = ord(reader.read(1))
    return reader.read(length).decode()


class PsvStatus(Enum):
    Unreliable = 0
    SemiReliable = 1
    Reliable = 2

    @classmethod
    def from_str(cls, s):
        if s == 'u' or s == 'unreliable':
            return cls.Unreliable
        elif s == 's' or s == 'semi-reliable':
            return cls.SemiReliable
        elif s == 'r' or r == 'reliable':
            return cls.Reliable
        else:
            raise ValueError('Unexpected PSV status {}'.format(s))

    def __str__(self):
        if self == PsvStatus.Unreliable:
            return 'unreliable'
        elif self == PsvStatus.SemiReliable:
            return 'semi-reliable'
        elif self == PsvStatus.Reliable:
            return 'reliable'
        else:
            raise TypeError('Unexpected PSV status {!r}'.format(self))

    def short_str(self):
        if self == PsvStatus.Unreliable:
            return 'unrel'
        elif self == PsvStatus.SemiReliable:
            return 's-rel'
        elif self == PsvStatus.Reliable:
            return 'rel'
        else:
            raise TypeError('Unexpected PSV status {!r}'.format(self))



class _PsvPosAllele(namedtuple('_PsvPosAllele', ('psv_ix region strand allele_ix'))):
    def __lt__(self, other):
        return self.region.__lt__(other.region)

    def to_variant_pos(self, alleles):
        return duplication_.VariantPosition(self.region, self.strand, alleles[self.allele_ix])


class _PsvPos:
    def __init__(self, psv, genome, psv_ix, varcall_params):
        """
        psv_ix = index of the PSV across all PSVs.
        """
        self.psv_ix = psv_ix
        self.psv_record = psv
        self.alleles = tuple(psv.alleles)
        self.psv_positions = [_PsvPosAllele(psv_ix,
            Interval(genome.chrom_id(psv.chrom), psv.start, psv.start + len(self.alleles[0])), True, 0)]

        for pos2 in psv.info['pos2']:
            pos2 = pos2.split(':')
            allele_ix = int(pos2[3]) if len(pos2) == 4 else 1
            allele = self.alleles[allele_ix]
            chrom_id = genome.chrom_id(pos2[0])
            start = int(pos2[1]) - 1
            self.psv_positions.append(
                _PsvPosAllele(psv_ix, Interval(chrom_id, start, start + len(allele)), pos2[2] == '+', allele_ix))
        self._init_pos_weights(varcall_params)

        self.n_copies = len(self.psv_positions)
        self.ref_cn = self.n_copies * 2

        self.f_values = np.array(list(map(np.double, psv.info['fval'])))
        self.info_content = float(psv.info['info'])
        self.status = PsvStatus.from_str(psv.info['rel'])

    def _init_pos_weights(self, varcall_params):
        match_weight = np.log1p(-varcall_params.error_rate)
        mismatch_weight = np.log(varcall_params.error_rate)

        # If a read has allele i what would be "probability" of each homologous position.
        # Stores an array of natural log weights, size = (n_alleles x n_positions).
        self.pos_weights = []
        for allele_ix in range(len(self.alleles)):
            curr_weights = [match_weight if pos.allele_ix == allele_ix else mismatch_weight
                for pos in self.psv_positions]
            self.pos_weights.append(np.array(curr_weights))

    @property
    def start(self):
        return self.psv_positions[0].region.start

    @property
    def ref(self):
        return self.alleles[0]

    @property
    def alts(self):
        return self.alleles[1:]

    def skip(self):
        # TODO: Use more sophisticated filters.
        return self.status != PsvStatus.Reliable

    def weighted_positions(self, allele_ix):
        """
        Returns two lists:
            - PSV positions (_PsvPosAllele),
            - probababilities of each position given a read that supports allele_ix.
        """
        return self.psv_positions, self.pos_weights[allele_ix]


AlleleObservation = namedtuple('AlleleObservation', 'allele_ix is_first_mate ln_qual')


class _SkipPosition(Exception):
    def __init__(self, start):
        self.start = start


def strand_bias_test(forw_counts, rev_counts):
    """
    Find strand bias for each allele and returns an array with p-values.
    For each allele, run Fisher Exact test comparing forward/reverse counts against
    the sum forward/reverse counts for all other alleles.
    """
    n_alleles = len(forw_counts)
    forw_sum = np.sum(forw_counts)
    rev_sum = np.sum(rev_counts)
    total_sum = forw_sum + rev_sum

    pvals = np.ones(n_alleles)
    if forw_sum == 0 or rev_sum == 0:
        # One of the rows is 0 -- all p-values = 1.
        return pvals

    if n_alleles == 2:
        col_sum1 = forw_counts[0] + rev_counts[0]
        if col_sum1 == 0 or col_sum1 == total_sum:
            # One of the columns is 0 -- all p-values = 1.
            return pvals

        matrix = (forw_counts, rev_counts)
        pvals[:] = fisher_exact(matrix)[1]
        return pvals

    for i in range(n_alleles):
        forw_i = forw_counts[i]
        rev_i = rev_counts[i]
        col_sum_i = forw_i + rev_i
        if col_sum_i == 0 or col_sum_i == total_sum:
            # One of the columns is 0 -- current p-values = 1.
            continue

        matrix = ((forw_i, forw_sum - forw_i), (rev_i, rev_sum - rev_i))
        pvals[i] = fisher_exact(matrix)[1]
    return pvals


def _round_qual(qual):
    if qual <= 0.0:
        return 0
    log = int(np.floor(np.log10(qual)))
    return round(qual, max(0, 2 - log))


class VariantReadObservations:
    def __init__(self, start, alleles, n_samples):
        """
        n_samples is None when this is a shallow copy
        """
        self.start = start
        # Alleles from the binary input.
        self.tmp_alleles = alleles

        self.variant = None
        # List of duplications.VariantPosition
        self.variant_positions = None
        self.new_vcf_records = None
        # For each variant position (each repeat copy), stores array old_to_new,
        # where old_to_new[variant_allele] -> variant_allele on that repeat copy.
        self._new_vcf_allele_corresp = None
        # For each variant position, stores the index of the reference allele (old_to_new[self._ref_allele] == 0).
        self._ref_alleles = None

        if n_samples is not None:
            # list of dictionaries (one for each sample),
            # each dictionary: key = read hash (47 bits), value = AlleleObservation.
            self.observations = [{} for _ in range(n_samples)]
            self.other_observations = np.zeros(n_samples, dtype=np.int16)
        else:
            self.observations = None
            self.other_observations = None

        # Once initialized, this vector will store the following information for each sample:
        # Vector (n_copies) with log-probabilities that the genotype is 0/0 on that copy.
        self._homoz_00_probs = []

        # List of instances of VariantReadObservations (because there could be several PSVs within one variant).
        self.psv_observations = None
        # If variant overlaps a PSV, psv_paralog_priors will contain a matrix (n_copies x n_alleles) with
        # log-probabilities of observing corresponding allele on the corresponding copy.
        self.psv_paralog_priors = None
        self.psv_paralog_cache = None
        # Parent VariantReadObservations. Inverse of self.psv_observations.
        self.parent = None

    @property
    def is_shallow_copy(self):
        return id(self.observations) == id(self.parent.observations)

    @property
    def is_psv(self):
        return self.parent is not None

    @property
    def has_psvs(self):
        return bool(self.psv_observations)

    @classmethod
    def from_binary(cls, reader, byteorder, sample_conv, n_samples):
        start = reader.read(4)
        if not start:
            raise StopIteration
        start = int.from_bytes(start, byteorder)
        self = cls(start, None, n_samples)

        n_part_obs = int.from_bytes(reader.read(2), byteorder)
        for _ in range(n_part_obs):
            sample_id = sample_conv[int.from_bytes(reader.read(2), byteorder)]
            count = int.from_bytes(reader.read(2), byteorder)
            if sample_id is not None:
                self.other_observations[sample_id] = count

        while True:
            allele_ix = ord(reader.read(1))
            if allele_ix == 255:
                break
            if allele_ix == 254:
                raise _SkipPosition(self.start)

            sample_id = sample_conv[int.from_bytes(reader.read(2), byteorder)]
            read_hash = int.from_bytes(reader.read(8), byteorder)
            ln_qual = -0.04 * ord(reader.read(1))
            is_first_mate = read_hash & 1
            read_hash -= is_first_mate

            if sample_id is None:
                continue
            # Read with the same hash is present (most probably read mate, collisions should be extremely rare).
            if read_hash in self.observations[sample_id]:
                if self.observations[sample_id][read_hash].allele_ix != allele_ix:
                    del self.observations[sample_id][read_hash]
                    self.other_observations[sample_id] += 2
            else:
                self.observations[sample_id][read_hash] = AlleleObservation(allele_ix, bool(is_first_mate), ln_qual)

        n_alleles = ord(reader.read(1))
        alleles = [None] * n_alleles
        for i in range(n_alleles):
            alleles[i] = _read_string(reader)
        self.tmp_alleles = tuple(alleles)
        self.psv_observations = []
        return self

    @property
    def ref(self):
        if self.tmp_alleles is not None:
            return self.tmp_alleles[0]
        return self.variant.alleles[0]

    def _update_observations(self, allele_corresp):
        for sample_id, sample_observations in enumerate(self.observations):
            old_size = len(sample_observations)
            # list(...) to make a copy of an iterator before changing the dictionary.
            for read_hash, allele_obs in list(sample_observations.items()):
                new_allele = allele_corresp[allele_obs.allele_ix]
                if new_allele is None:
                    del sample_observations[read_hash]
                elif new_allele != allele_obs.allele_ix:
                    sample_observations[read_hash] = AlleleObservation(new_allele,
                        allele_obs.is_first_mate, allele_obs.ln_qual)
            # Count deleted hashes.
            self.other_observations[sample_id] += old_size - len(sample_observations)

    def set_variant(self, variant, dupl_pos_finder):
        """
        Sets VCF variant.
        This function filters a set of alleles, if needed.
        All reads that support excess alleles, are added to other_observations.
        """
        assert self.variant is None
        assert variant.start == self.start
        self.variant = variant
        self.variant_positions = dupl_pos_finder.find_variant_pos(variant)

        old_alleles = self.tmp_alleles
        self.tmp_alleles = None
        new_alleles = tuple(variant.alleles)
        if new_alleles != old_alleles:
            allele_corresp = self._simple_allele_corresp(old_alleles, new_alleles)
            self._update_observations(allele_corresp)

    def copy_obs(self):
        """
        Only copies observations and other_observations.
        """
        n_samples = len(self.observations)
        new = VariantReadObservations(self.start, None, n_samples)
        for sample_id in range(n_samples):
            new.other_observations[sample_id] = self.other_observations[sample_id]
            new.observations[sample_id] = self.observations[sample_id].copy()
        return new

    def shallow_copy_obs(self):
        new = VariantReadObservations(self.start, None, None)
        new.other_observations = self.other_observations
        new.observations = self.observations
        return new

    @staticmethod
    def _simple_allele_corresp(old_alleles, new_alleles):
        allele_corresp = []
        for allele in old_alleles:
            allele_corresp.append(new_alleles.index(allele) if allele in new_alleles else None)
        return tuple(allele_corresp)

    @staticmethod
    def _psv_allele_corresp(variant, psv):
        """
        Creates allele correspondence from variant alleles and PSV alleles
        Variant can have different positions than PSV, but should always contain it.
        """
        var_start = variant.start
        var_alleles = variant.alleles
        var_ref_allele = var_alleles[0]
        var_end = var_start + len(var_ref_allele)

        psv_start = psv.start
        psv_end = psv_start + len(psv.ref)
        psv_alleles = psv.alleles
        if psv_start < var_start or var_end < psv_end:
            common.log('WARNING: PSV {}:{} is larger than the variant {}:{}. Skipping the PSV'
                .format(variant.chrom, psv_start + 1, variant.chrom, var_start + 1))
            return None
        if var_start == psv_start and var_end == psv_end:
            return VariantReadObservations._simple_allele_corresp(var_alleles, psv_alleles)

        rel_start = psv_start - var_start
        rel_end = psv_end - var_start
        var_sub_alleles = [psv.ref]
        for i in range(len(var_alleles) - 1):
            cigar = Cigar(variant.info['CIGAR'][i].replace('M', '='))
            var_alt_allele = var_alleles[i + 1]
            assert cigar.ref_len == var_end - var_start and cigar.read_len == len(var_alt_allele)
            cigar.init_proxy_index()

            alt_size_diff = len(var_alt_allele) - len(var_ref_allele)
            start2, end2 = cigar.aligned_region(rel_start, rel_end, alt_size_diff)
            sub_allele = var_alt_allele[start2 : end2]
            var_sub_alleles.append(sub_allele)
        return VariantReadObservations._simple_allele_corresp(var_sub_alleles, psv_alleles)

    def set_psv(self, psv: _PsvPos, varcall_params):
        if len(self.variant_positions) != len(psv.psv_positions):
            common.log('WARNING: Failed to initialize PSV {}:{}. '.format(self.variant.chrom, psv.start + 1) +
                'Most likely, because of the boundary of a duplication')
            return None
        allele_corresp = self._psv_allele_corresp(self.variant, psv.psv_record)
        if allele_corresp is None:
            return None
        full_match = np.all(np.arange(len(self.variant.alleles)) == allele_corresp)
        self._update_psv_paralog_priors(psv, allele_corresp, varcall_params)

        if full_match:
            new_psv_obs = self.shallow_copy_obs()
        else:
            new_psv_obs = self.copy_obs()
            new_psv_obs._update_observations(allele_corresp)

        new_psv_obs.variant = psv
        new_psv_obs.variant_positions = [psv_pos.to_variant_pos(new_psv_obs.variant.alleles)
            for psv_pos in new_psv_obs.variant.psv_positions]
        new_psv_obs.parent = self
        self.psv_observations.append(new_psv_obs)
        return new_psv_obs

    def _update_psv_paralog_priors(self, psv, allele_corresp, varcall_params):
        n_copies = len(self.variant_positions)
        n_alleles = len(self.variant.alleles)
        priors_update = np.full((n_copies, n_alleles), -np.inf)

        for psv_i, psv_pos in enumerate(psv.psv_positions):
            for var_i, var_pos in enumerate(self.variant_positions):
                if var_pos.region.intersects(psv_pos.region):
                    break
            else:
                raise ValueError('PSV position {} has no matches with variant positions {}'
                    .format(psv_pos, self.variant_positions))
            # NOTE: We increase f-value to 0.5 so that the alternative allele will not get very high prior.
            fval = max(0.5, psv.f_values[psv_i])
            allele_match = [psv_allele_ix == psv_pos.allele_ix for psv_allele_ix in allele_corresp]
            n_match = sum(allele_match)
            if n_match == 0 or n_match == n_alleles:
                priors_update[var_i] = 0
            else:
                prior_match = np.log(fval / n_match)
                prior_mismatch = np.log((1 - fval) / (n_alleles - n_match))
                priors_update[var_i] = np.where(allele_match, prior_match, prior_mismatch)

        if self.psv_paralog_priors is None:
            self.psv_paralog_priors = priors_update
            self.psv_paralog_cache = {}
        else:
            self.psv_paralog_priors += priors_update
            self.psv_paralog_cache.clear()

        self.psv_paralog_priors = np.maximum(self.psv_paralog_priors, varcall_params.mutation_rate)
        self.psv_paralog_priors -= logsumexp(self.psv_paralog_priors, axis=1)[:, np.newaxis]

    def calculate_psv_paralog_priors(self, ext_pos_to_var_pos, paralog_genotypes, paralog_genotype_probs):
        cache = self.psv_paralog_cache.get(ext_pos_to_var_pos)
        if cache is None:
            n_copies = len(ext_pos_to_var_pos)
            n_alleles = self.psv_paralog_priors.shape[1]
            ext_paralog_priors = np.zeros((n_copies, n_alleles))
            for i, ext_pos in enumerate(ext_pos_to_var_pos):
                new_row = logsumexp(self.psv_paralog_priors[ext_pos, :], axis=0)
                ext_paralog_priors[i] = new_row - logsumexp(new_row)
            gt_cache = {}
            self.psv_paralog_cache[ext_pos_to_var_pos] = (ext_paralog_priors, gt_cache)
        else:
            ext_paralog_priors, gt_cache = cache

        for i, gt in enumerate(paralog_genotypes):
            gt_prob = gt_cache.get(gt.genotype)
            if gt_prob is None:
                gt_prob = gt.overl_psv_prior(ext_paralog_priors)
                gt_cache[gt.genotype] = gt_prob
            paralog_genotype_probs[i] = gt_prob

    def init_vcf_records(self, genome, vcf_headers):
        alleles = self.variant.alleles
        self.new_vcf_records = []
        self._new_vcf_allele_corresp = []
        self._ref_alleles = []

        for vcf_i, pos_i in itertools.product(range(2), range(len(self.variant_positions))):
            pos = self.variant_positions[pos_i]
            if vcf_i == 0 and pos.region.start != self.variant.start:
                continue
            vcf_header = vcf_headers[vcf_i]

            record = vcf_header.new_record()
            record.chrom = pos.region.chrom_name(genome)
            record.start = pos.region.start

            if pos.sequence == self.variant.ref:
                new_alleles = alleles
                old_to_new = np.arange(len(alleles))
            else:
                new_alleles = [pos.sequence]
                for allele_ix, allele in enumerate(alleles):
                    if allele != new_alleles[0]:
                        new_alleles.append(allele)
                old_to_new = np.array(self._simple_allele_corresp(alleles, new_alleles))
            self._new_vcf_allele_corresp.append(old_to_new)

            if vcf_i == 1:
                new_ref = new_alleles[0]
                if new_ref in alleles:
                    self._ref_alleles.append(alleles.index(new_ref))
                else:
                    self._ref_alleles.append(None)

            if not pos.strand:
                new_alleles = tuple(common.rev_comp(allele) for allele in new_alleles)
            record.alleles = new_alleles

            pos2_str = []
            for j, pos2 in enumerate(self.variant_positions):
                if pos_i == j:
                    continue
                pos2_str.append('{}:{}:{}'.format(pos2.region.chrom_name(genome), pos2.region.start_1,
                    '+' if pos.strand == pos2.strand else '-'))
            record.info['pos2'] = pos2_str
            record.info['overlPSV'] = 'T' if self.has_psvs else 'F'
            self.new_vcf_records.append(record)
        assert len(self.new_vcf_records) == len(self.variant_positions) + 1
        assert len(self.variant_positions) == len(self._ref_alleles)

    def update_vcf_records(self, gt_pred, genome):
        PHRED_THRESHOLD = 3

        var_pscn = gt_pred.variant_pscn
        sample_id = gt_pred.sample_id
        read_depth = np.sum(gt_pred.all_allele_counts) + self.other_observations[sample_id]

        for i, record in enumerate(self.new_vcf_records):
            old_to_new = self._new_vcf_allele_corresp[i]
            rec_fmt = record.samples[sample_id]
            pooled_gt = gt_pred.pooled_genotype
            pooled_gt_qual = gt_pred.pooled_genotype_qual
            gt_filter = gt_pred.filter

            if pooled_gt is not None:
                pooled_gt = pooled_gt.convert(old_to_new)
                if i == 0:
                    rec_fmt['GT'] = pooled_gt.to_tuple()
                    if any(gt_filter.map(VarFilter.need_qual0)):
                        rec_fmt['GQ'] = 0
                        rec_fmt['GQ0'] = pooled_gt_qual
                    else:
                        rec_fmt['GQ'] = pooled_gt_qual
                else:
                    rec_fmt['PGT'] = str(pooled_gt)
                    rec_fmt['PGQ'] = pooled_gt_qual

            # int() because pysam does not recognize numpy.[u]intN types.
            rec_fmt['DP'] = int(read_depth)
            curr_n_alleles = len(record.alleles)
            all_allele_counts = [0] * curr_n_alleles
            good_allele_counts = [0] * curr_n_alleles
            strand_allele_counts = [0] * (2 * curr_n_alleles)
            strand_phred = [0] * curr_n_alleles
            for old_allele_ix, new_allele_ix in enumerate(old_to_new):
                all_allele_counts[new_allele_ix] = int(gt_pred.all_allele_counts[old_allele_ix])
                good_allele_counts[new_allele_ix] = int(gt_pred.good_allele_counts[old_allele_ix])
                strand_allele_counts[2 * new_allele_ix] = int(gt_pred.forw_counts[old_allele_ix])
                strand_allele_counts[2 * new_allele_ix + 1] = int(gt_pred.rev_counts[old_allele_ix])
                strand_phred[new_allele_ix] = gt_pred.strand_phred[old_allele_ix]
            rec_fmt['AD'] = all_allele_counts
            rec_fmt['ADq'] = good_allele_counts
            rec_fmt['SB'] = strand_allele_counts
            rec_fmt['FS'] = strand_phred
            if gt_filter:
                rec_fmt['FILT'] = gt_filter.to_tuple()
            if i == 0:
                continue

            ext_copy_i = var_pscn.var_pos_to_ext_pos and var_pscn.var_pos_to_ext_pos[i - 1]
            paralog_gts, paralog_gt_probs = gt_pred.paralog_genotype_qualities(ext_copy_i)
            if paralog_gts is not None:
                best_paralog_gt_i = np.argmax(paralog_gt_probs)
                paralog_gt = paralog_gts[best_paralog_gt_i]
                paralog_gt_qual = int(common.extended_phred_qual(paralog_gt_probs,
                    best_paralog_gt_i, rem_prob=gt_pred.remaining_prob))
                self._update_homoz_00_probs(sample_id, i - 1, self._ref_alleles[i - 1], paralog_gts, paralog_gt_probs)
            else:
                paralog_gt = None
                paralog_gt_qual = 0

            if paralog_gt_qual >= PHRED_THRESHOLD:
                out_gt = None
                # When going from the extended repeat copy to the variant position,
                # there are two cases:
                # - either extended repeat copy has only copy,
                # - or the paralog-specific genotype is clear (only one allele present).
                if len(var_pscn.ext_pos_to_var_pos[ext_copy_i]) == 1:
                    out_gt = tuple(sorted(old_to_new[allele_ix] for allele_ix in paralog_gt))
                elif len(set(paralog_gt)) == 1:
                    out_gt = (old_to_new[paralog_gt[0]],) * 2
                    gt_filter = gt_filter.copy()
                    gt_filter.add(VarFilter.UnknownPsCN)
                    rec_fmt['FILT'] = gt_filter.to_tuple()
                if out_gt is not None:
                    rec_fmt['GT'] = out_gt
                    if any(gt_filter.map(VarFilter.need_qual0)):
                        rec_fmt['GQ'] = 0
                        rec_fmt['GQ0'] = paralog_gt_qual
                    else:
                        rec_fmt['GQ'] = paralog_gt_qual

    def _update_homoz_00_probs(self, sample_id, copy_i, ref_allele, paralog_gts, paralog_gt_probs):
        while len(self._homoz_00_probs) <= sample_id:
            self._homoz_00_probs.append(None)
        if self._homoz_00_probs[sample_id] is None:
            self._homoz_00_probs[sample_id] = np.full(len(self.variant_positions), -np.inf)

        if ref_allele is None:
            # Ref allele is not present in any genotypes. Probability is -np.inf.
            return
        for paralog_gt, paralog_gt_qual in zip(paralog_gts, paralog_gt_probs):
            if all(allele == ref_allele for allele in paralog_gt):
                self._homoz_00_probs[sample_id][copy_i] = paralog_gt_qual
                break

    def finalize_vcf_records(self):
        MAX_QUAL = 10000.0

        for i, record in enumerate(self.new_vcf_records):
            record.filter.add('PASS')
            if i == 0:
                # Pooled Quality
                if self.has_psvs:
                    record.qual = MAX_QUAL
                else:
                    record.qual = _round_qual(self.variant.qual)
            else:
                # Paralog-specific Quality
                ref_allele_prob = 0.0
                for sample_homoz_00_probs in self._homoz_00_probs:
                    if sample_homoz_00_probs is not None:
                        ref_allele_prob += sample_homoz_00_probs[i - 1]

                if ref_allele_prob == 0.0:
                    record.qual = 0
                else:
                    record.qual = _round_qual(min(MAX_QUAL, -10.0 * ref_allele_prob / common.LOG10))

            gt_allele_count = [0] * len(record.alleles)
            for fmt in record.samples.values():
                if 'GT' in fmt:
                    for allele in fmt['GT']:
                        gt_allele_count[allele] += 1
            total_called_alleles = sum(gt_allele_count)
            record.info['AN'] = total_called_alleles
            if total_called_alleles:
                record.info['AC'] = gt_allele_count[1:]
                record.info['AF'] = [round(count / total_called_alleles, 5) for count in gt_allele_count[1:]]

    def get_psv_usability(self, gt_pred, varcall_params, psv_gt_out, debug_out):
        """
        Returns True if the PSV can be used for paralog-specific variant calling.
        """
        variant = self.variant
        if debug_out:
            debug_out.write('PSV {}:{} [{}],  alleles {} (original {}, alleles {})\n'.format(
                variant.psv_record.chrom, variant.start + 1, variant.psv_ix, ' '.join(variant.alleles),
                variant.psv_record.start + 1, ' '.join(variant.psv_record.alleles)))

        psv_gt_out.write('{}\t{}\t{}\t'.format(gt_pred.sample, variant.start + 1, variant.status.short_str()))
        assert self.parent is not None
        if gt_pred.skip:
            psv_gt_out.write('*\t*\t*\t*\t*\t*\t*\tF\tskip\n')
            return False

        variant_pscn = gt_pred.variant_pscn
        if variant_pscn.ext_pos_cn is None:
            psv_gt_out.write('*\t*\t*\t*\t*\t*\t*\tF\tunknown_pscn\n')
            return False
        # TODO: What if there are filters? What if the quality is low?

        psv_gt_out.write('{}\t{}\t{}\t{}\t'.format(
            ','.join(map(str, variant_pscn.var_pos_to_ext_pos)),
            ','.join(map(str, variant_pscn.ext_pos_cn)),
            ','.join(map(str, gt_pred.good_allele_counts)),
            str(gt_pred.pooled_genotype)))

        n_ext_copies = len(variant_pscn.ext_pos_cn)
        ext_alleles = []
        for i in range(n_ext_copies):
            curr_alleles = set(np.where(variant_pscn.ext_allele_corresp[i])[0])
            ext_alleles.append(','.join(map(str, curr_alleles)))
        psv_gt_out.write(' '.join(ext_alleles))

        if len(set(gt_pred.pooled_genotype)) == 1:
            # Non-informative: all reads support the same allele.
            psv_gt_out.write('\t*\t*\tF\tdeficient_gt\n')
            return False
        elif np.any(np.sum(variant_pscn.ext_allele_corresp, axis=0) == n_ext_copies):
            # Skip PSV -- the same allele supports several extended copies.
            psv_gt_out.write('\t*\t*\tF\tambigous_allele\n')
            return False

        mcopy_gts, mcopy_gt_probs = psv_multi_copy_genotypes(
            gt_pred.start_pooled_genotypes, gt_pred.start_pooled_genotype_probs,
            variant_pscn.ext_f_values, variant_pscn.ext_allele_corresp, variant_pscn.ext_pos_cn)
        matches_ref = np.array([all(variant_pscn.ext_allele_corresp[copy_i, allele]
            for copy_i, subgt in enumerate(gt.genotype) for allele in subgt) for gt in mcopy_gts])

        nonref_prob = logsumexp(mcopy_gt_probs[~matches_ref])
        # abs to get rid of -0.
        ref_qual = abs(-10 * nonref_prob / common.LOG10)
        best_mcopy_gt = mcopy_gts[np.argmax(mcopy_gt_probs)]
        psv_gt_out.write('\t{}\t{:.0f}\t'.format(best_mcopy_gt, ref_qual))

        if ref_qual < varcall_params.psv_ref_gt:
            psv_gt_out.write('F\tnon_ref\n')
            return False
        psv_gt_out.write('T\n')
        return True

    def psv_update_read_probabilities(self, sample_id, read_positions):
        assert self.parent is not None
        for read_hash, allele_obs in self.observations[sample_id].items():
            read_positions[read_hash].add(self.variant, allele_obs)

    @staticmethod
    def create_vcf_headers(genome, argv, samples):
        vcf_headers = []
        for i in range(2):
            # First pooled, second un-pooled.
            vcf_header = pysam.VariantHeader()
            vcf_header.add_line('##command="{}"'.format(common.command_to_str(argv)))
            for name, length in genome.names_lengths():
                vcf_header.add_line('##contig=<ID={},length={}>'.format(name, length))

            vcf_header.add_line('##FILTER=<ID=Conflict,Description="Conflicting variant calls">')
            vcf_header.add_line('##INFO=<ID=pos2,Number=.,Type=String,Description="Second positions of the variant. '
                'Format: chrom:pos:strand">')
            vcf_header.add_line('##INFO=<ID=overlPSV,Number=1,Type=Character,'
                'Description="Variants overlaps a PSV. Possible values: T/F">')
            vcf_header.add_line('##INFO=<ID=AC,Number=A,Type=Integer,'
                'Description="Total number of alternate alleles in called genotypes">')
            vcf_header.add_line('##INFO=<ID=AN,Number=1,Type=Integer,'
                'Description="Total number of alleles in called genotypes">')
            vcf_header.add_line('##INFO=<ID=AF,Number=A,Type=Float,'
                'Description="Estimated allele frequency in the range [0,1]">')
            vcf_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
            vcf_header.add_line('##FORMAT=<ID=FILT,Number=.,Type=String,Description="Sample-specific filter">')
            vcf_header.add_line('##FORMAT=<ID=GTs,Number=.,Type=String,Description="Possible genotypes.">')
            vcf_header.add_line('##FORMAT=<ID=GQ,Number=1,Type=Float,Description="The Phred-scaled Genotype Quality">')
            vcf_header.add_line('##FORMAT=<ID=GQ0,Number=1,Type=Float,Description='
                '"Unedited genotype quality in case there is a sample-specific filter present."')
            vcf_header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">')
            vcf_header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,'
                'Description="Number of pooled observation for each allele">')
            vcf_header.add_line('##FORMAT=<ID=ADq,Number=R,Type=Integer,'
                'Description="Number of pooled observation for each allele that pass base quality filters">')
            vcf_header.add_line('##FORMAT=<ID=SB,Number=.,Type=Integer,'
                'Description="Number of pooled observations for each allele and strand '
                '(forward REF, reverse REF, forward ALT[1], reverse ALT[1], etc).">')
            vcf_header.add_line('##FORMAT=<ID=FS,Number=R,Type=Float,'
                'Description="Phred-scaled p-value that there is a strand bias">')
            if i == 1:
                vcf_header.add_line('##FORMAT=<ID=PGT,Number=1,Type=String,Description="Pooled Genotype">')
                vcf_header.add_line('##FORMAT=<ID=PGQ,Number=1,Type=Float,'
                    'Description="The Phred-scaled Pooled Genotype Quality">')
                vcf_header.add_line('##FORMAT=<ID=PGTs,Number=.,Type=String,Description="Possible pooled genotypes.">')
            for sample in samples:
                vcf_header.add_sample(sample)
            vcf_headers.append(vcf_header)
        return tuple(vcf_headers)


def _next_or_none(iter):
    try:
        return next(iter)
    except StopIteration:
        return None


def read_freebayes_results(ra_reader, samples, vcf_file, dupl_pos_finder):
    byteorder = 'big' if ord(ra_reader.read(1)) else 'little'
    n_samples = len(samples)
    n_in_samples = int.from_bytes(ra_reader.read(2), byteorder)

    # Converter between sample IDs sample_conv[sample_id from input] = sample_id in `samples`.
    sample_conv = [None] * n_in_samples
    for i in range(n_in_samples):
        sample = _read_string(ra_reader)
        sample_conv[i] = samples.id_or_none(sample)

    all_read_allele_obs = []
    vcf_record = _next_or_none(vcf_file)
    while vcf_record is not None:
        try:
            read_allele_obs = VariantReadObservations.from_binary(ra_reader, byteorder, sample_conv, n_samples)
            if read_allele_obs.start != vcf_record.start:
                # If read_allele_obs.start < vcf_record.start:
                #     There is an extra variant observation that is absent from the vcf file (too low quality).
                # Otherwise:
                #     There is a variant without any read-allele observations (no reads that cover the PSV?)
                continue
            read_allele_obs.set_variant(vcf_record, dupl_pos_finder)
            all_read_allele_obs.append(read_allele_obs)
            vcf_record = _next_or_none(vcf_file)
        except StopIteration:
            break
        except _SkipPosition as exc:
            if exc.start == vcf_record.start:
                vcf_record = _next_or_none(vcf_file)
    return all_read_allele_obs


def add_psv_variants(locus, all_read_allele_obs, psv_records, genome, varcall_params):
    searcher = itree.NonOverlTree(all_read_allele_obs, itree.start, itree.variant_end)
    n_psvs = [0] * len(PsvStatus)

    for psv_ix, psv in enumerate(psv_records):
        i, j = searcher.overlap_ixs(psv.start, psv.start + len(psv.ref))
        # There are no good observations of the PSV.
        if i == j or 'fval' not in psv.info:
            continue
        if i + 1 != j:
            common.log('ERROR: PSV {}:{} overlaps several Freebayes variants'.format(psv.chrom, psv.pos))
            common.log('')
            continue

        # There is exactly one corresponding variant.
        assert i + 1 == j
        psv = _PsvPos(psv, genome, psv_ix, varcall_params)
        n_psvs[psv.status.value] += 1
        curr_psv_obs = all_read_allele_obs[i].set_psv(psv, varcall_params)

    common.log('[{}] Use {} PSVs. Of them {} reliable and {} semi-reliable'
        .format(locus.name, sum(n_psvs), n_psvs[PsvStatus.Reliable.value], n_psvs[PsvStatus.SemiReliable.value]))

    min_qual = varcall_params.min_freebayes_qual
    if min_qual <= 0.0:
        return all_read_allele_obs
    filt_read_allele_obs = []
    for var_obs in all_read_allele_obs:
        if var_obs.has_psvs or var_obs.variant.qual >= min_qual:
            filt_read_allele_obs.append(var_obs)
    n_removed = len(all_read_allele_obs) - len(filt_read_allele_obs)
    if n_removed:
        common.log('[{}] Removed {} low-quality SNVs'.format(locus.name, n_removed))
    return filt_read_allele_obs

def vcf_record_key(genome):
    def inner(record):
        return (genome.chrom_id(record.chrom), record.start)
    return inner


def open_and_write_vcf(filename, header, records, tabix):
    gzip = filename.endswith('.gz')
    with pysam.VariantFile(filename, 'wz' if gzip else 'w', header=header) as vcf_file:
        for record in records:
            vcf_file.write(record)
    if gzip and tabix is not None and tabix != 'none':
        common.Process([tabix, '-p', 'vcf', filename]).finish()


def write_vcf_file(filenames, vcf_headers, all_read_allele_obs, genome, tabix):
    pooled_records = []
    records = []
    for variant_obs in all_read_allele_obs:
        variant_obs.finalize_vcf_records()
        pooled_records.append(variant_obs.new_vcf_records[0])
        records.extend(variant_obs.new_vcf_records[1:])

    record_key = vcf_record_key(genome)
    records.sort(key=record_key)
    records = merge_duplicates(records)
    open_and_write_vcf(filenames.out_vcf, vcf_headers[1], records, tabix)

    pooled_records.sort(key=record_key)
    pooled_records = merge_duplicates(pooled_records)
    open_and_write_vcf(filenames.out_pooled_vcf, vcf_headers[0], pooled_records, tabix)


class DuplPositionFinder:
    def __init__(self, duplications):
        self._chrom_id = duplications[0].region1.chrom_id
        self._dupl_tree = itree.create_interval_tree(duplications, itree.region1_start, itree.region1_end)
        self._cache = {}

    def find_read_pos(self, region):
        if self._chrom_id != region.chrom_id:
            return ()
        start = region.start
        end = region.end
        key = (start, end)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        first_region = Interval(self._chrom_id, start, end)
        res = [first_region]
        for entry in self._dupl_tree.overlap(start, end):
            dupl = entry.data
            region2 = dupl.aligned_region(first_region) # TODO: Add padding?
            res.append(region2)
        res.sort()
        res = tuple(res)
        self._cache[key] = res
        return res

    def find_variant_pos(self, variant):
        """
        Returns list of tuples VariantPosition.
        """
        ref_len = len(variant.ref)
        region = Interval(self._chrom_id, variant.start, variant.start + ref_len)
        res = [duplication_.VariantPosition(region, True, variant.ref)]

        for entry in self._dupl_tree.overlap(region.start, region.end):
            dupl = entry.data
            try:
                pos2 = dupl.align_variant(variant)
            except errors.VariantOutOfBounds:
                continue
            if pos2 is not None:
                res.append(pos2)
        res.sort()
        return tuple(res)


FORBIDDEN_LOC_PENALTY = -20.0 * common.LOG10


class _ReadEndPositions:
    def __init__(self):
        self.seq_len = None
        # If the read is mapped uniquely, unique_read_pos will have an Interval. Otherwise, it is None.
        self.unique_read_pos = None
        # tuple: If true location is unknown, but there are locations that are certainly incorrect, store it here.
        self.forbidden_read_pos = None
        # Pairs (_PsvPosAllele, log probability).
        self.psv_positions = []

    def set_read_positions(self, read_coord):
        self.seq_len = read_coord.seq_len
        self.unique_read_pos = read_coord.get_certainly_correct_location()
        self.forbidden_read_pos = read_coord.get_certainly_incorrect_locations()

    def add(self, psv_pos, allele_ix):
        if self.unique_read_pos is not None:
            return
        positions, weights = psv_pos.weighted_positions(allele_ix)
        weights = np.array(weights)
        weights -= logsumexp(weights)
        self.psv_positions.extend(zip(positions, weights))

    def calculate_possible_locations(self, read_len_summand=10):
        """
        read_len_summand: maximum distance between PSV = seq_len + read_len_summand.
        Returns two lists:
            - List of Intervals. Last entry may be None: this corresponds to an unknown location.
            - List of corresponding probabilities.
        """
        if self.unique_read_pos is not None:
            return (self.unique_read_pos,), np.array((0.0,))

        possible_locations = []
        possible_loc_probs = []
        if self.psv_positions:
            self._calculate_psv_clusters(possible_locations, possible_loc_probs, self.seq_len + read_len_summand)
        n_locs = len(possible_locations)

        if self.forbidden_read_pos:
            for region in self.forbidden_read_pos:
                found = False
                for i in range(n_locs):
                    if region.intersects(possible_locations[i]):
                        possible_loc_probs[i] += FORBIDDEN_LOC_PENALTY
                        found = True
                if not found:
                    possible_locations.append(region)
                    possible_loc_probs.append(FORBIDDEN_LOC_PENALTY)

        if n_locs == 0:
            # There are no overlapping PSVs, so we add a proxy location, that overlaps all non-forbidden locations
            possible_locations.append(None)
            possible_loc_probs.append(0.0)
        possible_loc_probs = np.array(possible_loc_probs)
        possible_loc_probs -= logsumexp(possible_loc_probs)
        return tuple(possible_locations), possible_loc_probs

    def _calculate_psv_clusters(self, possible_locations, possible_loc_probs, max_psv_dist):
        curr_psv_ixs = set()
        curr_prob = 0.0
        start_region = None
        end_region = None
        loc_n_psvs = []

        self.psv_positions.sort()
        for psv_pos, psv_pos_prob in self.psv_positions:
            if start_region is None or (start_region.distance(psv_pos.region) > max_psv_dist or
                    psv_pos.psv_ix in curr_psv_ixs):
                if start_region is not None:
                    possible_locations.append(Interval(start_region.chrom_id,
                        max(0, start_region.start - self.seq_len), end_region.end + self.seq_len))
                    possible_loc_probs.append(curr_prob)
                    loc_n_psvs.append(len(curr_psv_ixs))
                start_region = psv_pos.region
                curr_psv_ixs.clear()
                curr_prob = 0.0

            curr_psv_ixs.add(psv_pos.psv_ix)
            end_region = psv_pos.region
            curr_prob += psv_pos_prob

        possible_locations.append(Interval(start_region.chrom_id,
            max(0, start_region.start - self.seq_len), end_region.end + self.seq_len))
        possible_loc_probs.append(curr_prob)
        loc_n_psvs.append(len(curr_psv_ixs))

        if len(set(loc_n_psvs)) != 1:
            # print('Different PSVs')
            MISSING_PSV_PENALTY = -5 * common.LOG10
            assert len(loc_n_psvs) == len(possible_locations)
            max_psvs = max(loc_n_psvs)
            for i, n_psvs in enumerate(loc_n_psvs):
                possible_loc_probs[i] += MISSING_PSV_PENALTY * (max_psvs - n_psvs)

    def debug(self, genome, out):
        if self.unique_read_pos:
            out.write('            Unique location: {}\n'.format(self.unique_read_pos.to_str(genome)))
        if self.forbidden_read_pos:
            out.write('            Forbidden locations:\n')
            for loc in self.forbidden_read_pos:
                out.write('                {}\n'.format(loc.to_str(genome)))
        if self.psv_positions:
            out.write('            PSVs:\n')
        for psv_pos, psv_pos_prob in self.psv_positions:
            out.write('                {}  psv {} allele {} = {:.3f}\n'.format(psv_pos.region.to_str_comma(genome),
                psv_pos.psv_ix, psv_pos.allele_ix, np.exp(psv_pos_prob)))


class _ReadPositions:
    def __init__(self):
        # Read positions for the second and second first.
        self._mate_read_pos = (_ReadEndPositions(), _ReadEndPositions())
        self._is_reverse = [None, None]
        self.requires_mate = True

        self.positions1 = None
        self.probs1 = None
        self.positions2 = None
        self.probs2 = None

    def set_read_positions(self, read_coord):
        self.requires_mate = read_coord.is_paired
        is_read1 = bool(read_coord.read_hash & np.uint8(1))
        self._mate_read_pos[is_read1].set_read_positions(read_coord)
        self._is_reverse[is_read1] = read_coord.is_reverse
        if self._is_reverse[1 - is_read1] is None:
            self._is_reverse[1 - is_read1] = not read_coord.is_reverse

    def add(self, psv_pos: _PsvPos, allele_obs: AlleleObservation):
        self._mate_read_pos[allele_obs.is_first_mate].add(psv_pos, allele_obs.allele_ix)

    def init_paired_read_pos_probs(self, max_mate_dist, no_mate_penalty):
        # NOTE: What happens if there is no read mate?
        self.positions1, self.probs1 = self._mate_read_pos[1].calculate_possible_locations()
        self.positions2, self.probs2 = self._mate_read_pos[0].calculate_possible_locations()

        if len(self.positions1) == 1 and self.positions1[0] is None \
                and all(pos2 is not None for pos2 in self.positions2):
            self.probs2 -= logsumexp(self.probs2)
            self.positions1, self.probs1 = self._init_second_positions(self.positions2, self.probs2, max_mate_dist)
            return
        if len(self.positions2) == 1 and self.positions2[0] is None \
                and all(pos1 is not None for pos1 in self.positions1):
            self.probs1 -= logsumexp(self.probs1)
            self.positions2, self.probs2 = self._init_second_positions(self.positions1, self.probs1, max_mate_dist)
            return

        assert self.positions1 and self.positions2
        new_probs1 = self.probs1 + no_mate_penalty
        new_probs2 = self.probs2 + no_mate_penalty
        n = len(self.positions1)
        m = len(self.positions2)
        for i in range(n):
            pos1 = self.positions1[i]
            for j in range(m):
                pos2 = self.positions2[j]
                if pos1 is None or pos2 is None or pos1.distance(pos2) <= max_mate_dist:
                    new_prob = self.probs1[i] + self.probs2[j]
                    new_probs1[i] = max(new_probs1[i], new_prob)
                    new_probs2[j] = max(new_probs2[j], new_prob)

        self.probs1 = new_probs1 - logsumexp(new_probs1)
        self.probs2 = new_probs2 - logsumexp(new_probs2)

    def _init_second_positions(self, positions1, probs1, max_mate_dist):
        positions2 = []
        for (pos1, prob1) in zip(positions1, probs1):
            pos2 = Interval(pos1.chrom_id, max(0, pos1.start - max_mate_dist), pos1.end + max_mate_dist)
            # This is a hack, but it will allow us to efficiently sort both intervals and probabilities.
            pos2.prob = prob1
            positions2.append(pos2)
        positions2.sort()

        upd_pos2 = []
        upd_probs2 = []
        for subregion, ixs in Interval.get_disjoint_subregions(positions2):
            upd_pos2.append(subregion)
            upd_probs2.append(positions2[ixs[0]].prob if len(ixs) == 1 else max(positions2[i].prob for i in ixs))
        upd_probs2 = np.array(upd_probs2)
        upd_probs2 -= logsumexp(upd_probs2)
        return tuple(upd_pos2), upd_probs2

    def get_mate_pos_probs(self, is_read1):
        if is_read1:
            return self.positions1, self.probs1
        return self.positions2, self.probs2

    def is_reverse(self, is_read1):
        return self._is_reverse[is_read1]

    def debug(self, genome, out):
        out.write('      * First mate ({}):\n'.format('R' if self._is_reverse[1] else 'F'))
        self._mate_read_pos[1].debug(genome, out)
        for pos1, prob1 in zip(self.positions1, self.probs1):
            out.write('        {:30} {:7.3f} ({:.5f})\n'
                .format('ANY' if pos1 is None else pos1.to_str(genome), prob1 / common.LOG10, np.exp(prob1)))

        out.write('      * Second mate ({}):\n'.format('R' if self._is_reverse[0] else 'F'))
        self._mate_read_pos[0].debug(genome, out)
        for pos2, prob2 in zip(self.positions2, self.probs2):
            out.write('        {:30} {:7.3f} ({:.5f})\n'
                .format('ANY' if pos2 is None else pos2.to_str(genome), prob2 / common.LOG10, np.exp(prob2)))


class ReadCollection:
    def __init__(self, sample_id, sample, coord_index):
        CLEAR_LAST_BIT = ~np.uint64(1)

        self.sample_id = sample_id
        self.sample = sample
        self.read_positions = defaultdict(_ReadPositions)
        coordinates = coord_index.load(sample_id)

        sum_length = 0
        for read_coord in coordinates.values():
            sum_length += read_coord.seq_len
            read_hash = read_coord.read_hash & CLEAR_LAST_BIT
            self.read_positions[read_hash].set_read_positions(read_coord)
        self.mean_len = int(round(sum_length / len(coordinates)))

    def add_psv_observations(self, psv_read_allele_obs, max_mate_dist, no_mate_penalty):
        for curr_allele_obs in psv_read_allele_obs:
            curr_allele_obs.psv_update_read_probabilities(self.sample_id, self.read_positions)
        for read_pos in self.read_positions.values():
        # for read_hash, read_pos in self.read_positions.items():
            # print('Calculate single mate pos probs: {}'.format(read_hash))
            read_pos.init_paired_read_pos_probs(max_mate_dist, no_mate_penalty)

    def debug_read_probs(self, genome, out):
        out.write('Sample {}\n'.format(self.sample))
        for read_hash, read_pos in self.read_positions.items():
            out.write('    Hash = {}\n'.format(read_hash))
            read_pos.debug(genome, out)


class VarFilter(Enum):
    Pass = 0
    UnknownAgCN   = 10
    UnknownPsCN   = 11
    LowReadDepth  = 13
    HighReadDepth = 14
    StrandBias    = 15

    def __str__(self):
        if self == VarFilter.Pass:
            return 'PASS'
        return self.name

    def need_qual0(self):
        return self == VarFilter.UnknownAgCN


@lru_cache(maxsize=None)
def log_multinomial_coeff(gt):
    return gammaln(len(gt) + 1) - gammaln(np.bincount(gt) + 1).sum()


class PooledGT:
    def __init__(self, tup):
        self._tup = tuple(tup)

    def __len__(self):
        return len(self._tup)

    def __getitem__(self, i):
        return self._tup.__getitem__(i)

    # def __iter__(self):
    #     return self._tup.__iter__()

    def __eq__(self, oth):
        return self._tup == oth._tup

    def __hash__(self):
        return self._tup.__hash__()

    def __str__(self):
        return '/'.join(map(str, self._tup))

    def to_tuple(self):
        return self._tup

    def convert(self, old_to_new):
        return PooledGT(sorted(old_to_new[allele_ix] for allele_ix in self._tup))


class MultiCopyGT:
    def __init__(self, gt, pscn, pooled_gt=None):
        if pooled_gt is None:
            self.pooled_genotype = PooledGT(sorted(gt))
        else:
            assert isinstance(pooled_gt, PooledGT)
            self.pooled_genotype = pooled_gt

        self.genotype = []
        shift = 0
        for i, cn in enumerate(pscn):
            self.genotype.append(tuple(sorted(gt[shift : shift + cn])))
            shift += cn
        self.genotype = tuple(self.genotype)
        self.n_copies = len(self.genotype)

        # Matrix (n_alleles x n_copies) to calculate read observation probability given this genotype.
        # This does not account for different location probabilities for a read.
        self.precomp_read_obs_probs = None

    def precompute_read_obs_probs(self, n_alleles, error_rate):
        self.precomp_read_obs_probs = np.zeros((n_alleles, self.n_copies))
        for copy_i, part_gt in enumerate(self.genotype):
            part_gt_len = len(part_gt)
            for allele_j in range(n_alleles):
                n_matches = part_gt.count(allele_j)

                self.precomp_read_obs_probs[allele_j, copy_i] = np.log(
                    (n_matches * (1 - error_rate) + (part_gt_len - n_matches) * error_rate) / part_gt_len)

    def calc_read_obs_prob(self, copy_probabilities, read_allele):
        return logsumexp(self.precomp_read_obs_probs[read_allele] + copy_probabilities)

    def __str__(self):
        return '_'.join('/'.join(map(str, gt)) for gt in self.genotype)

    def __eq__(self, oth):
        return self.genotype == oth.genotype

    def __hash__(self):
        return self.genotype.__hash__()

    def no_psv_prior(self, penalty):
        """
        Returns prior for multi-copy genotype that does not overlap PSV.
        For each extended copy, add penalty (log) for each non-reference allele present.
        1/1 and 0/1 will lead to the same penalty.
        """
        res = 0
        for gt in self.genotype:
            gt = set(gt)
            res += len(gt) - (0 in gt)
        return res * penalty

    def overl_psv_prior(self, paralog_priors):
        res = 0
        for i, gt in enumerate(self.genotype):
            res += log_multinomial_coeff(gt) + np.sum(paralog_priors[i, gt])
        return res

    @classmethod
    def create_all(cls, pooled_gt, pscn):
        if len(set(pooled_gt)) == 1:
            yield MultiCopyGT(pooled_gt, pscn, pooled_gt)
            return

        used_genotypes = set()
        for gt in set(itertools.permutations(pooled_gt)):
            new = MultiCopyGT(gt, pscn, pooled_gt)
            if new.genotype not in used_genotypes:
                used_genotypes.add(new.genotype)
                yield new

    @staticmethod
    def aggregate_genotypes(paralog_genotypes, paralog_genotype_probs, ext_copy_i):
        """
        Returns aggregated genotypes and their probabilities.
        If ext_copy_i is None -- aggregates by pooled genotype, otherwise aggregates by copy i genotypes.
        """
        gt_dict = defaultdict(list)
        for paralog_gt, paralog_gt_prob in zip(paralog_genotypes, paralog_genotype_probs):
            key = paralog_gt.pooled_genotype if ext_copy_i is None else paralog_gt.genotype[ext_copy_i]
            gt_dict[key].append(paralog_gt_prob)

        n = len(gt_dict)
        res_genotypes = [None] * n
        res_genotype_probs = np.zeros(n)
        i = 0
        for gt, gt_probs in gt_dict.items():
            res_genotypes[i] = gt
            res_genotype_probs[i] = logsumexp(gt_probs)
            i += 1
        return res_genotypes, res_genotype_probs


def psv_multi_copy_genotypes(pooled_gts, pooled_gt_probs, fvals, allele_corresp, pscn):
    log_fvals = np.log(fvals)
    log_not_fvals = np.log1p(-fvals)
    mcopy_gts = []
    mcopy_gt_probs = []

    for pooled_gt, pooled_gt_prob in zip(pooled_gts, pooled_gt_probs):
        for mcopy_gt in MultiCopyGT.create_all(pooled_gt, pscn):
            curr_prob = pooled_gt_prob
            for copy, copy_gt in enumerate(mcopy_gt.genotype):
                for allele in copy_gt:
                    if allele_corresp[copy, allele]:
                        curr_prob += log_fvals[copy]
                    else:
                        curr_prob += log_not_fvals[copy]
            mcopy_gts.append(mcopy_gt)
            mcopy_gt_probs.append(curr_prob)
    mcopy_gt_probs = np.array(mcopy_gt_probs)
    mcopy_gt_probs -= logsumexp(mcopy_gt_probs)
    return mcopy_gts, mcopy_gt_probs


def _genotype_probs_to_str(genotypes, probs):
    """
    Returns index of the best genotype and string representation of the genotypes and their probabilities.
    """
    MIN_PROB = -50 * common.LOG10
    ixs = np.argsort(-probs)
    s = ' '.join('{}={:.3g}'.format(genotypes[i], abs(probs[i] / common.LOG10)) for i in ixs if probs[i] >= MIN_PROB)
    return ixs[0], s


def _calculate_allele_probs(alleles, read_len):
    """
    Calculate probabilities of completely covering each allele based on the allele length and read length.

    For an allele of length `m` and read length `l` there are `l - m - 1` possible read position that would
    completely cover the allele (one base on each side of the allele).
    Next, allele probabilities are normalized in a way that the largest probability will be 1.
    Note, that the resulting array does not sum up to one.
    """
    allele_probs = np.zeros(len(alleles))
    for i, allele in enumerate(alleles):
        allele_probs[i] = max(0, read_len - len(allele) - 1)
    m = allele_probs.max()
    if m == 0.0:
        raise RuntimeError('All alleles are longer than the mean read length')
    allele_probs /= m
    return allele_probs


class VariantParalogCN:
    def __init__(self, variant_obs, sample_id, cn_profiles):
        self.cn_estimate = None
        # Variant genotype may have a different set of copies than the CN estimate,
        # because we merge all copies with unknown psCN.
        # New set of region is called "extended positions", and there are several lists, associated with them:
        # List of indices to go from CN estimate regions to the extended positions.
        self.cn_est_pos_to_ext_pos = None
        # List of tuples of indices: [(index, ...)], inverse of cn_est_pos_to_ext_pos.
        self.ext_pos_to_cn_est = None
        # List of integers: CN estimate for each extended position.
        self.ext_pos_cn = None
        # List of indices to go from variant position to an extended position.
        self.var_pos_to_ext_pos = None
        # List of tuples of indices: [(index, ...)], inverse of var_pos_to_ext_pos.
        self.ext_pos_to_var_pos = None

        # Matrix of booleans (n_ext_copies, n_alleles). If element is True, i-th extended copy has j-th allele.
        self.ext_allele_corresp = None
        # Average f-values across various extended repeat copies.
        self.ext_f_values = None

        for var_pos in variant_obs.variant_positions:
            cn_estimates = list(cn_profiles.cn_estimates(sample_id, var_pos.region))
            if cn_estimates:
                self._match_var_pos_to_cn_estimates(variant_obs, cn_estimates)
                break

        self.unknown_cn = False
        if self.cn_estimate is None or self.var_pos_to_ext_pos is None or None in self.var_pos_to_ext_pos:
            self.unknown_cn = True

        if not self.unknown_cn and variant_obs.is_psv:
            self._set_psv_fields(variant_obs.variant)

    def debug(self):
        print('Variant psCN:')
        print('    Unknown CN? {}'.format(self.unknown_cn))
        print('    CN est pos -> ext pos:    {}'.format(self.cn_est_pos_to_ext_pos))
        print('    ext pos    -> CN est pos: {}'.format(self.ext_pos_to_cn_est))
        print('    ext pos CN: {}'.format(self.ext_pos_cn))
        print('    var pos    -> ext pos:    {}'.format(self.var_pos_to_ext_pos))
        print('    ext pos    -> var pos:    {}'.format(self.ext_pos_to_var_pos))

    def _match_var_pos_to_cn_estimates(self, variant_obs, cn_estimates):
        if len(cn_estimates) > 1:
            assert len(cn_estimates) == 2
            est_a = cn_estimates[0]
            est_b = cn_estimates[1]
            if est_a.pred_cn != est_b.pred_cn or est_a.paralog_cn != est_b.paralog_cn:
                self.unknown_cn = True
                return
        self.cn_estimate = est = cn_estimates[0]

        # TODO: What to do with qualities and agCN/psCN filters?
        agcn = est.pred_cn
        if agcn is None:
            self.unknown_cn = True
            return

        pscn = np.array(est.paralog_cn)
        pscn_known = np.array([cn is not None for cn in pscn])
        n_copies = len(pscn)
        n_known = sum(pscn_known)
        if n_known == 0:
            self.unknown_cn = True
            return

        cn_regions = [est.sample_const_region.region1]
        if est.sample_const_region.regions2 is not None:
            cn_regions.extend(region for region, _ in est.sample_const_region.regions2)
        self.ext_pos_to_cn_est = []
        self.ext_pos_cn = []
        self.cn_est_pos_to_ext_pos = []

        if n_known < n_copies:
            self.ext_pos_to_cn_est.append([])
            self.ext_pos_cn.append(agcn - np.sum(pscn[pscn_known]))

        for i in range(n_copies):
            if pscn_known[i]:
                self.cn_est_pos_to_ext_pos.append(len(self.ext_pos_cn))
                self.ext_pos_to_cn_est.append((i,))
                self.ext_pos_cn.append(pscn[i])
            else:
                self.cn_est_pos_to_ext_pos.append(0)
                self.ext_pos_to_cn_est[0].append(i)
        self.ext_pos_to_cn_est[0] = tuple(self.ext_pos_to_cn_est[0])
        self.ext_pos_to_cn_est = tuple(self.ext_pos_to_cn_est)

        self.var_pos_to_ext_pos = [None] * len(variant_obs.variant_positions)
        self.ext_pos_to_var_pos = [[] for _ in range(len(self.ext_pos_cn))]
        for i, var_pos in enumerate(variant_obs.variant_positions):
            for j, cn_region in enumerate(cn_regions):
                if cn_region.intersects(var_pos.region):
                    ext_pos_i = self.cn_est_pos_to_ext_pos[j]
                    self.var_pos_to_ext_pos[i] = ext_pos_i
                    self.ext_pos_to_var_pos[ext_pos_i].append(i)
                    break
        self.ext_pos_to_var_pos = tuple(map(tuple, self.ext_pos_to_var_pos))

    def _set_psv_fields(self, psv):
        n_alleles = len(psv.alleles)
        n_ext_copies = len(self.ext_pos_cn)
        self.ext_allele_corresp = np.zeros((n_ext_copies, n_alleles), dtype=np.bool)
        self.ext_f_values = np.zeros(n_ext_copies)
        for var_i, ext_copy_i in enumerate(self.var_pos_to_ext_pos):
            allele_ix = psv.psv_positions[var_i].allele_ix
            self.ext_allele_corresp[ext_copy_i, allele_ix] = True
            self.ext_f_values[ext_copy_i] += psv.f_values[var_i]
        # NOTE: Here, we take mean f-value if needed. Alternatively, we can minimal f-value.
        self.ext_f_values /= np.bincount(self.var_pos_to_ext_pos)


class VariantGenotypePred:
    def __init__(self, sample_id, sample, variant_obs, cn_profiles):
        self.sample_id = sample_id
        self.sample = sample
        self.variant_obs = variant_obs
        self.variant_pscn = VariantParalogCN(variant_obs, sample_id, cn_profiles)
        self.filter = Filters()

        if self.variant_pscn.cn_estimate is None:
            self.filter.add(VarFilter.UnknownAgCN)
        elif self.variant_pscn.var_pos_to_ext_pos is None or None in self.variant_pscn.var_pos_to_ext_pos:
            self.filter.add(VarFilter.UnknownPsCN)
        self.skip = False

        # List of pairs (_ReadPositions, AlleleObservation).
        self.read_obs = None
        n_alleles = len(self.variant_obs.variant.alleles)
        self.all_allele_counts = np.zeros(n_alleles, dtype=np.uint16)
        self.good_allele_counts = np.zeros(n_alleles, dtype=np.uint16)
        # Forward/Reverse strand read counts for each allele.
        self.forw_counts = np.zeros(n_alleles, dtype=np.uint16)
        self.rev_counts = np.zeros(n_alleles, dtype=np.uint16)
        self.strand_phred = None

        self.pooled_genotype = None
        self.pooled_genotype_qual = None
        self.paralog_genotypes = None
        self.paralog_genotype_probs = None

        self.start_pooled_genotypes = None
        self.start_pooled_genotype_probs = None
        self.remaining_prob = None

    def init_read_counts(self, all_read_positions, varcall_params, debug_out):
        variant = self.variant_obs.variant
        write_debug = debug_out is not None
        if write_debug:
            debug_out.write('Add reads to variant {}:{}\n'.format(variant.chrom, variant.pos))

        self.read_obs = []
        for read_hash, allele_obs in self.variant_obs.observations[self.sample_id].items():
            allele_ix = allele_obs.allele_ix
            self.all_allele_counts[allele_ix] += 1
            if write_debug:
                debug_out.write('    Add read {:21}   {}  {} mate,  basequal {:.0f}\n'.format(
                    read_hash, allele_ix, 2 - allele_obs.is_first_mate, -10 * allele_obs.ln_qual / common.LOG10))

            if allele_obs.ln_qual <= varcall_params.base_log_qual:
                self.good_allele_counts[allele_ix] += 1
                paired_read_position = all_read_positions[read_hash]
                if paired_read_position.is_reverse(allele_obs.is_first_mate):
                    self.rev_counts[allele_ix] += 1
                else:
                    self.forw_counts[allele_ix] += 1
                self.read_obs.append((paired_read_position, allele_obs))

        read_depth = len(self.read_obs)
        if read_depth < varcall_params.min_read_depth:
            self.filter.add(VarFilter.LowReadDepth)
            self.skip = True
        elif read_depth > varcall_params.max_read_depth:
            self.filter.add(VarFilter.HighReadDepth)
            self.skip = True

        strand_pvals = strand_bias_test(self.forw_counts, self.rev_counts)
        self.strand_phred = -10 * np.log10(strand_pvals)
        if np.max(self.strand_phred) >= varcall_params.max_strand_bias:
            self.filter.add(VarFilter.StrandBias)
        # abs to get rid of -0.
        self.strand_phred = np.abs(np.floor(self.strand_phred))

    def init_genotypes(self, varcall_params, mean_read_len, out, only_pooled=False):
        if self.skip:
            return

        best_gt_ix = self._init_pooled_genotypes(varcall_params, mean_read_len, out)
        threshold = min(varcall_params.pooled_gt_thresh, self.start_pooled_genotype_probs[best_gt_ix])
        ixs_remv = np.where(self.start_pooled_genotype_probs < threshold)[0]
        if len(ixs_remv) == 0:
            filt_pooled_genotypes = self.start_pooled_genotypes
            self.remaining_prob = -np.inf
        else:
            ixs_keep = np.where(self.start_pooled_genotype_probs >= threshold)[0]
            filt_pooled_genotypes = (self.start_pooled_genotypes[i] for i in ixs_keep)
            self.remaining_prob = logsumexp(self.start_pooled_genotype_probs[ixs_remv])

        if self.variant_pscn.cn_estimate is None or self.variant_pscn.ext_pos_cn is None or only_pooled:
            return
        self.paralog_genotypes = []
        for pooled_gt in filt_pooled_genotypes:
            self.paralog_genotypes.extend(MultiCopyGT.create_all(pooled_gt, self.variant_pscn.ext_pos_cn))
        self._calculate_paralog_priors(varcall_params, out)

    def _init_pooled_genotypes(self, varcall_params, mean_read_len, out):
        '''
        If return_best: return
            - pooled genotypes that pass a threshold,
            - sum probability of the removed genotypes.
        '''
        sample_cn = 2 if self.variant_pscn.cn_estimate is None else self.variant_pscn.cn_estimate.pred_cn
        variant = self.variant_obs.variant
        allele_probs = _calculate_allele_probs(variant.alleles, mean_read_len)
        pooled_gt_counts = all_gt_counts(len(variant.alleles), sample_cn)

        scaled_pooled_gt_counts = [allele_probs * gt_counts for gt_counts in pooled_gt_counts]
        _, self.start_pooled_genotype_probs = genotype_likelihoods(sample_cn, self.good_allele_counts,
            error_prob=varcall_params.error_rate, gt_counts=scaled_pooled_gt_counts)
        n_copies_range = np.arange(len(pooled_gt_counts[0]))
        self.start_pooled_genotypes = [PooledGT(np.repeat(n_copies_range, gt_counts))
            for gt_counts in pooled_gt_counts]

        best_i, pooled_gt_str = _genotype_probs_to_str(self.start_pooled_genotypes, self.start_pooled_genotype_probs)
        if out:
            out.write('{}\t{}\tpooled_init\t{}\n'.format(self.sample, variant.start + 1, pooled_gt_str))

        self.pooled_genotype = self.start_pooled_genotypes[best_i]
        self.pooled_genotype_qual = int(common.phred_qual(self.start_pooled_genotype_probs, best_i))
        return best_i

    def _calculate_paralog_priors(self, varcall_params, out):
        self.paralog_genotype_probs = np.zeros(len(self.paralog_genotypes))
        if len(self.paralog_genotypes) < 2:
            return

        if self.variant_obs.psv_paralog_priors is None:
            for i, gt in enumerate(self.paralog_genotypes):
                self.paralog_genotype_probs[i] = gt.no_psv_prior(varcall_params.mutation_rate)
        else:
            self.variant_obs.calculate_psv_paralog_priors(self.variant_pscn.ext_pos_to_var_pos,
                self.paralog_genotypes, self.paralog_genotype_probs)

        self.paralog_genotype_probs -= logsumexp(self.paralog_genotype_probs)
        _, paralog_priors_str = _genotype_probs_to_str(self.paralog_genotypes, self.paralog_genotype_probs)
        out.write('{}\t{}\tparalog_priors\t{}\n'.format(self.sample, self.variant_obs.variant.pos, paralog_priors_str))

    def paralog_genotype_qualities(self, ext_copy_i):
        """
        Returns possible marginal paralog-specific genotypes for the extended copy `i` and their qualities.
        """
        if self.paralog_genotypes is None or ext_copy_i is None:
            return None, None
        return MultiCopyGT.aggregate_genotypes(self.paralog_genotypes, self.paralog_genotype_probs, ext_copy_i)

    def var_get_shared_pos2(self, ext_copy_i, pos2_str, genome):
        assert ext_copy_i == 0
        ext_variant_regions = tuple(self.variant_obs.variant_positions[i].region
            for i in self.variant_pscn.ext_pos_to_var_pos[ext_copy_i])
        res = []

        for i, pos2 in enumerate(pos2_str):
            pos2 = pos2.split(':')
            pos2_chrom = genome.chrom_id(pos2[0])
            pos2_start = int(pos2[1]) - 1

            for var_reg in ext_variant_regions:
                if var_reg.chrom_id == pos2_chrom and var_reg.start == pos2_start:
                    res.append(i)
                    break
        assert len(res) == len(ext_variant_regions) - 1
        return tuple(res)

    def _add_read(self, curr_read_positions, allele_obs, n_copies, log_pscn_frac):
        read_pos, read_pos_probs = curr_read_positions.get_mate_pos_probs(allele_obs.is_first_mate)
        n_read_pos = len(read_pos)
        ext_read_pos_probs = np.full(n_copies, -np.inf)
        for i, variant_pos in enumerate(self.variant_obs.variant_positions):
            ext_pos_i = self.variant_pscn.var_pos_to_ext_pos[i]
            for j, read_pos_j in enumerate(read_pos):
                if read_pos_j is None or read_pos_j.intersects(variant_pos.region):
                    ext_read_pos_probs[ext_pos_i] = np.logaddexp(ext_read_pos_probs[ext_pos_i], read_pos_probs[j])
                    break
            else:
                ext_read_pos_probs[ext_pos_i] = np.logaddexp(ext_read_pos_probs[ext_pos_i], FORBIDDEN_LOC_PENALTY)

        ext_read_pos_probs += log_pscn_frac
        ext_read_pos_probs -= logsumexp(ext_read_pos_probs)
        read_allele = allele_obs.allele_ix
        for i, paralog_gt in enumerate(self.paralog_genotypes):
            self.paralog_genotype_probs[i] += paralog_gt.calc_read_obs_prob(ext_read_pos_probs, read_allele)

    def utilize_reads(self, varcall_params, out):
        if self.paralog_genotypes is None:
            return

        variant = self.variant_obs.variant
        n_alleles = len(variant.alleles)
        for gt in self.paralog_genotypes:
            gt.precompute_read_obs_probs(n_alleles, varcall_params.error_rate)

        n_copies = len(self.variant_pscn.ext_pos_cn)
        log_pscn_frac = np.log(self.variant_pscn.ext_pos_cn) - np.log(np.sum(self.variant_pscn.ext_pos_cn))
        for paired_read_position, allele_obs in self.read_obs:
            self._add_read(paired_read_position, allele_obs, n_copies, log_pscn_frac)
        self.paralog_genotype_probs -= logsumexp(self.paralog_genotype_probs)

        _, paralog_str = _genotype_probs_to_str(self.paralog_genotypes, self.paralog_genotype_probs)
        out.write('{}\t{}\tparalog_probs\t{}\n'.format(self.sample, variant.pos, paralog_str))

        pooled_genotypes, pooled_genotype_probs = MultiCopyGT.aggregate_genotypes(
            self.paralog_genotypes, self.paralog_genotype_probs, None)
        best_pooled, pooled_str = _genotype_probs_to_str(pooled_genotypes, pooled_genotype_probs)
        out.write('{}\t{}\tpooled_probs\t{}\n'.format(self.sample, self.variant_obs.variant.pos, pooled_str))
        self.pooled_genotype = pooled_genotypes[best_pooled]
        self.pooled_genotype_qual = int(common.extended_phred_qual(
            pooled_genotype_probs, best_pooled, rem_prob=self.remaining_prob))


class VarCallParameters:
    def __init__(self, args, samples):
        self.pooled_gt_thresh = args.limit_pooled * common.LOG10
        self.mutation_rate = args.mutation_rate * common.LOG10
        self._set_use_af(args, samples) # TODO: Actually use AF.
        self.no_mate_penalty = args.no_mate_penalty * common.LOG10
        self.psv_ref_gt = args.psv_ref_gt

        self.error_rate = np.power(10.0, args.error_rate)
        assert 0 < self.error_rate < 1

        self.base_qual = args.base_qual
        self.base_log_qual = -0.1 * self.base_qual * common.LOG10

        self.min_read_depth = args.limit_depth[0]
        self.max_read_depth = args.limit_depth[1]
        self.max_strand_bias = args.max_strand_bias
        self.min_freebayes_qual = args.limit_qual

    def _set_use_af(self, args, samples):
        use_af_arg = args.use_af.lower()
        if use_af_arg in ('yes', 'y', 'true', 't'):
            self.use_af = True

        elif use_af_arg in ('no', 'n', 'false', 'f'):
            self.use_af = False

        elif use_af_arg.startswith('over-'):
            count = use_af_arg.split('-', 1)[-1]
            if not count.isdigit():
                raise ValueError('Cannot parse --use-af {}'.format(use_af_arg))
            self.use_af = len(samples) >= int(count)

        else:
            raise ValueError('Cannot parse --use-af {}'.format(use_af_arg))


def _merge_overlapping_variants(variants):
    res = []
    start = end = variants[0].start
    ref = ''
    for var in variants:
        assert var.start >= start
        var_ref = var.ref
        var_end = var.start + len(var_ref)
        if var_end > end:
            ref += var_ref[end - var.start : ]
            end = var_end

    alt_alleles = []
    alt_dict = {}
    all_allele_corresp = []
    for var in variants:
        prefix = ref[ : var.start - start]
        suffix = ref[var.start + len(var.ref) - start : ]
        allele_corresp = [0]
        for alt in var.alts:
            allele = prefix + alt + suffix
            allele_ix = alt_dict.get(allele)
            if allele_ix is None:
                alt_alleles.append(allele)
                alt_dict[allele] = allele_ix = len(alt_alleles)
            allele_corresp.append(allele_ix)
        all_allele_corresp.append(allele_corresp)

    new_vars = []
    for var, allele_corresp in zip(variants, all_allele_corresp):
        nvar = var.copy()
        nvar.start = start
        nvar.ref = ref
        nvar.alts = alt_alleles
        for sample, fmt in var.samples.items():
            nfmt = nvar.samples[sample]
            if 'GT' in fmt:
                nfmt['GT'] = tuple(sorted(allele_corresp[i] for i in fmt['GT']))
            if 'PGT' in fmt:
                new_pgt = sorted(allele_corresp[int(i)] for i in fmt['PGT'].split('/'))
                nfmt['PGT'] = '/'.join(map(str, new_pgt))
        # common.log('    Convert:')
        # common.log('      | {}'.format(str(var).strip()))
        # common.log('      > {}'.format(str(nvar).strip()))
        new_vars.append(nvar)
    return new_vars


def _process_overlapping_variants(variants):
    # common.log('Process overlapping variants ({})'.format(len(variants)))
    # for var in variants:
    #     common.log('!    {}'.format(str(var).strip()))

    var0 = variants[0]
    for i in range(1, len(variants)):
        if variants[i].start != var0.start or variants[i].alleles != var0.alleles:
            # common.log('    Requires merge')
            variants = _merge_overlapping_variants(variants)
            break

    gts = defaultdict(set)
    gt_qual = defaultdict(float)
    pooled_gts = defaultdict(set)
    pooled_gt_qual = defaultdict(float)
    filters = defaultdict(Filters)

    # common.log('START')
    best_var = None
    best_ncopies = 0
    for i, var in enumerate(variants):
        pos2 = var.info['pos2']
        curr_ncopies = 1 if len(pos2) == 0 else len(pos2) + 1
        if curr_ncopies > best_ncopies:
            best_var = var
            best_ncopies = curr_ncopies

        for sample, fmt in var.samples.items():
            if 'GT' in fmt:
                gts[sample].add(fmt['GT'])
                gt_qual[sample] = max(gt_qual[sample], fmt['GQ'])
            elif 'GTs' in fmt:
                gts[sample].update(fmt['GTs'])

            if 'PGT' in fmt:
                pooled_gts[sample].add(fmt['PGT'])
                pooled_gt_qual[sample] = max(pooled_gt_qual[sample], fmt['PGQ'])
            elif 'PGTs' in fmt:
                pooled_gts[sample].update(fmt['PGTs'])

            if 'FILT' in fmt:
                filters[sample].update_str(VarFilter, fmt['FILT'])

    var = best_var.copy()
    for sample in best_var.samples:
        fmt = var.samples[sample]
        curr_gts = tuple(gts[sample])
        if len(curr_gts) != 1:
            if 'GT' in fmt:
                del fmt['GT']
                del fmt['GQ']
            if len(curr_gts) > 1:
                fmt['GTs'] = tuple('/'.join(map(str, gt)) for gt in curr_gts)
        else:
            fmt['GT'] = curr_gts[0]
            fmt['GQ'] = gt_qual[sample]

        curr_pgts = tuple(pooled_gts[sample])
        if len(curr_pgts) != 1:
            if 'PGT' in fmt:
                del fmt['PGT']
                del fmt['PGQ']
            if len(curr_pgts) > 1:
                fmt['PGTs'] = curr_pgts
        else:
            fmt['PGT'] = curr_pgts[0]
            fmt['PGQ'] = pooled_gt_qual[sample]

        curr_filt = filters[sample]
        if curr_filt:
            fmt['FILT'] = curr_filt.to_tuple()
            if any(curr_filt.map(VarFilter.need_qual0)):
                if 'GQ' in fmt:
                    fmt['GQ0'] = fmt['GQ']
                    fmt['GQ'] = 0
    # common.log('    === {}'.format(str(var).strip()))
    return var


def merge_duplicates(variants):
    prev_chrom = None
    prev_end = None
    start_ix = 0

    merged_variants = []
    for i, variant in enumerate(variants):
        if variant.chrom != prev_chrom or variant.start >= prev_end:
            if i - start_ix > 1:
                merged_variants.append(_process_overlapping_variants(variants[start_ix:i]))
            elif i:
                merged_variants.append(variants[i - 1])
            prev_chrom = variant.chrom
            prev_end = variant.start + len(variant.ref)
            start_ix = i
        else:
            prev_end = max(prev_end, variant.start + len(variant.ref))

    n = len(variants)
    if n - start_ix > 1:
        merged_variants.append(_process_overlapping_variants(variants[start_ix:]))
    elif n:
        merged_variants.append(variants[n - 1])
    return merged_variants
