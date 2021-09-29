import sys
import operator
import itertools
import pysam
import numpy as np
from enum import Enum
from scipy.stats import poisson, multinomial
from scipy.special import logsumexp
from functools import lru_cache
from collections import namedtuple, defaultdict

from . import duplication as duplication_
from .cigar import Cigar, Operation
from .genome import Interval
from .paralog_cn import Filters
from . import common
from . import itree
from . import polynomials


def copy_vcf_fields(source_rec, target_rec, old_to_new=None):
    """
    Copies format fields from source_rec to target_rec.
        Copies fields GQ, DP, GT, AD.
        NOTE: In the future we can add GL field, but it is unclear how to use it with ploidy > 2,
        also it is unclear, how to sum values if two alleles were combined.

    old_to_new: None or list of indices, old_to_new[old_allele_index] = new_allele_index.
    """
    n_alleles = len(target_rec.alleles)
    for sample, source in source_rec.samples.items():
        target = target_rec.samples[sample]
        target['GQ'] = source['GQ']
        target['DP'] = source['DP']
        source_gt = source['GT']
        if not source_gt or source_gt[0] is None:
            source_gt = None

        if old_to_new is None:
            target['GT'] = source_gt
            if source['AD'][0] is not None:
                target['AD'] = source['AD']
            continue

        if source_gt:
            target['GT'] = sorted(old_to_new[j] for j in source_gt)
        if source['AD'][0] is not None:
            source_allele_depth = source['AD']
            target_allele_depth = [0] * n_alleles
            for i, j in enumerate(old_to_new):
                target_allele_depth[j] += source_allele_depth[i]
            target['AD'] = target_allele_depth


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
def all_gt_counts(n_alleles, ploidy):
    """
    all_gt_counts(n_alleles=3, ploidy=2) -> (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)
    all_gt_counts(n_alleles=2, ploidy=3) -> (3, 0),    (2, 1),    (1, 2),    (0, 3)
    """
    if n_alleles == 1:
        return ((ploidy,),)

    genotype = [0] * n_alleles
    genotype[0] = ploidy
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


def genotype_likelihoods(ploidy, allele_counts, error_prob=0.001, gt_counts=None):
    """
    Returns two lists of the same size: [gt_counts: tuple of ints], [likelihood: float].
    Returned likelihoods are not normalized and should be normalized next.

    allele_counts stores counts of reads that cover each allele, and other_count shows number of reads that
    do not correspond to any allele.
    """
    # TODO: use other_count.

    n_alleles = len(allele_counts)
    total_reads = sum(allele_counts)
    gt_counts = gt_counts or all_gt_counts(n_alleles, ploidy)
    likelihoods = np.zeros(len(gt_counts))

    for i, gt in enumerate(gt_counts):
        # Element-wise maximum. Use it to replace 0 with error probability.
        probs = np.maximum(gt, error_prob * ploidy)
        probs /= sum(probs)
        likelihoods[i] = multinomial.logpmf(allele_counts, total_reads, probs)
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
        self.n_copies = len(alleles)
        self.n_alleles = max(alleles) + 1
        self.cn = cn

        self.sample_genotypes = all_gt_counts(self.n_copies, cn)
        self.psv_genotypes = all_gt_counts(self.n_alleles, cn)
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


def _fill_psv_gts(sample_id, sample_cn, psv_infos, psv_counts, psv_start_ix, psv_end_ix, max_genotypes):
    for psv_ix in range(psv_start_ix, psv_end_ix):
        counts = psv_counts[psv_ix][sample_id]
        if counts.skip:
            continue

        psv_info = psv_infos[psv_ix]
        if psv_info.psv_gt_probs[sample_id] is not None:
            continue

        precomp_data = psv_info.precomp_datas.get(sample_cn)
        if precomp_data is None:
            precomp_data = _PrecomputedData(psv_info.allele_corresp, sample_cn)
            psv_info.precomp_datas[sample_cn] = precomp_data

        if len(precomp_data.sample_genotypes) > max_genotypes:
            continue
        psv_genotypes = precomp_data.psv_genotypes
        _, probs = genotype_likelihoods(sample_cn, counts.allele_counts, gt_counts=psv_genotypes)
        probs -= logsumexp(probs)
        psv_info.psv_gt_probs[sample_id] = probs
        psv_info.sample_cns[sample_id] = sample_cn


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
            sample_cn = sample_const_region.pred_cn
            if sample_cn > max_agcn:
                continue

            psv_start_ix, psv_end_ix = psv_searcher.overlap_ixs(reg_start, reg_end)
            _fill_psv_gts(sample_id, sample_cn, psv_infos, psv_counts, psv_start_ix, psv_end_ix, max_genotypes)


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
    for psv_info in psv_infos:
        psv_info.support_matrix = [None] * n_samples

    f_values = region_group_extra.psv_f_values
    psv_exponents = region_group_extra.psv_infos
    psv_gt_coefs_cache = [{} for _ in range(n_psvs)]

    for sample_id in range(n_samples):
        for psv_ix in range(n_psvs):
            psv_info = psv_infos[psv_ix]
            psv_gt_probs = psv_info.psv_gt_probs[sample_id]
            if np.isnan(f_values[psv_ix, 0]) or psv_gt_probs is None:
                continue

            sample_cn = psv_info.sample_cns[sample_id]
            if sample_cn in psv_gt_coefs_cache[psv_ix]:
                curr_psv_gt_coefs = psv_gt_coefs_cache[psv_ix][sample_cn]
            else:
                precomp_data = psv_info.precomp_datas[sample_cn]
                f_pow_combs = precomp_data.f_power_combinations(f_values[psv_ix])
                curr_psv_gt_coefs = precomp_data.eval_poly_matrices(f_pow_combs)
                psv_gt_coefs_cache[psv_ix][sample_cn] = curr_psv_gt_coefs

            psv_info.support_matrix[sample_id] = support_row = np.zeros(len(curr_psv_gt_coefs))
            for sample_gt_ix, psv_gt_coefs in enumerate(curr_psv_gt_coefs):
                support_row[sample_gt_ix] = logsumexp(psv_gt_coefs + psv_gt_probs)
            support_row *= psv_exponents[psv_ix].info_content


def _read_string(reader):
    length = ord(reader.read(1))
    return reader.read(length).decode()


class PsvStatus(Enum):
    Unreliable = 0
    Semireliable = 1
    Reliable = 2

    @classmethod
    def from_str(cls, s):
        if s == 'u' or s == 'unreliable':
            return cls.Unreliable
        elif s == 's' or s == 'semi-reliable':
            return cls.Semireliable
        elif s == 'r' or r == 'reliable':
            return cls.Reliable
        else:
            raise ValueError('Unexpected PSV status {}'.format(s))


class _PsvPosAllele(namedtuple('_PsvPosAllele', ('psv_ix interval strand allele_ix'))):
    def __lt__(self, other):
        return self.interval.__lt__(other.interval)


class _PsvPos:
    def __init__(self, psv, genome, psv_ix, variant_ix):
        """
        psv_ix = index of the PSV across all PSVs,
        variant_ix = tuple (index of the variant across all variants, index of PSV in the variant).
        """
        self.psv_ix = psv_ix
        self.variant_ix = variant_ix

        self.psv_record = psv
        self.alleles = tuple(psv.alleles)
        self.positions = [_PsvPosAllele(psv_ix,
            Interval(genome.chrom_id(psv.chrom), psv.start, psv.start + len(self.alleles[0])), True, 0)]

        for pos2 in psv.info['pos2']:
            pos2 = pos2.split(':')
            allele_ix = int(pos2[3]) if len(pos2) == 4 else 1
            allele = self.alleles[allele_ix]
            chrom_id = genome.chrom_id(pos2[0])
            start = int(pos2[1]) - 1
            self.positions.append(
                _PsvPosAllele(psv_ix, Interval(chrom_id, start, start + len(allele)), pos2[2] == '+', allele_ix))
        self._init_pos_weights()

        self.n_copies = len(self.positions)
        self.ref_cn = self.n_copies * 2

        self.f_values = np.array(list(map(np.double, psv.info['fval'])))
        self.info_content = float(psv.info['info'])
        self.status = PsvStatus.from_str(psv.info['rel'])

    def _init_pos_weights(self):
        MATCH_WEIGHT = np.log(0.99)
        MISMATCH_WEIGHT = np.log(0.01)

        # If a read has allele i what would be "probability" of each homologous position.
        # Stores an array of natural log weights, size = (n_alleles x n_positions).
        self.pos_weights = []
        for allele_ix in range(len(self.alleles)):
            curr_weights = [MATCH_WEIGHT if pos.allele_ix == allele_ix else MISMATCH_WEIGHT for pos in self.positions]
            self.pos_weights.append(np.array(curr_weights))

    @property
    def start(self):
        return self.positions[0].interval.start

    @property
    def ref(self):
        return self.alleles[0]

    def skip(self):
        # TODO: Use more sophisticated filters.
        return self.status != PsvStatus.Reliable

    def weighted_positions(self, allele_ix):
        """
        Returns two arrays: one with _PsvPosAllele,
        second with probababilities of each position given a read that supports allele_ix.
        """
        return self.positions, self.pos_weights[allele_ix]


AlleleObservation = namedtuple('AlleleObservation', ('allele_ix', 'is_first_mate'))


class _SkipPosition(Exception):
    def __init__(self, start):
        self.start = start


class VariantReadObservations:
    def __init__(self, start, alleles, n_samples):
        self.start = start
        # Alleles from the binary input.
        self.tmp_alleles = alleles

        self.variant = None
        self.variant_positions = None
        self.new_vcf_records = None
        self.new_vcf_allele_corresp = None
        if n_samples is not None:
            # list of dictionaries (one for each sample),
            # each dictionary: key = read hash (47 bits), value = AlleleObservation.
            self.observations = [{} for _ in range(n_samples)]
            self.other_observations = np.zeros(n_samples, dtype=np.int16)
        else:
            self.observations = None
            self.other_observations = None
        self.use_samples = None

        # List of instances of VariantReadObservations (because there could be several PSVs within one variant).
        self.psv_observations = None
        # Parent VariantReadObservations. Inverse of self.psv_observations.
        self.parent = None

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
            self.other_observations[sample_id] = count

        while True:
            allele_ix = ord(reader.read(1))
            if allele_ix == 255:
                break
            if allele_ix == 254:
                raise _SkipPosition(self.start)

            sample_id = sample_conv[int.from_bytes(reader.read(2), byteorder)]
            read_hash = int.from_bytes(reader.read(8), byteorder)
            is_first_mate = read_hash & 1
            read_hash -= is_first_mate

            # Read with the same hash is present (most probably read mate, collisions should be extremely rare).
            if read_hash in self.observations[sample_id]:
                if self.observations[sample_id][read_hash].allele_ix != allele_ix:
                    del self.observations[sample_id][read_hash]
                    self.other_observations[sample_id] += 2
            else:
                self.observations[sample_id][read_hash] = AlleleObservation(allele_ix, bool(is_first_mate))

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
                    sample_observations[read_hash] = AlleleObservation(new_allele, allele_obs.is_first_mate)
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

        if new_alleles == old_alleles:
            # No need to change any alleles.
            return

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
        return allele_corresp

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
        assert var_start <= psv_start and psv_end <= var_end
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

    def set_psv(self, psv: _PsvPos):
        allele_corresp = self._psv_allele_corresp(self.variant, psv.psv_record)
        full_match = np.all(np.arange(len(self.variant.alleles)) == allele_corresp)

        new_psv_obs = self.shallow_copy_obs() if full_match else self.copy_obs()
        if not full_match:
            new_psv_obs._update_observations(allele_corresp)
        new_psv_obs.variant = psv
        new_psv_obs.parent = self
        self.psv_observations.append(new_psv_obs)
        return new_psv_obs

    def init_vcf_records(self, genome, vcf_header):
        alleles = self.variant.alleles
        self.new_vcf_records = []
        self.new_vcf_allele_corresp = []
        for i, pos in enumerate(self.variant_positions):
            record = vcf_header.new_record()
            record.chrom = pos.region.chrom_name(genome)
            record.start = pos.region.start

            if pos.sequence == self.variant.ref:
                new_alleles = alleles
                old_to_new = np.arange(len(alleles))
            else:
                new_alleles = [pos.sequence]
                for allele in alleles:
                    if allele != new_alleles[0]:
                        new_alleles.append(allele)
                old_to_new = np.array(self._simple_allele_corresp(alleles, new_alleles))

            if not pos.strand:
                new_alleles = tuple(common.rev_comp(allele) for allele in new_alleles)
            record.alleles = new_alleles

            pos2_str = []
            for j, pos2 in enumerate(self.variant_positions):
                if i == j:
                    continue
                pos2_str.append('{}:{}:{}'.format(pos2.region.chrom_name(genome), pos2.region.start_1,
                    '+' if pos.strand == pos2.strand else '-'))
            record.info['pos2'] = pos2_str
            record.info['overlPSV'] = 'T' if self.psv_observations else 'F'
            record.qual = 100
            record.filter.add('PASS')
            self.new_vcf_records.append(record)
            self.new_vcf_allele_corresp.append(old_to_new)

    def update_vcf_records(self, var_gts, genome):
        GT_THRESHOLD = np.log(0.55)

        n_alleles = len(self.variant.alleles)
        sample_id = var_gts.sample_id
        allele_counts = np.bincount([obs.allele_ix for obs in self.observations[sample_id].values()],
            minlength=n_alleles)
        read_depth = np.sum(allele_counts) + self.other_observations[sample_id]

        for i, record in enumerate(self.new_vcf_records):
            old_to_new = self.new_vcf_allele_corresp[i]
            rec_fmt = record.samples[sample_id]
            rec_fmt['PGT'] = '/'.join(str(old_to_new[allele_ix]) for allele_ix in var_gts.pooled_genotype)
            rec_fmt['DP'] = int(read_depth)
            curr_allele_counts = [0] * len(record.alleles)
            for j, count in zip(old_to_new, allele_counts):
                curr_allele_counts[j] = int(count)
            rec_fmt['AD'] = curr_allele_counts
            if var_gts.filter:
                rec_fmt['GTfilter'] = var_gts.filter.to_tuple()

            if var_gts.paralog_genotypes is not None:
                ext_copy_i = var_gts.var_pos_to_ext_pos[i]
                best_gt_ix = np.argmax(var_gts.paralog_gt_probs)
                gt_prob = var_gts.paralog_gt_probs[best_gt_ix]
                best_gt = var_gts.paralog_genotypes[best_gt_ix]
                if gt_prob >= GT_THRESHOLD:
                    rec_fmt['GT'] = tuple(old_to_new[allele_ix] for allele_ix in best_gt.genotype[ext_copy_i])
                    rec_fmt['GQ'] = int(common.phred_qual(var_gts.paralog_gt_probs, best_gt_ix))
                    if len(var_gts.ext_pos_to_var_pos[ext_copy_i]) > 1:
                        rec_fmt['shared_pos2'] = var_gts.var_get_shared_pos2(ext_copy_i, record.info['pos2'], genome)

    def set_use_samples(self, max_other_obs_frac=0.1, min_read_depth=10):
        n_samples = len(self.observations)
        self.use_samples = np.zeros(n_samples, dtype=np.bool)
        for sample_id, sample_observations in enumerate(self.observations):
            oth = self.other_observations[sample_id]
            total = oth + len(sample_observations)
            self.use_samples[sample_id] = total >= min_read_depth and oth <= max_other_obs_frac * total

        if self.psv_observations:
            for psv_observations in self.psv_observations:
                # Not a shallow copy.
                if id(self.observations) == id(psv_observations.observations):
                    psv_observations.use_samples = self.use_samples.copy()
                else:
                    psv_observations.set_use_samples(max_other_obs_frac, min_read_depth)

    def psv_update_read_probabilities(self, sample_id, read_positions):
        assert self.parent is not None
        if not self.use_samples[sample_id] or self.variant.skip():
            return
        for read_hash, allele_obs in self.observations[sample_id].items():
            read_positions[read_hash].add(self.variant, allele_obs)

    @staticmethod
    def create_vcf_header(genome, argv, samples):
        vcf_header = pysam.VariantHeader()
        vcf_header.add_line('##command="%s"' % ' '.join(argv))
        for name, length in zip(genome.chrom_names, genome.chrom_lengths):
            vcf_header.add_line('##contig=<ID={},length={}>'.format(name, length))

        vcf_header.add_line('##INFO=<ID=pos2,Number=.,Type=String,Description="Second positions of the variant. '
            'Format: chrom:pos:strand">')
        vcf_header.add_line('##INFO=<ID=overlPSV,Number=1,Type=Character,'
            'Description="Variants overlaps a PSV. Possible values: T/F">')
        vcf_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        vcf_header.add_line('##FORMAT=<ID=shared_pos2,Number=.,Type=Integer,'
            'Description="Genotype (GT) is shared with these variant positions (0-based indices from pos2 field).'
            'This happens if several positions are indistinguishable in a sample '
            'and the genotype is calculated jointly for several duplication copies">')
        vcf_header.add_line('##FORMAT=<ID=GTfilter,Number=1,Type=String,Description="GT filter">')
        vcf_header.add_line('##FORMAT=<ID=GQ,Number=1,Type=Float,Description="The Phred-scaled Genotype Quality">')
        vcf_header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">')
        vcf_header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,'
            'Description="Number of observation for each allele">')
        vcf_header.add_line('##FORMAT=<ID=PGT,Number=1,Type=String,Description="Pooled Genotype">')

        for sample in samples:
            vcf_header.add_sample(sample)
        return vcf_header


def _next_or_none(iter):
    try:
        return next(iter)
    except StopIteration:
        return None


def read_freebayes_binary(ra_reader, samples, vcf_file, dupl_pos_finder):
    byteorder = 'big' if ord(ra_reader.read(1)) else 'little'
    n_samples = len(samples)
    n_in_samples = int.from_bytes(ra_reader.read(2), byteorder)

    # Converter between sample IDs sample_conv[sample_id from input] = sample_id in parascopy.
    sample_conv = np.zeros(n_in_samples, dtype=np.uint16)
    for i in range(n_in_samples):
        sample = _read_string(ra_reader)
        sample_conv[i] = samples.id(sample)

    positions = []
    vcf_record = _next_or_none(vcf_file)
    while vcf_record is not None:
        try:
            read_allele_obs = VariantReadObservations.from_binary(ra_reader, byteorder, sample_conv, n_samples)
            if read_allele_obs.start != vcf_record.start:
                assert read_allele_obs.start < vcf_record.start
                continue
            read_allele_obs.set_variant(vcf_record, dupl_pos_finder)
            positions.append(read_allele_obs)
            vcf_record = _next_or_none(vcf_file)
        except StopIteration:
            break
        except _SkipPosition as exc:
            if exc.start == vcf_record.start:
                vcf_record = _next_or_none(vcf_file)
    return positions


def add_psv_variants(all_read_allele_obs, psv_records, genome):
    searcher = itree.NonOverlTree(all_read_allele_obs, itree.start, itree.variant_end)

    psv_read_allele_obs = []
    n_psvs = [0] * len(PsvStatus)

    for psv_ix, psv in enumerate(psv_records):
        i, j = searcher.overlap_ixs(psv.start, psv.start + len(psv.ref))
        # There are no good observations of the PSV.
        if i == j or 'fval' not in psv.info:
            psv_read_allele_obs.append(None)
            continue

        if i + 1 != j:
            common.log('ERROR: PSV {}:{} overlaps several Freebayes variants'.format(psv.chrom, psv.pos))
            common.log('')
            psv_read_allele_obs.append(None)
            continue

        # There is exactly one corresponding variant.
        assert i + 1 == j
        variant_ix = (i, len(all_read_allele_obs[i].psv_observations))
        psv = _PsvPos(psv, genome, psv_ix, variant_ix)
        n_psvs[psv.status.value] += 1
        curr_psv_obs = all_read_allele_obs[i].set_psv(psv)
        assert len(psv_read_allele_obs) == psv_ix
        psv_read_allele_obs.append(curr_psv_obs)

    common.log('Use {} PSVs. Of them {} reliable and {} semi-reliable'
        .format(sum(n_psvs), n_psvs[PsvStatus.Reliable.value], n_psvs[PsvStatus.Semireliable.value]))
    return psv_read_allele_obs


def write_vcf_file(output, vcf_header, all_read_allele_obs, genome, tabix):
    records = []
    for variant in all_read_allele_obs:
        records.extend(variant.new_vcf_records)
    records.sort(key=lambda record: (genome.chrom_id(record.chrom), record.start))

    gzip = output.endswith('.gz')
    with pysam.VariantFile(output, 'wz' if gzip else 'w', header=vcf_header) as vcf_file:
        for record in records:
            vcf_file.write(record)
    if gzip and tabix != 'none':
        common.Process([tabix, '-p', 'vcf', output]).finish()


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
        Returns list of tuples _VariantPosition.
        """
        ref_len = len(variant.ref)
        region = Interval(self._chrom_id, variant.start, variant.start + ref_len)
        res = [duplication_.VariantPosition(region, True, variant.ref)]

        for entry in self._dupl_tree.overlap(region.start, region.end):
            dupl = entry.data
            pos2 = dupl.align_variant(variant)
            if pos2 is not None:
                res.append(pos2)
        res.sort()
        return tuple(res)


class _ReadEndPositions:
    def __init__(self):
        self.seq_len = None
        self.read_positions = None

        # _PsvPosAllele entries.
        self.psv_positions = []
        # Natural-log position weights (not necessarily normalized).
        self.psv_pos_weights = []

    def set_read_positions(self, read_coord, dupl_pos_finder):
        self.seq_len = read_coord.seq_len
        if read_coord.mapped_uniquely:
            self.read_positions = (read_coord.aln_region,)
        else:
            self.read_positions = dupl_pos_finder.find_read_pos(read_coord.aln_region)
            if len(self.read_positions) == 1:
                # Because we know that the read is not mapped uniquely.
                self.read_positions = ()

    def add(self, psv_pos, allele_ix):
        positions, weights = psv_pos.weighted_positions(allele_ix)
        weights = np.array(weights)
        weights -= logsumexp(weights)
        self.psv_positions.extend(positions)
        self.psv_pos_weights.extend(weights)

    def calc_single_mate_pos_probs(self):
        """
        Returns (None, None), if the read is not present in the read coordinates (too far from the duplication).

        Otherwise returns two lists:
            - List of regions,
            - List of region log probabilities.

        If mapping position is not known but is outside of the duplication, returns two empty lists.
        """
        if self.read_positions is None:
            # Read is not present in the read coordinates.
            return None, None

        n_read_pos = len(self.read_positions)
        if n_read_pos == 0:
            # Unknown possible positions.
            return (), np.zeros(0)
        elif n_read_pos == 1:
            # Unique possible position.
            return (self.read_positions[0],), np.zeros(1)
        elif not self.psv_positions:
            # There are no PSVs, all positions are equally likely.
            return self.read_positions, np.full(n_read_pos, -np.log(n_read_pos))

        sum_weight = np.zeros(n_read_pos)
        overl_psvs = np.zeros(n_read_pos, dtype=np.uint16)
        # TODO: Check overlapping PSVs.
        for psv_pos, psv_weight in zip(self.psv_positions, self.psv_pos_weights):
            psv_interval = psv_pos.interval
            for read_pos_ix, read_pos in enumerate(self.read_positions):
                if read_pos.intersects(psv_interval):
                    sum_weight[read_pos_ix] += psv_weight
                    overl_psvs[read_pos_ix] += 1
                    break
            # else:
            #     # No read positions found.
            #     print('        Could not find position for PSV {!r}'.format(psv_pos.interval))

        MISSING_PSV_PENALTY = np.log(0.01)
        max_psvs = np.max(overl_psvs)
        sum_weight += (max_psvs - overl_psvs) * MISSING_PSV_PENALTY
        sum_weight -= logsumexp(sum_weight)
        return self.read_positions, sum_weight

    def debug(self, genome, psvs):
        if not self.read_positions:
            print('        Unknown possible positions')
        else:
            for pos in self.read_positions:
                print('        {} ({} position)'.format(pos.to_str_comma(genome),
                    'possible' if len(self.read_positions) > 1 else 'unique'))

        ixs = sorted(range(len(self.psv_positions)), key=lambda i: self.psv_positions[i])
        for i in ixs:
            pos = self.psv_positions[i]
            print('        {}:{}  psv {} - {} [{}]  = {:.3f}'.format(pos.interval.to_str_comma(genome),
                '+' if pos.strand else '-', pos.psv_ix,
                psvs[pos.psv_ix].variant.alleles[pos.allele_ix], pos.allele_ix, np.exp(self.psv_pos_weights[i])))


class _ReadPositions:
    def __init__(self):
        # Read positions for the first and second mate.
        self.mate_read_pos = (_ReadEndPositions(), _ReadEndPositions())
        self.requires_mate = True

        self.positions1 = None
        self.probs1 = None
        self.positions2 = None
        self.probs2 = None

    def set_read_positions(self, read_coord, dupl_pos_finder):
        self.requires_mate = read_coord.is_paired
        is_read1 = bool(read_coord.read_hash & np.uint8(1))
        self.mate_read_pos[1 - is_read1].set_read_positions(read_coord, dupl_pos_finder)

    def add(self, psv_pos: _PsvPos, allele_obs: AlleleObservation):
        self.mate_read_pos[1 - allele_obs.is_first_mate].add(psv_pos, allele_obs.allele_ix)

    def init_paired_read_pos_probs(self, max_mate_dist):
        self.positions1, self.probs1 = self.mate_read_pos[0].calc_single_mate_pos_probs()
        self.positions2, self.probs2 = self.mate_read_pos[1].calc_single_mate_pos_probs()

        # pos_probs2 is None if the read mate is too far away, and it is empty,
        # if mate is near, but its position is unknown.
        if not self.positions1 or not self.positions2:
            return

        # Penalty for not having a mate is 10^-5.
        NO_MATE_PENALTY = -5 * common.LOG10
        new_probs1 = self.probs1 + NO_MATE_PENALTY
        new_probs2 = self.probs2 + NO_MATE_PENALTY

        n = len(self.positions1)
        m = len(self.positions2)
        for i in range(n):
            pos1 = self.positions1[i]
            for j in range(m):
                if pos1.distance(self.positions2[j]) <= max_mate_dist:
                    new_prob = self.probs1[i] + self.probs2[j]
                    new_probs1[i] = max(new_probs1[i], new_prob)
                    new_probs2[j] = max(new_probs2[j], new_prob)

        self.probs1 = new_probs1 - logsumexp(new_probs1)
        self.probs2 = new_probs2 - logsumexp(new_probs2)

    def get_mate_pos_probs(self, is_read1):
        if is_read1:
            return self.positions1, self.probs1
        return self.positions2, self.probs2

    def debug(self, genome, psvs):
        print('      * First mate:')
        self.mate_read_pos[0].debug(genome, psvs)
        print('      * Second mate:')
        self.mate_read_pos[1].debug(genome, psvs)

        if self.positions1 is not None:
            if self.positions1 is not None:
                print('      $ Weighted first mate positions: {} {}'.format(len(self.positions1), len(self.probs1)))
                for pos1, prob1 in zip(self.positions1, self.probs1):
                    print('        {:30} {:.3f}'.format(pos1.to_str(genome), prob1))
            else:
                print('      $ Unknown first mate positions.')

            if self.positions2 is not None:
                print('      $ Weighted second mate positions: {} {}'.format(len(self.positions2), len(self.probs2)))
                for pos2, prob2 in zip(self.positions2, self.probs2):
                    print('        {:30} {:.3f}'.format(pos2.to_str(genome), prob2))
            else:
                print('      $ Unknown second mate positions.')


class ReadCollection:
    def __init__(self, sample_id, coord_index, dupl_pos_finder):
        CLEAR_LAST_BIT = ~np.uint64(1)

        self.sample_id = sample_id
        self.read_positions = defaultdict(_ReadPositions)
        coordinates = coord_index.load(sample_id)
        for read_coord in coordinates.values():
            read_hash = read_coord.read_hash & CLEAR_LAST_BIT
            self.read_positions[read_hash].set_read_positions(read_coord, dupl_pos_finder)

    def add_psv_observations(self, psv_read_allele_obs, max_mate_dist):
        for curr_allele_obs in psv_read_allele_obs:
            if curr_allele_obs is not None:
                curr_allele_obs.psv_update_read_probabilities(self.sample_id, self.read_positions)
        # for read_pos in self.read_positions.values(): TODO: RETURN
        for read_hash, read_pos in self.read_positions.items():
            read_pos.init_paired_read_pos_probs(max_mate_dist)

    def debug_read_probs(self, sample, genome, psvs):
        for psv in psvs:
            if psv is None:
                print('PSV is None')
                continue
            print('PSV {} [{}],  alleles {} (orig {}, alleles {})  use samples {}, parent use {}'.format(
                psv.start + 1, psv.variant.psv_ix,
                ' '.join(psv.variant.alleles), psv.variant.psv_record.start + 1, ' '.join(psv.variant.psv_record.alleles),
                psv.use_samples, psv.parent.use_samples))

        print('Sample {}'.format(sample))
        for read_hash, read_pos in self.read_positions.items():
            print('    Hash = {}'.format(read_hash))
            read_pos.debug(genome, psvs)


class Filter(Enum):
    Pass = 0
    UnclearObs = 11
    Unknown_agCN = 12
    Unknown_psCN = 13

    def __str__(self):
        if self == Filter.Pass:
            return 'PASS'
        if self == Filter.UnclearObs:
            return 'UnclearObs'
        if self == Filter.Unknown_agCN:
            return 'Unknown_agCN'
        if self == Filter.Unknown_psCN:
            return 'Unknown_psCN'


class MultiCopyGT:
    ALLELE_MATCH_PROB = 0.99
    ALLELE_ERROR_PROB = 1 - ALLELE_MATCH_PROB

    def __init__(self, gt, pscn):
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

    def precompute_read_obs_probs(self, n_alleles):
        self.precomp_read_obs_probs = np.zeros((n_alleles, self.n_copies))
        for copy_i, part_gt in enumerate(self.genotype):
            part_gt_len = len(part_gt)
            for allele_j in range(n_alleles):
                n_matches = part_gt.count(allele_j)
                self.precomp_read_obs_probs[allele_j, copy_i] = np.log((n_matches * MultiCopyGT.ALLELE_MATCH_PROB
                    + (part_gt_len - n_matches) * MultiCopyGT.ALLELE_ERROR_PROB) / part_gt_len)

    def calc_read_obs_prob(self, copy_probabilities, read_allele):
        return logsumexp(self.precomp_read_obs_probs[read_allele] + copy_probabilities)

    def __str__(self):
        return '_'.join('/'.join(map(str, gt)) for gt in self.genotype)

    @classmethod
    def create_all(cls, pooled_gt, pscn):
        if len(set(pooled_gt)) == 1:
            yield MultiCopyGT(pooled_gt, pscn)
            return

        used_genotypes = set()
        for gt in set(itertools.permutations(pooled_gt)):
            new = MultiCopyGT(gt, pscn)
            if new.genotype not in used_genotypes:
                used_genotypes.add(new.genotype)
                yield new


class VariantGenotypePred:
    def __init__(self, variant_obs):
        self.variant_obs = variant_obs
        self.sample_id = None
        self.pooled_genotype = None
        self.filter = Filters()

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

        self.paralog_genotypes = None
        self.paralog_gt_probs = None

    def init_genotypes(self, sample_id, sample, cn_profiles):
        self.sample_id = sample_id
        # TODO: If there is no pooled genotype?
        self.pooled_genotype = self.variant_obs.variant.samples[sample]['GT']

        if not self.variant_obs.use_samples[sample_id]:
            self.filter.add(Filter.UnclearObs)
            return self

        any_cn_estimates = False
        for var_pos in self.variant_obs.variant_positions:
            cn_estimates = list(cn_profiles.cn_estimates(sample_id, var_pos.region))
            if cn_estimates:
                any_cn_estimates = True
                self._match_var_pos_to_cn_estimates(cn_estimates)
                break
        if not any_cn_estimates:
            self.filter.add(Filter.Unknown_agCN)
            return self

        if self.ext_pos_cn is not None:
            self.paralog_genotypes = list(MultiCopyGT.create_all(self.pooled_genotype, self.ext_pos_cn))
            self._init_gt_priors()
        return self

    def _match_var_pos_to_cn_estimates(self, cn_estimates):
        if len(cn_estimates) > 1:
            assert len(cn_estimates) == 2
            est_a = cn_estimates[0]
            est_b = cn_estimates[1]
            if est_a.pred_cn != est_b.pred_cn or est_a.paralog_cn != est_b.paralog_cn:
                print('    SetB unknown agCN')
                self.filter.add(Filter.Unknown_agCN)
                return
        self.cn_estimate = est = cn_estimates[0]

        # TODO: What to do with qualities and agCN/psCN filters?
        agcn = est.pred_cn
        if agcn is None:
            print('    SetC unknown agCN')
            self.filter.add(Filter.Unknown_agCN)
            return

        pscn = np.array(est.paralog_cn)
        pscn_known = np.array([cn is not None for cn in pscn])
        n_copies = len(pscn)
        n_known = sum(pscn_known)
        if n_known == 0:
            self.filter.add(Filter.Unknown_psCN)
            return

        cn_regions = [est.sample_const_region.region1] + [region for region, _ in est.sample_const_region.regions2]
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

        self.var_pos_to_ext_pos = [None] * len(self.variant_obs.variant_positions)
        self.ext_pos_to_var_pos = [[] for _ in range(len(self.ext_pos_cn))]
        for i, var_pos in enumerate(self.variant_obs.variant_positions):
            for j, cn_region in enumerate(cn_regions):
                if cn_region.intersects(var_pos.region):
                    ext_pos_i = self.cn_est_pos_to_ext_pos[j]
                    self.var_pos_to_ext_pos[i] = ext_pos_i
                    self.ext_pos_to_var_pos[ext_pos_i].append(i)
                    break
        self.ext_pos_to_var_pos = tuple(map(tuple, self.ext_pos_to_var_pos))

    def _init_gt_priors(self):
        # TODO: Make more complicated.
        self.paralog_gt_probs = np.zeros(len(self.paralog_genotypes))

    def var_get_shared_pos2(self, ext_copy_i, pos2_str, genome):
        assert ext_copy_i == 0
        ext_variant_regions = tuple(self.variant_obs.variant_positions[i].region
            for i in self.ext_pos_to_var_pos[ext_copy_i])
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
        if not read_pos:
            return

        k = 0
        n_read_pos = len(read_pos)
        ext_read_pos_probs = np.full(n_copies, -np.inf)
        for i, variant_pos in enumerate(self.variant_obs.variant_positions):
            for j in range(k, n_read_pos):
                if read_pos[j].intersects(variant_pos.region):
                    k = j + 1
                    ext_pos_i = self.var_pos_to_ext_pos[i]
                    ext_read_pos_probs[ext_pos_i] = np.logaddexp(ext_read_pos_probs[ext_pos_i], read_pos_probs[j])
                    break
        ext_read_pos_probs += log_pscn_frac
        ext_read_pos_probs -= logsumexp(ext_read_pos_probs)
        # print('        Initial  read pos probs:', read_pos_probs)
        # print('        Extended read pos probs:', ext_read_pos_probs)

        read_allele = allele_obs.allele_ix
        gt_probs = np.zeros(len(self.paralog_genotypes))
        for i, gt in enumerate(self.paralog_genotypes):
            gt_probs[i] = gt.calc_read_obs_prob(ext_read_pos_probs, read_allele)
        gt_probs -= logsumexp(gt_probs)
        # print('        Update genotype probabilities by {}'.format(gt_probs))
        self.paralog_gt_probs += gt_probs

    def add_reads(self, read_positions):
        if self.paralog_genotypes is None or len(self.paralog_genotypes) < 2:
            return

        n_alleles = len(self.variant_obs.variant.alleles)
        for gt in self.paralog_genotypes:
            gt.precompute_read_obs_probs(n_alleles)

        n_copies = len(self.ext_pos_cn)
        log_pscn_frac = np.log(self.ext_pos_cn) - np.log(sum(self.ext_pos_cn))
        for read_hash, allele_obs in self.variant_obs.observations[self.sample_id].items():
            # print('    Add read {}, allele obs {}'.format(read_hash, allele_obs))
            self._add_read(read_positions[read_hash], allele_obs, n_copies, log_pscn_frac)
        self.paralog_gt_probs -= logsumexp(self.paralog_gt_probs)
