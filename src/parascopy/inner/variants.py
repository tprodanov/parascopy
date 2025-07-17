import sys
import operator
import itertools
import pysam
import numpy as np
from enum import Enum
from scipy.stats import poisson, multinomial, fisher_exact, binom
from scipy.special import logsumexp, gammaln
from functools import lru_cache
from collections import namedtuple, defaultdict, OrderedDict

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
    if cn == 0:
        assert len(gt_counts) == 1
        return gt_counts, np.zeros(1)

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
        self.n_poss_pscns = len(self.poss_pscns)
        self.psv_genotypes = all_gt_counts(self.n_alleles, cn)
        self.n_psv_genotypes = len(self.psv_genotypes)
        self.f_combinations, f_comb_ixs = _all_f_combinations(self.n_copies, cn)

        self._gt_coefs_cache = {}
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
                dtype=np.int32)
            self.poly_matrices.append(poly_matrix)
            self.f_ix_converts.append(f_ix_convert)

    def fval_to_gt_coefs(self, f_values):
        fval_tup = tuple(f_values)
        gt_coefs = self._gt_coefs_cache.get(fval_tup)
        if gt_coefs is None:
            f_pow = f_values[:, np.newaxis] ** range(0, self.cn + 1)
            f_pow_comb = np.prod(f_pow[range(self.n_copies), self.f_combinations], axis=1)

            gt_coefs = np.zeros((self.n_poss_pscns, self.n_psv_genotypes))
            with np.errstate(divide='ignore'):
                for i in range(self.n_poss_pscns):
                    f_ix_convert = self.f_ix_converts[i]
                    gt_vec = np.sum(self.poly_matrices[i] * f_pow_comb[f_ix_convert], axis=1)
                    gt_coefs[i] = np.log(np.clip(gt_vec, 0, 1))
            self._gt_coefs_cache[fval_tup] = gt_coefs
        return gt_coefs


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
            if precomp_data.n_poss_pscns <= max_genotypes:
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
                    curr_psv_gt_coefs = precomp_data.fval_to_gt_coefs(f_values[psv_ix])
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
        allele1_len = len(self.alleles[0])
        self._is_indel = any(len(allele) != allele1_len for allele in self.alleles)

        for pos2 in psv.info['pos2']:
            if pos2.endswith(':+') or pos2.endswith(':-'):
                pos2 = pos2.rsplit(':', 2)
                allele_ix = 1
            else:
                pos2 = pos2.rsplit(':', 3)
                allele_ix = int(pos2[3])
            chrom_id = genome.chrom_id(pos2[0])
            start = int(pos2[1]) - 1
            allele = self.alleles[allele_ix]
            self.psv_positions.append(
                _PsvPosAllele(psv_ix, Interval(chrom_id, start, start + len(allele)), pos2[2] == '+', allele_ix))
        self._init_pos_weights(varcall_params)

        self.n_copies = len(self.psv_positions)
        self.ref_cn = self.n_copies * 2

        self.f_values = np.array(list(map(np.double, psv.info['fval'])))
        self.info_content = float(psv.info['info'])
        self.status = PsvStatus.from_str(psv.info['rel'])

    @property
    def is_reliable(self):
        return self.status == PsvStatus.Reliable

    @property
    def is_indel(self):
        return self._is_indel

    def _init_pos_weights(self, varcall_params):
        error_rate = varcall_params.error_rate[self._is_indel]
        match_weight = np.log1p(-error_rate)
        mismatch_weight = np.log(error_rate)

        # If a read has allele i what would be "probability" of each homologous position.
        # Stores an array of natural log weights, size = (n_alleles x n_positions).
        self.pos_weights = []
        for allele_ix in range(len(self.alleles)):
            curr_weights = np.array([match_weight if pos.allele_ix == allele_ix else mismatch_weight
                for pos in self.psv_positions])
            curr_weights -= logsumexp(curr_weights)
            self.pos_weights.append(curr_weights)

    @property
    def start(self):
        return self.psv_positions[0].region.start

    @property
    def ref(self):
        return self.alleles[0]

    @property
    def alts(self):
        return self.alleles[1:]

    def weighted_positions(self, allele_ix):
        """
        Returns iterator over pairs:
            - PSV position (_PsvPosAllele),
            - normalized probability of the position given a read that supports allele_ix.
        """
        return zip(self.psv_positions, self.pos_weights[allele_ix])


AlleleObservation = namedtuple('AlleleObservation', 'allele_ix is_first_mate ln_qual')


class _SkipPosition(Exception):
    def __init__(self, start):
        self.start = start


def strand_bias_test(strand_counts):
    """
    Input: np.array((2, n_alleles)).
    Find strand bias for each allele and returns an array with p-values.
    For each allele, run Fisher Exact test comparing forward/reverse counts against
    the sum forward/reverse counts for all other alleles.

    Returns Phred-score of Fisher test p-values for all alleles.
    """
    n_alleles = strand_counts.shape[1]
    forw_sum, rev_sum = np.sum(strand_counts, axis=1)
    total_sum = forw_sum + rev_sum

    if forw_sum == 0 or rev_sum == 0:
        # One of the rows is 0 -- all p-values = 1.
        return 0

    if n_alleles == 2:
        if 0 < np.sum(strand_counts[:, 0]) < total_sum:
            pval = fisher_exact(strand_counts)[1]
            # Use +0 to remove -0.
            return -10.0 * np.log10(pval) + 0.0
        # Otherwise, if there are no observations for one of the alleles -- all p-values = 1.
        return 0

    max_phred = 0
    for i in range(n_alleles):
        forw_i, rev_i = strand_counts[:, i]
        if 0 < forw_i + rev_i < total_sum:
            pval = fisher_exact(((forw_i, forw_sum - forw_i), (rev_i, rev_sum - rev_i)))[1]
            max_phred = max(max_phred, -10.0 * np.log10(pval) + 0.0)
    return max_phred


def _round_qual(qual):
    if qual <= 0.0:
        return 0
    log = int(np.floor(np.log10(qual)))
    return round(qual, max(0, 2 - log))


_ALN_ARGS = None


def _align_alleles(ref_allele, alt_allele):
    import parasail
    from . import alignment
    global _ALN_ARGS
    if _ALN_ARGS is None:
        _ALN_ARGS = alignment.Weights().parasail_args()
    aln = parasail.nw_trace_scan_sat(alt_allele, ref_allele, *_ALN_ARGS)
    cigar = []
    for value in aln.cigar.seq:
        length = value >> 4
        op_num = value & 0xf
        Cigar.append(cigar, length, Operation(op_num))
    cigar = Cigar.from_tuples(cigar)
    common.log(f'    Fixed CIGAR: {cigar}')
    assert cigar.ref_len == len(ref_allele) and cigar.read_len == len(alt_allele)
    return cigar


class VariantReadObservations:
    def __init__(self, var_region, alleles, n_samples):
        """
        n_samples is None when this is a shallow copy
        """
        self.var_region = var_region
        # Alleles from the binary input.
        self.tmp_alleles = alleles

        self.variant = None
        # List of duplications.VariantPosition
        self.variant_positions = None
        # Number of variant positions that are out of bounds.
        self.pos_out_of_bounds = 0
        self.new_vcf_records = None
        # For each variant position (each repeat copy), stores array old_to_new,
        # where old_to_new[variant_allele] -> variant_allele on that repeat copy.
        self._new_vcf_allele_corresp = None
        # For each variant position, stores the index of the reference allele (old_to_new[self._ref_allele] == 0).
        self._ref_alleles = None
        self._is_indel = None

        if n_samples is not None:
            # list of dictionaries (one for each sample),
            # each dictionary: key = read hash (47 bits), value = AlleleObservation.
            self.observations = [{} for _ in range(n_samples)]
            self.other_observations = np.zeros(n_samples, dtype=np.int32)
        else:
            self.observations = None
            self.other_observations = None

        # Once initialized, this vector will store the following information for each sample:
        # Vector (n_copies) with log-probabilities that the genotype is 0/0 on that copy.
        self._homoz_00_probs = []

        # List of instances of VariantReadObservations (because there could be several PSVs within one variant).
        self.psv_observations = None
        # If variant overlaps a PSV, psv_priors_matrix will contain a matrix (n_copies x n_alleles) with
        # log-probabilities of observing corresponding allele on the corresponding copy.
        self.psv_priors_matrix = None
        # Dictionary from ext_to_var -> pairs ()
        self.psv_paralog_cache = None
        # Parent VariantReadObservations. Inverse of self.psv_observations.
        self.parent = None

    @property
    def start(self):
        return self.var_region.start

    @property
    def is_shallow_copy(self):
        return id(self.observations) == id(self.parent.observations)

    @property
    def is_psv(self):
        return self.parent is not None

    @property
    def is_indel(self):
        return self._is_indel

    @property
    def has_psvs(self):
        return bool(self.psv_observations)

    @classmethod
    def from_binary(cls, chrom_id, reader, byteorder, sample_conv, n_samples):
        start = reader.read(4)
        if not start:
            raise StopIteration
        start = int.from_bytes(start, byteorder)

        self = cls(None, None, n_samples)
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
                raise _SkipPosition(start)

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
        allele1_len = len(alleles[0])
        self._is_indel = any(len(allele) != allele1_len for allele in alleles)
        self.var_region = Interval(chrom_id, start, start + len(alleles[0]))
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
        assert variant.start == self.var_region.start
        assert variant.start + len(variant.ref) == self.var_region.end
        self.variant = variant
        self.variant_positions, self.pos_out_of_bounds = dupl_pos_finder.find_variant_pos(variant, self.var_region)

        # Use this to remove duplicate alleles, if any.
        new_alleles = tuple(OrderedDict.fromkeys(variant.alleles))
        variant.alleles = new_alleles
        ref_allele_len = len(new_alleles[0])
        self._is_indel = any(len(allele) != ref_allele_len for allele in new_alleles)

        old_alleles = self.tmp_alleles
        self.tmp_alleles = None
        if new_alleles != old_alleles:
            allele_corresp = self._simple_allele_corresp(old_alleles, new_alleles)
            self._update_observations(allele_corresp)

    def copy_obs(self, new_region):
        """
        Only copies observations and other_observations.
        """
        n_samples = len(self.observations)
        new = VariantReadObservations(new_region, None, n_samples)
        for sample_id in range(n_samples):
            new.other_observations[sample_id] = self.other_observations[sample_id]
            new.observations[sample_id] = self.observations[sample_id].copy()
        return new

    def shallow_copy_obs(self, new_region):
        new = VariantReadObservations(new_region, None, None)
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
            common.log('WARN: PSV {}:{} is larger than the variant {}:{}. Skipping the PSV'
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
            if cigar.ref_len != var_end - var_start or cigar.read_len != len(var_alt_allele):
                common.log('WARN: VCF file contains incorrect CIGAR for {}:{} ({} and {} -> {})'.format(
                    variant.chrom, var_start + 1, var_ref_allele, var_alt_allele, cigar))
                cigar = _align_alleles(var_ref_allele, var_alt_allele)
            cigar.init_proxy_index()

            alt_size_diff = len(var_alt_allele) - len(var_ref_allele)
            start2, end2 = cigar.aligned_region(rel_start, rel_end, alt_size_diff)
            sub_allele = var_alt_allele[start2 : end2]
            var_sub_alleles.append(sub_allele)
        return VariantReadObservations._simple_allele_corresp(var_sub_alleles, psv_alleles)

    def set_psv(self, psv: _PsvPos, varcall_params):
        if self.pos_out_of_bounds:
            return None
        if len(self.variant_positions) != len(psv.psv_positions):
            common.log('WARN: Failed to initialize PSV {}:{}. '.format(self.variant.chrom, psv.start + 1) +
                'Most likely, because of the boundary of a duplication')
            return None

        allele_corresp = self._psv_allele_corresp(self.variant, psv.psv_record)
        if allele_corresp is None:
            return None
        full_match = np.all(np.arange(len(self.variant.alleles)) == allele_corresp)
        self._update_psv_paralog_priors(psv, allele_corresp, varcall_params)

        if full_match:
            new_psv_obs = self.shallow_copy_obs(psv.psv_positions[0].region)
        else:
            new_psv_obs = self.copy_obs(psv.psv_positions[0].region)
            new_psv_obs._update_observations(allele_corresp)

        new_psv_obs.variant = psv
        new_psv_obs.variant_positions = [psv_pos.to_variant_pos(new_psv_obs.variant.alleles)
            for psv_pos in new_psv_obs.variant.psv_positions]
        new_psv_obs.parent = self
        new_psv_obs._is_indel = psv.is_indel
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

        if self.psv_priors_matrix is None:
            self.psv_priors_matrix = priors_update
            self.psv_paralog_cache = {}
        else:
            self.psv_priors_matrix += priors_update
            self.psv_paralog_cache.clear()

        self.psv_priors_matrix = np.maximum(self.psv_priors_matrix, varcall_params.mutation_rate)
        self.psv_priors_matrix -= logsumexp(self.psv_priors_matrix, axis=1)[:, np.newaxis]

    def calculate_psv_paralog_priors(self, ext_to_var, paralog_genotypes, paralog_genotype_probs):
        cache = self.psv_paralog_cache.get(ext_to_var)
        if cache is None:
            n_copies = len(ext_to_var)
            n_alleles = self.psv_priors_matrix.shape[1]
            ext_paralog_priors = np.zeros((n_copies, n_alleles))
            for i, ext_pos in enumerate(ext_to_var):
                new_row = logsumexp(self.psv_priors_matrix[ext_pos, :], axis=0)
                ext_paralog_priors[i] = new_row - logsumexp(new_row)
            gt_cache = {}
            self.psv_paralog_cache[ext_to_var] = (ext_paralog_priors, gt_cache)
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
            paired_allele_counts = [0] * (2 * curr_n_alleles)
            for old_allele_ix, new_allele_ix in enumerate(old_to_new):
                all_allele_counts[new_allele_ix] = int(gt_pred.all_allele_counts[old_allele_ix])
                good_allele_counts[new_allele_ix] = int(gt_pred.good_allele_counts[old_allele_ix])
                strand_allele_counts[2 * new_allele_ix] = int(gt_pred.strand_counts[0, old_allele_ix])
                strand_allele_counts[2 * new_allele_ix + 1] = int(gt_pred.strand_counts[1, old_allele_ix])
                paired_allele_counts[2 * new_allele_ix] = int(gt_pred.paired_counts[0, old_allele_ix])
                paired_allele_counts[2 * new_allele_ix + 1] = int(gt_pred.paired_counts[1, old_allele_ix])
            rec_fmt['AD'] = all_allele_counts
            rec_fmt['ADq'] = good_allele_counts
            rec_fmt['SB'] = strand_allele_counts
            rec_fmt['FS'] = int(gt_pred.strand_phred)
            if True: # any(unpaired_count > 0 for unpaired_count in itertools.islice(paired_allele_counts, 1, None, 2)):
                rec_fmt['PP'] = paired_allele_counts
                rec_fmt['FP'] = int(gt_pred.paired_phred)
            if gt_filter:
                rec_fmt['FILT'] = gt_filter.to_tuple()
            if i == 0:
                continue

            ext_copy_i = var_pscn.var_to_ext[i - 1] if var_pscn.pscn_known else None
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

            if paralog_gt is not None and len(paralog_gt) == 0:
                # Paralog-specific copy number = 0.
                gt_filter = gt_filter.copy()
                gt_filter.add(VarFilter.ZeroPsCN)
                rec_fmt['FILT'] = gt_filter.to_tuple()

            elif paralog_gt_qual >= PHRED_THRESHOLD:
                out_gt = None
                # When going from the extended repeat copy to the variant position,
                # there are two cases:
                # - either extended repeat copy has only copy,
                # - or the paralog-specific genotype is clear (only one allele present).
                if len(var_pscn.ext_to_var[ext_copy_i]) == 1:
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
            debug_out.write('PSV #{:4d}   {}:{}    alleles {}    f-values {}\n'.format(variant.psv_ix,
                variant.psv_record.chrom, variant.start + 1, ' '.join(variant.alleles), variant.f_values))
            if variant.alleles != variant.psv_record.alleles:
                debug_out.write('    Original pos {},   original alleles {}\n'
                    .format(variant.psv_record.start + 1, ' '.join(variant.psv_record.alleles)))

        psv_gt_out.write('{}\t{}\t{}\t'.format(gt_pred.sample, variant.start + 1, variant.status.short_str()))
        assert self.parent is not None
        if gt_pred.skip:
            psv_gt_out.write('*\t*\t*\t*\t*\t*\t*\tF\tskip\n')
            return False

        variant_pscn = gt_pred.variant_pscn
        if variant_pscn.agcn == 0:
            psv_gt_out.write('*\t*\t*\t*\t*\t*\t*\tF\tzero_agcn\n')
            return False

        if not variant_pscn.pscn_known:
            psv_gt_out.write('*\t*\t*\t*\t*\t*\t*\tF\tunknown_pscn\n')
            return False
        # TODO: What if there are filters? What if the quality is low?

        psv_gt_out.write('{}\t{}\t{}\t{}\t'.format(
            ','.join(map(str, variant_pscn.var_to_ext)),
            ','.join(map(str, variant_pscn.extended_cn)),
            ','.join(map(str, gt_pred.good_allele_counts)),
            str(gt_pred.pooled_genotype)))

        n_ext_copies = len(variant_pscn.extended_cn)
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
            variant_pscn.ext_f_values, variant_pscn.ext_allele_corresp, variant_pscn.extended_cn)
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

    @staticmethod
    def create_vcf_headers(genome, argv, samples):
        vcf_headers = []
        # First pooled, second un-pooled.
        for i in range(2):
            vcf_header = pysam.VariantHeader()
            vcf_header.add_line(common.vcf_command_line(argv))
            for name, length in genome.names_lengths():
                vcf_header.add_line('##contig=<ID={},length={}>'.format(name, length))

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
                '"Unedited genotype quality in case there is a sample-specific filter present.">')
            vcf_header.add_line('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">')
            vcf_header.add_line('##FORMAT=<ID=AD,Number=R,Type=Integer,'
                'Description="Number of pooled observation for each allele">')
            vcf_header.add_line('##FORMAT=<ID=ADq,Number=R,Type=Integer,'
                'Description="Number of pooled observation for each allele that pass base quality filters">')
            vcf_header.add_line('##FORMAT=<ID=SB,Number=.,Type=Integer,'
                'Description="Number of pooled observations for each allele and strand '
                '(forward REF, reverse REF, forward ALT[1], reverse ALT[1], etc).">')
            vcf_header.add_line('##FORMAT=<ID=FS,Number=1,Type=Float,'
                'Description="Strand bias: maximal Phred-scaled p-value across all alleles.">')
            vcf_header.add_line('##FORMAT=<ID=PP,Number=.,Type=Integer,'
                'Description="Number of pooled reads with/without proper pair for each allele '
                '(with pair REF, without pair REF, with pair ALT[1], without pair ALT[1], etc).">')
            vcf_header.add_line('##FORMAT=<ID=FP,Number=1,Type=Integer,'
                'Description="Unpaired reads bias: maximal Phred-scaled p-value across all alleles.">')
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

    chrom_id = dupl_pos_finder.chrom_id
    all_read_allele_obs = []
    vcf_record = _next_or_none(vcf_file)
    while vcf_record is not None:
        try:
            read_allele_obs = VariantReadObservations.from_binary(
                chrom_id, ra_reader, byteorder, sample_conv, n_samples)
            if read_allele_obs.var_region.start != vcf_record.start:
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
            common.log('WARN: PSV {}:{} overlaps several Freebayes variants'.format(psv.chrom, psv.pos))
            continue

        # There is exactly one corresponding variant.
        assert i + 1 == j
        psv = _PsvPos(psv, genome, psv_ix, varcall_params)
        if np.any(np.isnan(psv.f_values)):
            continue
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
    def __init__(self, chrom_id, duplications):
        self._chrom_id = chrom_id
        self._dupl_tree = itree.create_interval_tree(duplications, itree.region1_start, itree.region1_end)
        self._cache = {}

    @property
    def chrom_id(self):
        return self._chrom_id

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

    def find_variant_pos(self, variant, var_region):
        """
        Returns
        - tuple of `VariantPosition`s,
        - number of positions that were out of bounds.
        """
        assert self._chrom_id == var_region.chrom_id
        res = [duplication_.VariantPosition(var_region, True, variant.ref)]

        out_of_bounds = 0
        for entry in self._dupl_tree.overlap(var_region.start, var_region.end):
            dupl = entry.data
            try:
                pos2 = dupl.align_variant(variant)
            except errors.VariantOutOfBounds:
                out_of_bounds += 1
                continue
            if pos2 is not None:
                res.append(pos2)
        res.sort()
        return tuple(res), out_of_bounds


FORBIDDEN_LOC_PENALTY = -20.0 * common.LOG10
MISSING_PSV_PENALTY = -4.0 * common.LOG10


class _ReadEndPositions:
    def __init__(self):
        self.seq_len = None
        # Original read location.
        self.original_loc = None
        # If the read is mapped uniquely, unique_loc will have an Interval. Otherwise, it is None.
        self.unique_loc = None
        # tuple: If true location is unknown, but there are locations that are certainly incorrect, store it here.
        self.forbidden_locs = None
        # List of all variant positions (chrom_id, pos).
        self.var_positions = []

        # Possible read locations and their probabilities.
        self.possible_locs = None
        self.possible_loc_probs = None
        # Pairs (_PsvPosAllele, log probability).
        self.overl_psvs = []

    @property
    def have_possible_locations(self):
        return self.unique_loc is not None or bool(self.var_positions)

    def add_read_coord(self, read_coord):
        self.seq_len = read_coord.seq_len
        self.original_loc = read_coord.old_region
        self.unique_loc = read_coord.get_true_location()
        self.forbidden_locs = read_coord.get_forbidden_locations()

    def add_var_positions(self, var_positions):
        self.var_positions.extend(var_positions)

    def debug(self, genome, out):
        out.write('            Original location: {}\n'.format(
            '*' if self.original_loc is None else self.original_loc.to_str(genome)))
        if self.unique_loc:
            out.write('            Unique location: {}\n'.format(self.unique_loc.to_str(genome)))
        if self.forbidden_locs:
            out.write('            Forbidden locations:\n')
            for loc in self.forbidden_locs:
                out.write('                {}\n'.format(loc.to_str(genome)))
        if self.overl_psvs:
            out.write('            PSVs:\n')
        for psv_pos, psv_pos_prob in self.overl_psvs:
            out.write('                #{}   {}:{:,}  allele {} = {:.3f}\n'.format(
                psv_pos.psv_ix, genome.chrom_name(psv_pos.region.chrom_id), psv_pos.region.start + 1,
                psv_pos.allele_ix, np.exp(psv_pos_prob)))
        if self.var_positions:
            out.write('            Variant positions: ')
            out.write('  '.join('{:,}'.format(var_region.start + 1) for var_region in self.var_positions))
            out.write('\n')

    def find_possible_locations(self):
        """
        Finds possible locations based on the overlapped variants and returns a list of Intervals.
        """
        if self.unique_loc is not None:
            self.possible_locs = (self.unique_loc,)
            self.possible_loc_probs = np.zeros(1)
            return
        if not self.var_positions:
            self.possible_locs = ()
            self.possible_loc_probs = ()
            return

        self._cluster_var_positions()
        if self.forbidden_locs:
            n_locs = len(self.possible_locs)
            for region in self.forbidden_locs:
                for i in range(n_locs):
                    if region.intersects(self.possible_locs[i]):
                        self.possible_loc_probs[i] += FORBIDDEN_LOC_PENALTY

        self.possible_locs = tuple(self.possible_locs)
        self.possible_loc_probs = np.array(self.possible_loc_probs)
        self.possible_loc_probs -= logsumexp(self.possible_loc_probs)

    def _cluster_var_positions(self):
        # Add PADDING around found locations.
        PADDING = 10
        # Add possible location is longer than seq_len + LEN_MODIF, make the location forbidden.
        LEN_MODIF = 10
        # If distance between variants is less than VAR_DIST, combine them into the same location.
        VAR_DIST = 50
        max_len = self.seq_len + LEN_MODIF

        self.var_positions.sort()
        start_ix = 0
        self.possible_locs = []
        self.possible_loc_probs = []
        for i in range(1, len(self.var_positions)):
            init = self.var_positions[start_ix]
            prev = self.var_positions[i - 1]
            curr = self.var_positions[i]
            if init.chrom_id != curr.chrom_id or (curr.start - init.end > max_len and
                    curr.start - prev.end > VAR_DIST):
                self.possible_locs.append(Interval(init.chrom_id, init.start - PADDING, prev.end + PADDING))
                self.possible_loc_probs.append(FORBIDDEN_LOC_PENALTY * (prev.start - init.end > max_len))
                start_ix = i

        init = self.var_positions[start_ix]
        last = self.var_positions[-1]
        assert init.chrom_id == last.chrom_id
        self.possible_locs.append(Interval(init.chrom_id, init.start - PADDING, last.end + PADDING))
        self.possible_loc_probs.append(FORBIDDEN_LOC_PENALTY * (last.start - init.end > max_len))

    def add_psv_obs(self, psv_pos, allele_ix):
        if self.unique_loc is not None:
            return

        weighted_positions = list(psv_pos.weighted_positions(allele_ix))
        self.overl_psvs.extend(weighted_positions)
        for i, region in enumerate(self.possible_locs):
            for pos, weight in weighted_positions:
                if region.intersects(pos.region):
                    self.possible_loc_probs[i] += weight
                    break
            else:
                self.possible_loc_probs[i] += MISSING_PSV_PENALTY


class _ReadPositions:
    def __init__(self):
        # Read positions for the second and second first.
        self._mate_read_pos = (_ReadEndPositions(), _ReadEndPositions())
        self._is_reverse = [None, None]
        self.hash = None
        self._requires_mate = common.UNDEF
        self._in_proper_pair = common.UNDEF

        self.positions1 = None
        self.probs1 = None
        self.positions2 = None
        self.probs2 = None

    def add_read_coord(self, read_hash, read_coord):
        if self.hash is None:
            self.hash = read_hash
        else:
            assert self.hash == read_hash
        self._requires_mate = read_coord.is_paired
        is_read1 = bool(read_coord.read_hash & np.uint8(1))
        self._mate_read_pos[is_read1].add_read_coord(read_coord)
        self._is_reverse[is_read1] = read_coord.is_reverse
        if self._is_reverse[1 - is_read1] is None:
            self._is_reverse[1 - is_read1] = not read_coord.is_reverse

    def check_proper_pair(self, max_mate_dist):
        orig1 = self._mate_read_pos[1].original_loc
        orig2 = self._mate_read_pos[0].original_loc
        self._in_proper_pair = orig1 is not None and orig2 is not None and orig1.distance(orig2) < max_mate_dist

    def add_var_positions(self, var_positions, is_first_mate):
        self._mate_read_pos[is_first_mate].add_var_positions(var_positions)

    def add_psv_obs(self, psv_pos: _PsvPos, allele_obs: AlleleObservation):
        self._mate_read_pos[allele_obs.is_first_mate].add_psv_obs(psv_pos, allele_obs.allele_ix)

    def find_possible_locations(self):
        self._mate_read_pos[1].find_possible_locations()
        self._mate_read_pos[0].find_possible_locations()

    def calculate_paired_loc_probs(self, max_mate_dist, no_mate_penalty):
        mate1 = self._mate_read_pos[1]
        mate2 = self._mate_read_pos[0]
        self.positions1 = mate1.possible_locs
        self.positions2 = mate2.possible_locs
        self.probs1 = mate1.possible_loc_probs
        self.probs2 = mate2.possible_loc_probs

        if not mate1.have_possible_locations or not mate2.have_possible_locations:
            # Cannot update any paired-end probabilities.
            if len(self.probs1) > 1:
                self.probs1 -= logsumexp(self.probs1)
            if len(self.probs2) > 1:
                self.probs2 -= logsumexp(self.probs2)
            return

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

    def get_mate_pos_probs(self, is_read1):
        if is_read1:
            return self.positions1, self.probs1
        return self.positions2, self.probs2

    def is_reverse(self, is_read1):
        return self._is_reverse[is_read1]

    def single_end_or_proper_pair(self):
        return not self._requires_mate or self._in_proper_pair

    def debug(self, genome, out):
        out.write('    {:x} ({}proper pair)\n'.format(self.hash, '' if self._in_proper_pair else 'not '))
        out.write('      * Mate 1 ({}):\n'.format('Rev' if self._is_reverse[1] else 'Forw'))
        self._mate_read_pos[1].debug(genome, out)
        if self.positions1:
            for loc, prob in zip(self.positions1, self.probs1):
                out.write('        {:30} {:7.3f} ({:.5f})\n'.format(
                    loc.to_str_comma(genome), prob / common.LOG10, np.exp(prob)))
        elif self._mate_read_pos[1].forbidden_locs:
            out.write('        Forbidden locations:\n')
            for loc in self._mate_read_pos[1].forbidden_locs:
                out.write('        {}\n'.format(loc.to_str_comma(genome)))

        out.write('      * Mate 2 ({}):\n'.format('Rev' if self._is_reverse[0] else 'Forw'))
        self._mate_read_pos[0].debug(genome, out)
        if self.positions2:
            for loc, prob in zip(self.positions2, self.probs2):
                out.write('        {:30} {:7.3f} ({:.5f})\n'.format(
                    loc.to_str_comma(genome), prob / common.LOG10, np.exp(prob)))
        elif self._mate_read_pos[0].forbidden_locs:
            out.write('        Forbidden locations:\n')
            for loc in self._mate_read_pos[0].forbidden_locs:
                out.write('        {}\n'.format(loc.to_str_comma(genome)))


class ReadCollection:
    def __init__(self, sample_id, sample, coord_index):
        CLEAR_LAST_BIT = ~np.uint64(1)

        self.sample_id = sample_id
        self.sample = sample
        self.read_positions = defaultdict(_ReadPositions)
        self.max_mate_dist = coord_index.max_mate_dist
        coordinates = coord_index.load(sample_id)
        if len(coordinates) == 0:
            raise RuntimeError('Sample {} has no appropriate reads!'.format(sample))

        sum_length = 0
        for read_coord in coordinates.values():
            sum_length += read_coord.seq_len
            read_hash = read_coord.read_hash & CLEAR_LAST_BIT
            self.read_positions[read_hash].add_read_coord(read_hash, read_coord)
        self.mean_len = int(round(sum_length / len(coordinates)))
        for read_pos in self.read_positions.values():
            read_pos.check_proper_pair(self.max_mate_dist)

    def find_possible_locations(self):
        for read_pos in self.read_positions.values():
            read_pos.find_possible_locations()

    def add_psv_observations(self, psv_gt_preds, no_mate_penalty):
        for gt_pred in psv_gt_preds:
            variant = gt_pred.variant_obs.variant
            for read_pos, allele_obs in gt_pred.read_obs:
                read_pos.add_psv_obs(variant, allele_obs)

        for read_pos in self.read_positions.values():
            read_pos.calculate_paired_loc_probs(self.max_mate_dist, no_mate_penalty)

    def debug_read_probs(self, genome, out):
        out.write('{}: Read probabilities\n'.format(self.sample))
        for read_pos in self.read_positions.values():
            read_pos.debug(genome, out)


class VarFilter(Enum):
    Pass = 0
    UnknownAgCN   = 10
    UnknownPsCN   = 11
    ConflictingCN = 12
    OutOfBounds   = 13
    ZeroPsCN      = 14

    LowReadDepth  = 20
    HighReadDepth = 21
    StrandBias    = 22
    UnpairedBias  = 23

    def __str__(self):
        if self == VarFilter.Pass:
            return 'PASS'
        return self.name

    @classmethod
    def from_str(cls, s):
        if s == 'PASS':
            return VarFilter.Pass
        if s == 'UnknownAgCN':
            return VarFilter.UnknownAgCN
        if s == 'UnknownPsCN':
            return VarFilter.UnknownPsCN
        if s == 'ConflictingCN':
            return VarFilter.ConflictingCN
        if s == 'OutOfBounds':
            return VarFilter.OutOfBounds
        if s == 'ZeroPsCN':
            return VarFilter.ZeroPsCN
        if s == 'LowReadDepth':
            return VarFilter.LowReadDepth
        if s == 'HighReadDepth':
            return VarFilter.HighReadDepth
        if s == 'StrandBias':
            return VarFilter.StrandBias
        if s == 'UnpairedBias':
            return VarFilter.UnpairedBias
        assert False

    def need_qual0(self):
        if self == VarFilter.UnknownAgCN or self == VarFilter.StrandBias or self == VarFilter.UnpairedBias:
            return True
        return False


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
                if n_matches:
                    self.precomp_read_obs_probs[allele_j, copy_i] = np.log(
                        (n_matches * (1 - error_rate) + (part_gt_len - n_matches) * error_rate) / part_gt_len)
                else:
                    self.precomp_read_obs_probs[allele_j, copy_i] = np.log(error_rate)

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


def _calculate_allele_probs(variant, read_len):
    """
    Calculate probabilities of completely covering each allele based on the allele length and read length.

    For an allele of length `m` and read length `l` there are `l - m - 1` possible read position that would
    completely cover the allele (one base on each side of the allele).
    Next, allele probabilities are normalized in a way that the largest probability will be 1.
    Note, that the resulting array does not sum up to one.
    """
    alleles = variant.alleles
    EPS = 0.1 # Should be smaller than 1
    allele_probs = np.full(len(alleles), EPS)
    for i, allele in enumerate(alleles):
        allele_probs[i] = max(EPS, read_len - len(allele) - 1)
    m = allele_probs.max()
    if m == EPS:
        common.log('WARN: All alleles for {}:{} are longer than the mean read length'.format(
            variant.chrom, variant.pos + 1))
    allele_probs /= m
    return allele_probs


class VariantParalogCN:
    def __init__(self, sample_id, sample, variant_obs, cn_profiles, assume_cn):
        # Three arrays:
        # How to get from variant position to an extended copy.
        self._var_to_ext = None
        # How to get from extended copy to variant position
        self._ext_to_var = None
        # Extended copy copy number.
        self._ext_cn = None
        # If paralog-specific copy number is unknown, three arrays above are None.
        # If aggregate copy number is also unknown, self._agcn is also None.
        self._agcn = None
        self._conflicting_cn = False

        # Matrix of booleans (n_ext_copies, n_alleles). If element is True, i-th extended copy has j-th allele.
        self._ext_allele_corresp = None
        # Average f-values across various extended repeat copies.
        self._ext_fvals = None

        self._set_variant_pscn(sample_id, sample, variant_obs, cn_profiles, assume_cn)
        if self.pscn_known and variant_obs.is_psv:
            self._set_ext_fvals(variant_obs.variant)

    def debug(self):
        print('Variant psCN:')
        print('    Known agCN? {}   Known psCN? {}'.format(self.agcn_known, self.pscn_known))
        print('    agCN {}     extended CN {}'.format(self._agcn, self._ext_cn))
        print('    var pos -> ext pos: {}'.format(self.var_to_ext))
        print('    ext pos -> var pos: {}'.format(self.ext_to_var))

    @property
    def conflicting_cn(self):
        return self._conflicting_cn

    @property
    def agcn_known(self):
        return self._agcn is not None

    @property
    def agcn(self):
        return self._agcn

    @property
    def pscn_known(self):
        return self._ext_cn is not None

    @property
    def extended_cn(self):
        return self._ext_cn

    @property
    def var_to_ext(self):
        return self._var_to_ext

    @property
    def ext_to_var(self):
        return self._ext_to_var

    @property
    def ext_allele_corresp(self):
        return self._ext_allele_corresp

    @property
    def ext_f_values(self):
        return self._ext_fvals

    def _set_variant_pscn(self, sample_id, sample, variant_obs, cn_profiles, assume_cn):
        """
        Sets arrays self._var_to_ext,  self._ext_to_var,  self._ext_cn.
        """
        cn_estimates = cn_profiles.cn_estimates(sample_id, variant_obs.var_region)
        if not cn_estimates:
            # No CN estimates available.
            return

        est = cn_estimates[0]
        if len(cn_estimates) > 1:
            assert len(cn_estimates) == 2
            est_b = cn_estimates[1]
            if est.pred_cn != est_b.pred_cn:
                self._conflicting_cn = True
                return
        self._agcn = est.pred_cn

        n_copies = len(variant_obs.variant_positions)
        assumed_pscn = None if assume_cn is None else \
            self._load_assumed_pscn(sample_id, sample, variant_obs, assume_cn)
        if assumed_pscn:
            self._ext_cn = assumed_pscn
            self._var_to_ext = tuple(range(n_copies))
            self._ext_to_var = tuple((i,) for i in range(n_copies))
            return

        if len(cn_estimates) > 1:
            # est_b is already defined above.
            if est.paralog_cn != est_b.paralog_cn:
                self._conflicting_cn = True
                return
        self._set_regular_pscn(variant_obs, est)

    def _load_assumed_pscn(self, sample_id, sample, variant_obs, assume_cn):
        """
        Returns tuples with paralog-specific copy numbers.
        """
        var_regions = [pos.region for pos in variant_obs.variant_positions]
        assumed_pscn, n_unknown = assume_cn.from_regions(var_regions, sample_id, sample, genome=None, ploidy=None)
        if n_unknown:
            if n_unknown < len(var_regions):
                common.log(
                    'WARN: Variant {}:{} sample {}. Input copy number from "{}" is not available for all copies'
                        .format(variant_obs.variant.chrom, variant_obs.variant.start + 1, sample, assume_cn.filename))
            return None
        self._agcn = sum(assumed_pscn)
        return assumed_pscn

    def _set_regular_pscn(self, variant_obs, cn_est):
        pscn = cn_est.paralog_cn
        if all(cn is None for cn in pscn):
            return
        cn_regions = [cn_est.sample_const_region.region1]
        if cn_est.sample_const_region.regions2 is not None:
            cn_regions.extend(map(operator.itemgetter(0), cn_est.sample_const_region.regions2))
        n_var_pos = len(variant_obs.variant_positions)
        n_regions = len(cn_regions)

        if n_var_pos != n_regions:
            if n_var_pos + variant_obs.pos_out_of_bounds != n_regions:
                variant = variant_obs.variant
                common.log(('WARN: Variant {}: number of variant positions ({}) + positions out of bounds ({}) '
                    'does not match the number of regions ({})').format(variant.start + 1, n_var_pos,
                    variant_obs.pos_out_of_bounds, n_regions))
            return

        cn_region_used = np.zeros(n_regions, dtype=np.bool_)
        pos_to_region = np.zeros(n_var_pos, dtype=np.int32)
        for i, var_pos in enumerate(variant_obs.variant_positions):
            for j, cn_region in enumerate(cn_regions):
                if not cn_region_used[j] and cn_region.intersects(var_pos.region):
                    cn_region_used[j] = True
                    pos_to_region[i] = j
                    break
            else:
                # There is no cn region for that variant position.
                if variant_obs.pos_out_of_bounds == 0:
                    common.log('WARN: Variant {}: could not match variant positions and CN regions'
                        .format(variant_obs.variant.start + 1))
                return

        self._var_to_ext = [None] * n_var_pos
        self._ext_to_var = []
        self._ext_cn = []
        for i, var_pos in enumerate(variant_obs.variant_positions):
            j = pos_to_region[i]
            if pscn[j] is not None:
                self._var_to_ext[i] = len(self._ext_to_var)
                self._ext_to_var.append((i,))
                self._ext_cn.append(pscn[j])

        sum_ext_cn = sum(self._ext_cn)
        if None in self._var_to_ext:
            j = len(self._ext_to_var)
            last_ext_to_var = []
            self._ext_cn.append(self._agcn - sum_ext_cn)
            for i in range(n_var_pos):
                if self._var_to_ext[i] is None:
                    self._var_to_ext[i] = j
                    last_ext_to_var.append(i)
            self._ext_to_var.append(tuple(last_ext_to_var))
        else:
            assert sum_ext_cn == self._agcn

        self._var_to_ext = tuple(self._var_to_ext)
        self._ext_to_var = tuple(self._ext_to_var)
        self._ext_cn = tuple(self._ext_cn)

    def _set_ext_fvals(self, psv):
        n_alleles = len(psv.alleles)
        n_ext_copies = len(self._ext_cn)
        self._ext_allele_corresp = np.zeros((n_ext_copies, n_alleles), dtype=np.bool_)
        self._ext_fvals = np.zeros(n_ext_copies)
        for var_i, ext_copy_i in enumerate(self._var_to_ext):
            allele_ix = psv.psv_positions[var_i].allele_ix
            self._ext_allele_corresp[ext_copy_i, allele_ix] = True
            self._ext_fvals[ext_copy_i] += psv.f_values[var_i]
        # NOTE: Here, we take mean f-value if needed. Alternatively, we can minimal f-value.
        self._ext_fvals /= np.bincount(self._var_to_ext)


class VariantGenotypePred:
    def __init__(self, sample_id, sample, variant_obs, cn_profiles, assume_cn):
        self.sample_id = sample_id
        self.sample = sample
        self.variant_obs = variant_obs
        self.variant_pscn = VariantParalogCN(sample_id, sample, variant_obs, cn_profiles, assume_cn)
        self.filter = Filters()

        if not self.variant_pscn.agcn_known:
            self.filter.add(VarFilter.UnknownAgCN)
        elif not self.variant_pscn.pscn_known:
            self.filter.add(VarFilter.UnknownPsCN)
            if variant_obs.pos_out_of_bounds:
                self.filter.add(VarFilter.OutOfBounds)

        if self.variant_pscn.conflicting_cn:
            self.filter.add(VarFilter.ConflictingCN)
        self.skip = False

        # List of pairs (_ReadPositions, AlleleObservation).
        self.read_obs = None
        n_alleles = len(self.variant_obs.variant.alleles)
        self.all_allele_counts = np.zeros(n_alleles, dtype=np.uint32)
        self.good_allele_counts = np.zeros(n_alleles, dtype=np.uint32)
        # Forward/Reverse strand read counts for each allele.
        self.strand_counts = np.zeros((2, n_alleles), dtype=np.uint32)
        self.strand_phred = None
        # Count paired and unpaired reads supporting each allele.
        self.paired_counts = np.zeros((2, n_alleles), dtype=np.uint32)
        self.paired_phred = None

        self.pooled_genotype = None
        self.pooled_genotype_qual = None
        self.paralog_genotypes = None
        self.paralog_genotype_probs = None

        self.start_pooled_genotypes = None
        self.start_pooled_genotype_probs = None
        self.remaining_prob = None

    def init_read_counts(self, all_read_positions, varcall_params, debug_out=None):
        variant = self.variant_obs.variant
        write_debug = debug_out is not None

        self.read_obs = []
        for read_hash, allele_obs in self.variant_obs.observations[self.sample_id].items():
            allele_ix = allele_obs.allele_ix
            self.all_allele_counts[allele_ix] += 1
            if write_debug:
                debug_out.write('{}\t{:x}\t{}\t{}\t{:.0f}\n'.format(
                    variant.pos, read_hash, 2 - allele_obs.is_first_mate, allele_ix,
                    -10 * allele_obs.ln_qual / common.LOG10))

            if allele_obs.ln_qual <= varcall_params.base_log_qual[self.variant_obs.is_indel]:
                self.good_allele_counts[allele_ix] += 1
                paired_read_position = all_read_positions[read_hash]
                self.read_obs.append((paired_read_position, allele_obs))
                is_reverse = paired_read_position.is_reverse(allele_obs.is_first_mate)
                self.strand_counts[np.uint8(is_reverse), allele_ix] += 1
                proper_pair = paired_read_position.single_end_or_proper_pair()
                self.paired_counts[np.uint8(not proper_pair), allele_ix] += 1

        read_depth = len(self.read_obs)
        if read_depth < varcall_params.min_read_depth:
            self.filter.add(VarFilter.LowReadDepth)
            self.skip = True
        elif read_depth > varcall_params.max_read_depth:
            self.filter.add(VarFilter.HighReadDepth)
            self.skip = True

        self.strand_phred = strand_bias_test(self.strand_counts)
        if self.strand_phred >= varcall_params.max_strand_bias:
            self.filter.add(VarFilter.StrandBias)
        self.paired_phred = strand_bias_test(self.paired_counts)
        if self.paired_phred >= varcall_params.max_unpaired_bias:
            self.filter.add(VarFilter.UnpairedBias)

    def update_read_locations(self):
        var_positions = tuple(var_pos.region for var_pos in self.variant_obs.variant_positions)
        for paired_read_position, allele_obs in self.read_obs:
            paired_read_position.add_var_positions(var_positions, allele_obs.is_first_mate)

    def init_genotypes(self, varcall_params, mean_read_len, out, only_pooled=False):
        # TODO: Add a parameter pooled_priors=False and calculate pooled genotype priors.

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

        if not self.variant_pscn.pscn_known or only_pooled:
            return
        self.paralog_genotypes = []
        for pooled_gt in filt_pooled_genotypes:
            self.paralog_genotypes.extend(MultiCopyGT.create_all(pooled_gt, self.variant_pscn.extended_cn))
        self._calculate_paralog_priors(varcall_params, out)

    def _init_pooled_genotypes(self, varcall_params, mean_read_len, out):
        '''
        If return_best: return
            - pooled genotypes that pass a threshold,
            - sum probability of the removed genotypes.
        '''
        sample_cn = self.variant_pscn.agcn if self.variant_pscn.agcn_known else 2
        variant = self.variant_obs.variant
        allele_probs = _calculate_allele_probs(variant, mean_read_len)
        pooled_gt_counts = all_gt_counts(len(variant.alleles), sample_cn)

        scaled_pooled_gt_counts = [allele_probs * gt_counts for gt_counts in pooled_gt_counts]
        _, self.start_pooled_genotype_probs = genotype_likelihoods(sample_cn, self.good_allele_counts,
            error_prob=varcall_params.error_rate[self.variant_obs.is_indel], gt_counts=scaled_pooled_gt_counts)
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

        if self.variant_obs.psv_priors_matrix is None:
            for i, gt in enumerate(self.paralog_genotypes):
                self.paralog_genotype_probs[i] = gt.no_psv_prior(varcall_params.mutation_rate)
        else:
            self.variant_obs.calculate_psv_paralog_priors(self.variant_pscn.ext_to_var,
                self.paralog_genotypes, self.paralog_genotype_probs)

        self.paralog_genotype_probs -= logsumexp(self.paralog_genotype_probs)
        _, paralog_priors_str = _genotype_probs_to_str(self.paralog_genotypes, self.paralog_genotype_probs)
        if out is not None:
            out.write('{}\t{}\tparalog_priors\t{}\n'.format(
                self.sample, self.variant_obs.variant.pos, paralog_priors_str))

    def paralog_genotype_qualities(self, ext_copy_i):
        """
        Returns possible marginal paralog-specific genotypes for the extended copy `i` and their qualities.
        """
        if self.paralog_genotypes is None or ext_copy_i is None:
            return None, None
        return MultiCopyGT.aggregate_genotypes(self.paralog_genotypes, self.paralog_genotype_probs, ext_copy_i)

    def _add_read(self, curr_read_positions, allele_obs, n_copies, log_pscn_frac):
        read_pos, read_pos_probs = curr_read_positions.get_mate_pos_probs(allele_obs.is_first_mate)
        n_read_pos = len(read_pos)
        ext_read_pos_probs = np.full(n_copies, -np.inf)
        for i, variant_pos in enumerate(self.variant_obs.variant_positions):
            ext_pos_i = self.variant_pscn.var_to_ext[i]
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

    def utilize_reads(self, varcall_params, out=None):
        if self.paralog_genotypes is None:
            return

        variant = self.variant_obs.variant
        n_alleles = len(variant.alleles)
        error_rate = varcall_params.error_rate[self.variant_obs.is_indel]
        for gt in self.paralog_genotypes:
            gt.precompute_read_obs_probs(n_alleles, error_rate)

        n_copies = len(self.variant_pscn.extended_cn)
        ext_cn = np.maximum(self.variant_pscn.extended_cn, 0.01)
        log_pscn_frac = np.log(ext_cn) - np.log(np.sum(ext_cn))
        for paired_read_position, allele_obs in self.read_obs:
            self._add_read(paired_read_position, allele_obs, n_copies, log_pscn_frac)
        self.paralog_genotype_probs -= logsumexp(self.paralog_genotype_probs)

        _, paralog_str = _genotype_probs_to_str(self.paralog_genotypes, self.paralog_genotype_probs)
        pooled_genotypes, pooled_genotype_probs = MultiCopyGT.aggregate_genotypes(
            self.paralog_genotypes, self.paralog_genotype_probs, None)
        best_pooled, pooled_str = _genotype_probs_to_str(pooled_genotypes, pooled_genotype_probs)
        if out is not None:
            out.write('{}\t{}\tparalog_probs\t{}\n'.format(self.sample, variant.start + 1, paralog_str))
            out.write('{}\t{}\tpooled_probs\t{}\n'.format(self.sample, variant.start + 1, pooled_str))
        self.pooled_genotype = pooled_genotypes[best_pooled]
        self.pooled_genotype_qual = int(common.extended_phred_qual(
            pooled_genotype_probs, best_pooled, rem_prob=self.remaining_prob))

    @staticmethod
    def select_non_conflicing_psvs(gt_preds, max_mate_dist, error_rate, debug_out=None,
            one_side_neighbors=10, threshold_pval=-3):
        """
        Out of the full set of potentially informative PSVs,
        select a subset of non-conflicting informative PSVs.
        For each PSV, examine at most `one_side_neighbors` PSVs to the left and to the right, if they are within
        the `max_mate_dist` of each other.
        Two PSVs are in conflict if the number of conflicting reads is too high:
            log10(p-value) < threshold_pval, where p-value is calculated using the Binomial distribution with
            p = 2*error_rate - error_rate^2.
        """
        n_psvs = len(gt_preds)
        if not n_psvs:
            return gt_preds

        # Convert from log10 to natural log.
        threshold_pval *= common.LOG10
        binom_prob = error_rate * (2 - error_rate)
        assert 0 < binom_prob < 1
        read_obs_dicts = [
            { read_pos.hash: allele_obs.allele_ix for read_pos, allele_obs in gt_pred.read_obs }
            for gt_pred in gt_preds
        ]

        adj_matr = np.ones((n_psvs, n_psvs), dtype=np.bool_)
        any_conflicts = False
        for i in range(n_psvs - 1):
            gt_pred1 = gt_preds[i]
            pos1 = gt_pred1.variant_obs.start
            read_obs1 = read_obs_dicts[i]

            for j in range(i + 1, min(n_psvs, i + one_side_neighbors + 1)):
                gt_pred2 = gt_preds[j]
                pos2 = gt_pred2.variant_obs.start
                if pos2 - pos1 > max_mate_dist:
                    break
                read_obs2 = read_obs_dicts[j]
                if gt_pred1._has_conflict(read_obs1, gt_pred2, read_obs2, max_mate_dist, binom_prob, threshold_pval,
                        debug_out):
                    adj_matr[i, j] = adj_matr[j, i] = False
                    any_conflicts = True

        if not any_conflicts:
            return gt_preds

        min_fvals = np.array([np.min(gt_pred.variant_obs.variant.f_values) for gt_pred in gt_preds])
        fitness = np.sqrt(1.0 - min_fvals)
        retain_nodes = _find_maximal_clique(adj_matr, fitness)
        if debug_out:
            for i in np.where(~retain_nodes)[0]:
                psv = gt_preds[i].variant_obs.variant
                debug_out.write('    Remove PSV #{:4d} {},   {:2d} conflicts,  f-values {}\n'.format(
                    psv.psv_ix, psv.start + 1, n_psvs - np.sum(adj_matr[i]), psv.f_values))
        return list(itertools.compress(gt_preds, retain_nodes))

    def _has_conflict(self, read_obs1, gt_pred2, read_obs2, max_mate_dist, binom_prob, threshold_pval, debug_out):
        common_hashes = set(read_obs1) & set(read_obs2)
        total = len(common_hashes)
        if not total:
            return False

        psv1 = self.variant_obs.variant
        psv2 = gt_pred2.variant_obs.variant
        conflict_alleles = np.ones((len(psv1.alleles), len(psv2.alleles)), dtype=np.bool_)
        for pos1 in psv1.psv_positions:
            for pos2 in psv2.psv_positions:
                if pos1.region.distance(pos2.region) <= max_mate_dist:
                    conflict_alleles[pos1.allele_ix, pos2.allele_ix] = False

        conflicts = 0
        for read_hash in common_hashes:
            allele_ix1 = read_obs1[read_hash]
            allele_ix2 = read_obs2[read_hash]
            conflicts += conflict_alleles[allele_ix1, allele_ix2]
        log_pval = binom.logsf(conflicts, total, binom_prob) if conflicts else 0
        has_conflict = log_pval <= threshold_pval
        if debug_out and (conflicts > 1 or log_pval <= threshold_pval):
            debug_out.write('    PSVs #{:4d} & #{:4d}:   conflicts: {:3d} / {:3d},   p-val {:6.2f} -> {} conflict\n'
                .format(psv1.psv_ix, psv2.psv_ix, conflicts, total, log_pval / common.LOG10,
                'has' if has_conflict else ' no'))
        return has_conflict


def _maximal_clique_subfn(compl_matr, n_edges, fitness, orig_ixs, retain_nodes):
    """
    Greedily finds maximal clique.
    n_edges: vector with number of edges for each node.
    """
    n = len(fitness)
    m = np.sum(n_edges > 0)

    while m:
        i = np.argmax(fitness * n_edges)
        assert n_edges[i] > 0
        # Remove i-th node.
        retain_nodes[orig_ixs[i]] = False
        n_edges -= compl_matr[i]
        n_edges[i] = 0
        m2 = np.sum(n_edges > 0)
        assert m2 < m
        m = m2


def _find_maximal_clique(adj_matr, fitness):
    """
    Finds an approximate maximal clique from a symmetric adjacency matrix.
    Goal of the function -> find a clique with minimum sum fitness.
    Returns vector of booleans (True for retained nodes).

    Works by greedily removing nodes with minimal (fitness * n_edges).
    """
    n = len(fitness)
    assert adj_matr.shape == (n, n)
    compl_matr = ~adj_matr
    n_edges = np.sum(compl_matr, axis=1)
    retain_nodes = np.ones(n, dtype=np.bool_)

    sub_ixs = np.where(n_edges > 0)[0]
    _maximal_clique_subfn(compl_matr[np.ix_(sub_ixs, sub_ixs)], n_edges[sub_ixs], fitness[sub_ixs],
        sub_ixs, retain_nodes)
    return retain_nodes


class VarCallParameters:
    def __init__(self, args, samples):
        self.pooled_gt_thresh = args.limit_pooled * common.LOG10
        self.mutation_rate = args.mutation_rate * common.LOG10
        # self._set_use_af(args, samples) # TODO: Actually use AF.
        self.no_mate_penalty = args.no_mate_penalty * common.LOG10
        self.psv_ref_gt = args.psv_ref_gt
        self.skip_paralog_gts = args.skip_paralog

        # Two error rates and base_log_qual: first for SNPs, second for indels.
        self.error_rate = (args.error_rate[0], args.error_rate[1])
        assert 0 < self.error_rate[0] < 1 and 0 < self.error_rate[1] < 1
        self.base_log_qual = (-0.1 * args.base_qual[0] * common.LOG10, -0.1 * args.base_qual[1] * common.LOG10)

        self.min_read_depth = args.limit_depth[0]
        self.max_read_depth = args.limit_depth[1]
        self.max_strand_bias = args.strand_bias
        self.max_unpaired_bias = args.unpaired_bias
        self.min_freebayes_qual = args.limit_qual

    # def _set_use_af(self, args, samples):
    #     use_af_arg = args.use_af.lower()
    #     if use_af_arg in ('yes', 'y', 'true', 't'):
    #         self.use_af = True
    #     elif use_af_arg in ('no', 'n', 'false', 'f'):
    #         self.use_af = False
    #     elif use_af_arg.startswith('over-'):
    #         count = use_af_arg.split('-', 1)[-1]
    #         if not count.isdigit():
    #             raise ValueError('Cannot parse --use-af {}'.format(use_af_arg))
    #         self.use_af = len(samples) >= int(count)
    #     else:
    #         raise ValueError('Cannot parse --use-af {}'.format(use_af_arg))


def _process_overlapping_variants(variants):
    var_scores = np.zeros(len(variants))
    for i, variant in enumerate(variants):
        score = 0.0
        for fmt in variant.samples.values():
            gq = fmt.get('GQ')
            score += 0 if gq is None else gq
        var_scores[i] = score

    res_vars = []
    for i in np.argsort(-var_scores):
        var = variants[i]
        start = var.start
        end = start + len(var.ref)
        for var2 in res_vars:
            start2 = var2.start
            end2 = start2 + len(var2.ref)
            if start < end2 and start2 < end:
                break
        else:
            res_vars.append(var)
    res_vars.sort(key=itree.start)
    return res_vars


def merge_duplicates(variants):
    prev_chrom = None
    prev_end = None
    start_ix = 0

    merged_variants = []
    for i, variant in enumerate(variants):
        if variant.chrom != prev_chrom or variant.start >= prev_end:
            if i - start_ix > 1:
                merged_variants.extend(_process_overlapping_variants(variants[start_ix:i]))
            elif i:
                merged_variants.append(variants[i - 1])
            prev_chrom = variant.chrom
            prev_end = variant.start + len(variant.ref)
            start_ix = i
        else:
            prev_end = max(prev_end, variant.start + len(variant.ref))

    n = len(variants)
    if n - start_ix > 1:
        merged_variants.extend(_process_overlapping_variants(variants[start_ix:]))
    elif n:
        merged_variants.append(variants[n - 1])
    return merged_variants
