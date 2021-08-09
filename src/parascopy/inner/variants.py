import sys
import operator
import itertools
import pysam
import numpy as np
from scipy.stats import poisson, multinomial
from scipy.special import logsumexp
from functools import lru_cache

from .cigar import Cigar, Operation
from .genome import Interval
from . import common


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


def _create_record(record, header, cigars, ref_start, ref_end):
    new_rec = header.new_record()
    new_rec.chrom = record.chrom
    new_rec.start = record.start + ref_start
    min_len = ref_end - ref_start

    alleles = [record.ref[ref_start:ref_end]]
    old_to_new = [0]
    for cigar, alt in zip(cigars, record.alts):
        read_start, read_end = cigar.read_region(ref_start, ref_end)
        min_len = min(min_len, read_end - read_start)
        new_alt = alt[read_start : read_end]
        try:
            old_to_new.append(alleles.index(new_alt))
        except ValueError:
            old_to_new.append(len(alleles))
            alleles.append(new_alt)

    # Removing excessive padding.
    if min_len > 1:
        prefix_len = common.common_prefix(*alleles)
        if prefix_len > 1:
            alleles = [allele[prefix_len - 1 : ] for allele in alleles]
            new_rec.start += prefix_len - 1
            min_len -= prefix_len - 1

    if min_len > 1:
        suffix_len = common.common_suffix(*alleles)
        if suffix_len == min_len:
            suffix_len -= 1
        if suffix_len:
            alleles = [allele[ : -suffix_len] for allele in alleles]

    new_rec.alleles = alleles
    new_rec.qual = record.qual
    for filt in record.filter:
        new_rec.filter.add(filt)

    if len(alleles) == len(cigars) + 1:
        copy_vcf_fields(record, new_rec)
    else:
        copy_vcf_fields(record, new_rec, old_to_new)
    return new_rec


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


def from_freebayes(record, header):
    """
    Converts freebayes vcf record (pysam.VariantRecord) into one/several clipped records. Returns iterator.
    """
    ref_len = len(record.ref)
    can_discard = [True] * ref_len

    cigars = tuple(Cigar(cigar_str) for cigar_str in record.info['CIGAR'])
    for cigar in cigars:
        ref_pos = 0
        for length, op in cigar:
            if op.consumes_both():
                # Match or mismatch
                if op == Operation.SeqMismatch:
                    for i in range(length):
                        can_discard[i + ref_pos] = False
                ref_pos += length
            else:
                # Gap
                assert ref_pos
                can_discard[ref_pos - 1] = False
                if op.consumes_ref():
                    for i in range(length):
                        can_discard[i + ref_pos] = False
                    ref_pos += length

    ref_start = None
    for ref_pos, discard in enumerate(can_discard):
        if discard:
            if ref_start is not None:
                yield _create_record(record, header, cigars, ref_start, ref_pos)
                ref_start = None
        elif ref_start is None:
            ref_start = ref_pos
    if ref_start is not None:
        yield _create_record(record, header, cigars, ref_start, ref_len)


class _AdditionalInfo:
    # 0: diploid, 1: psv, 2: pooled.
    def __init__(self, value):
        self._value = value

    @property
    def is_diploid(self):
        return self._value == 0

    @property
    def is_psv(self):
        return self._value == 1

    @property
    def is_pooled(self):
        return self._value == 2

    def __str__(self):
        if self._value == 0:
            return 'diploid'
        elif self._value == 1:
            return 'psv'

        if self.primary:
            return 'pooled(pri)'
        else:
            return 'pooled(sec)'

    @classmethod
    def diploid(cls):
        return _AdditionalInfo(0)

    @classmethod
    def psv(cls):
        return _AdditionalInfo(1)

    @classmethod
    def pooled(cls, index, primary):
        res = _AdditionalInfo(2)
        res.index = index
        res.primary = primary
        return res


def _store_multivariant(multivar, multivariants, primary_pooled, secondary_pooled):
    if multivar is None:
        return
    multivar_ix = len(multivariants)
    for info in multivar.pooled_infos():
        if info.primary:
            assert primary_pooled[info.index] is None
            primary_pooled[info.index] = multivar_ix
        else:
            secondary_pooled[info.index].add(multivar_ix)
    multivariants.append(multivar)


def construct_multivariants(diploid_variants, psvs, pooled_variants, genome):
    """
    Arguments:
        diploid_filename: str - path to diploid vcf file,
        psvs: list of PSV variants,
        pooled_variants: list of lists, each list contains several related variants.
    """
    # List of tuples (chrom_id, variant, additional_info).
    variants = []
    for variant in diploid_variants:
        variants.append((genome.chrom_id(variant.chrom), variant, _AdditionalInfo.diploid()))
    for psv in psvs:
        variants.append((genome.chrom_id(psv.chrom), psv, _AdditionalInfo.psv()))

    for i, variant_copies in enumerate(pooled_variants):
        for j, variant in enumerate(variant_copies):
            variants.append((genome.chrom_id(variant.chrom), variant, _AdditionalInfo.pooled(i, j == 0)))
    variants.sort(key=lambda tup: (tup[0], tup[1].start))

    multivariants = []
    # primary_pooled[pooled_ix] = multivar_ix.
    primary_pooled = [None] * len(pooled_variants)
    # secondary_pooled[pooled_ix] = set(multivar_ix).
    secondary_pooled = [set() for _ in range(len(pooled_variants))]

    current = None
    for chrom_id, variant, info in variants:
        if current is None or not current.intersects(chrom_id, variant):
            _store_multivariant(current, multivariants, primary_pooled, secondary_pooled)
            current = MultiVariant(chrom_id, variant, info)
        else:
            current.append(variant, info)
    _store_multivariant(current, multivariants, primary_pooled, secondary_pooled)

    for primary_ix, secondary_ixs in zip(primary_pooled, secondary_pooled):
        for secondary_ix in secondary_ixs:
            multivariants[primary_ix].add_copy(multivariants[secondary_ix])
    return multivariants


class MultiVariant:
    def __init__(self, chrom_id, variant, info):
        self._variants = [(variant, info)]
        self._interval = Interval(chrom_id, variant.start, variant.start + len(variant.ref))
        self._other_copies = []

    def intersects(self, chrom_id, variant):
        return self._interval.chrom_id == chrom_id and self._interval.end > variant.start

    def append(self, variant, info):
        self._variants.append((variant, info))
        self._interval = Interval(self._interval.chrom_id,
            self._interval.start, max(self._interval.end, variant.start + len(variant.ref)))

    def pooled_infos(self):
        for _, info in self._variants:
            if info.is_pooled:
                yield info

    def add_copy(self, copy):
        self._other_copies.append(copy)

    def high_qual_or_psv(self, qual=10):
        for variant, info in itertools.chain(self._variants, *(copy._variants for copy in self._other_copies)):
            if (variant.qual and variant.qual >= qual) or info.is_psv:
                return True
        return False

    def has_psv(self):
        return any(info.is_psv for _, info in self._variants)

    def write_to(self, outp):
        outp.write('    MultiVariant:\n')
        for variant, info in self._variants:
            outp.write('        %-12s  %s' % (info, variant))

        for i, copy in enumerate(self._other_copies):
            outp.write('    ---- Copy %d\n' % (i + 2))
            for variant, info in copy._variants:
                outp.write('        %-12s  %s' % (info, variant))
        outp.write('\n')

    def to_psv_multivariants(self, res):
        psv = None
        pooled = None
        for variant, info in self._variants:
            if info.is_psv:
                if psv is not None:
                    common.log('WARN: Several PSVs within a single multivariant at position %s:%d-%d' % (psv.chrom,
                        self._interval.start_1, self._interval.end))
                    res.append(PsvMultiVariant(variant, None, 'several'))
                    continue
                psv = variant
            elif info.is_pooled and info.primary:
                if pooled is not None:
                    common.log('WARN: Several PSVs within a single multivariant at position %s:%d-%d' % (psv.chrom,
                        self._interval.start_1, self._interval.end))
                    continue
                pooled = variant

        assert psv
        if pooled is None:
            res.append(PsvMultiVariant(psv, None, 'no_pooled'))
        elif pooled.pos != psv.pos or pooled.ref != psv.ref:
            res.append(PsvMultiVariant(psv, pooled, 'not_exact'))
        else:
            res.append(PsvMultiVariant(psv, pooled, 'success'))


class PsvMultiVariant:
    @staticmethod
    def from_multivariants(multivariants):
        res = []
        for multivar in multivariants:
            if multivar.has_psv():
                multivar.to_psv_multivariants(res)
        return res

    def __init__(self, psv, pooled, status):
        self._psv = psv
        self._pooled = pooled
        self._status = status
        self._alleles = list(psv.alleles)
        self._n_psv_alleles = len(self._alleles)
        self._pooled_to_psv = None

        if self._status == 'success':
            non_psv_alts = [alt for alt in pooled.alts if alt not in self._psv.alts]
            self._alleles += non_psv_alts
            var_num = { seq: i for i, seq in enumerate(self._alleles) }
            self._pooled_to_psv = [var_num[allele] for allele in pooled.alleles]

    @property
    def alts_str(self):
        res = ','.join(self._alleles[1:self._n_psv_alleles])
        if self._n_psv_alleles < len(self._alleles):
            res += ';%s' % ','.join(self._alleles[self._n_psv_alleles:])
        return res

    def _to_dict(self):
        return dict(chrom=self._psv.chrom, pos=self._psv.pos, ref=self._psv.ref, alts=self.alts_str,
            pos2=','.join(self._psv.info['pos2']), status=self._status)

    def write_matrix_row(self, writer):
        row = self._to_dict()
        if self._status != 'success':
            writer.writerow(row)
            return

        for sample, sample_dict in self._pooled.samples.items():
            if sample_dict['AD'][0] is None:
                continue
            coverage = [0] * len(self._alleles)
            for i, depth in enumerate(sample_dict['AD']):
                coverage[self._pooled_to_psv[i]] += depth

            coverage_str = ','.join(map(str, coverage[:self._n_psv_alleles]))
            if self._n_psv_alleles < len(self._alleles):
                coverage_str += ';%s' % ','.join(map(str, coverage[self._n_psv_alleles:]))
            row[sample] = coverage_str
        writer.writerow(row)


def gt_counts_from_pos2(pos2, n_alleles):
    if len(pos2) == 1:
        return (2, 2)
    res = [0] * n_alleles
    res[0] = 2
    for entry in pos2:
        res[int(entry.rsplit(':', 1)[-1])] += 2
    return tuple(res)


def _summarize_read_depth(exp_mean, obs_value):
    if obs_value < exp_mean:
        p_value = 2.0 * poisson.cdf(obs_value, exp_mean)
        side = '  d'
    else:
        p_value = 2.0 * (1.0 - poisson.cdf(obs_value, exp_mean))
        side = 'u  '
    if p_value > 0.05:
        side = ' n '
    return '%s:%.3e' % (side, p_value)


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


def _fill_psv_gts(sample_id, sample_cn, psv_infos, psv_counts, psv_start_ix, psv_end_ix):
    from . import paralog_cn

    for psv_ix in range(psv_start_ix, psv_end_ix):
        counts = psv_counts[psv_ix][sample_id]
        if counts.skip:
            continue

        psv_info = psv_infos[psv_ix]
        if psv_info.psv_gt_probs[sample_id] is not None:
            continue

        if sample_cn not in psv_info.precomp_datas:
            psv_info.precomp_datas[sample_cn] = paralog_cn._PrecomputedData(psv_info.allele_corresp, sample_cn)
        psv_genotypes = psv_info.precomp_datas[sample_cn].psv_genotypes
        _, probs = genotype_likelihoods(sample_cn, counts.allele_counts, gt_counts=psv_genotypes)
        probs -= logsumexp(probs)
        psv_info.psv_gt_probs[sample_id] = probs
        psv_info.sample_cns[sample_id] = sample_cn


def calculate_all_psv_gt_probs(region_group_extra):
    psv_ixs = region_group_extra.region_group.psv_ixs
    if len(psv_ixs) == 0:
        return
    common.log('    Calculating PSV genotype probabilities')
    psv_counts = region_group_extra.psv_read_counts
    psv_finder = region_group_extra.psv_finder
    psv_infos = region_group_extra.psv_infos
    ref_cn = region_group_extra.region_group.cn

    n_psvs = len(psv_counts)
    n_samples = len(psv_counts[0])

    for sample_id in range(n_samples):
        for sample_const_region in region_group_extra.sample_const_regions[sample_id]:
            reg_start = sample_const_region.region1.start
            reg_end = sample_const_region.region1.end
            sample_cn = sample_const_region.pred_cn
            psv_start_ix, psv_end_ix = psv_finder.select(reg_start, reg_end)
            _fill_psv_gts(sample_id, sample_cn, psv_infos, psv_counts, psv_start_ix, psv_end_ix)

        reliable_region = region_group_extra.sample_reliable_regions[sample_id]
        if reliable_region is not None:
            psv_start_ix, psv_end_ix = psv_finder.select(reliable_region.start, reliable_region.end)
            _fill_psv_gts(sample_id, ref_cn, psv_infos, psv_counts, psv_start_ix, psv_end_ix)


def calculate_support_matrix(region_group_extra):
    """
    Calculates probabilities of sample genotypes according to all individual PSVs (if they have f-values).
    """
    if not region_group_extra.has_f_values:
        return

    psv_finder = region_group_extra.psv_finder
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
