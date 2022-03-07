import operator
from collections import Counter, namedtuple
import parasail
import itertools
import os

from . import genome as genome_
from .genome import Interval
from .cigar import Cigar, Operation
from .alignment import Alignment
from . import errors
from . import common
from . import variants as variants_


def parse_info(info):
    # filter(bool, ...) removes empty entries,
    # split(..., 1) splits at most 1 time.
    return dict(entry.split('=', 1) for entry in filter(bool, info.split(';')))


class TangledRegion:
    def __init__(self, region1: Interval):
        self._region1 = region1

    def to_str(self, genome):
        return self._region1.to_bed(genome) + '\t*\t*\t*\t*\ttangled'

    def to_str_pretty(self, genome):
        return '{}\t{:,}\t{:,}\ttangled region'.format(
            self._region1.chrom_name(genome), self._region1.start, self._region1.end)

    @property
    def region1(self):
        return self._region1

    @property
    def is_tangled_region(self):
        return True

    def __repr__(self):
        return '{!r} (tangled)'.format(self._region1)


VariantPosition = namedtuple('VariantPosition', 'region strand sequence')


class Duplication:
    def __init__(self, region1: Interval, region2: Interval, strand: bool):
        self._region1 = region1
        self._region2 = region2
        self._strand = strand
        self._info = {}
        self._info['SEP'] = region2.start_distance(region1)

        self._seq1 = None
        # Not reversed-complimented even if the duplication has - strand.
        self._seq2 = None
        # Padding [left1, right1, left2, right2]
        self._padding_seq = None
        self._full_cigar = None

    @classmethod
    def from_tuple(cls, tup, genome):
        chrom_id1 = genome.chrom_id(tup[0])
        start1 = int(tup[1])
        end1 = int(tup[2])
        region1 = Interval(chrom_id1, start1, end1)
        if tup[4] == '*':
            return TangledRegion(region1)

        chrom_id2 = genome.chrom_id(tup[4])
        start2 = int(tup[5])
        end2 = int(tup[6])
        region2 = Interval(chrom_id2, start2, end2)

        strand = tup[3] == '+'
        dupl = cls(region1, region2, strand)
        dupl.info.update(entry.split('=', 1) for entry in filter(bool, tup[7].split(';')))
        return dupl

    @property
    def is_tangled_region(self):
        return False

    def aligned_region(self, region1):
        """
        Returns region2 that corresponds to region1. If the aligned region is empty, returns None.
        """
        assert region1.intersects(self._region1)
        if region1.contains(self._region1):
            return self._region2

        start1 = max(0, region1.start - self._region1.start)
        end1 = min(len(self._region1), region1.end - self._region1.start)
        start2, end2 = self._full_cigar.aligned_region(start1, end1)
        if end1 <= start1 or end2 <= start2:
            return None
        if self._strand:
            return Interval(self._region2.chrom_id, self._region2.start + start2, self._region2.start + end2)
        else:
            return Interval(self._region2.chrom_id, self._region2.end - end2, self._region2.end - start2)

    def aligned_pos(self, pos1):
        """
        Given pos1 returns rough pos2 estimate.
        """
        rel_pos1 = pos1 - self._region1.start
        if rel_pos1 < 0 or rel_pos1 >= len(self._region1):
            return None

        start2, end2 = self._full_cigar.aligned_region(rel_pos1, rel_pos1 + 1)
        if self._strand:
            return self._region2.start + start2
        else:
            return self._region2.end - end2

    def sub_duplication(self, region1):
        """
        Returns self if region1.contains(self.region1). Otherwise, returns new duplication instance
        with aligned subregions of self.region1 and self.region2, such that region1.contains(self.region1).
        """
        assert region1.intersects(self._region1)
        if region1.contains(self._region1):
            return self
        start1 = max(0, region1.start - self._region1.start)
        end1 = min(len(self._region1), region1.end - self._region1.start)

        start1, end1, start2, end2, full_cigar = self._full_cigar.subcigar(start1, end1)
        region1 = Interval(self._region1.chrom_id, self._region1.start + start1, self._region1.start + end1)
        if self._strand:
            region2 = Interval(self._region2.chrom_id, self._region2.start + start2, self._region2.start + end2)
        else:
            region2 = Interval(self._region2.chrom_id, self._region2.end - end2, self._region2.end - start2)

        new_dupl = Duplication(region1, region2, self._strand)
        full_cigar.init_index()
        new_dupl._full_cigar = full_cigar
        if self._seq1 and self._seq2:
            new_dupl.set_sequences(seq1=self._seq1[start1:end1], seq2=self._seq2[start2:end2])
        return new_dupl

    @property
    def region1(self):
        return self._region1

    @property
    def region2(self):
        return self._region2

    def regions(self):
        return (self._region1, self._region2)

    # Do not use property.setter to make it more explicit.
    def set_region1(self, region1):
        self._region1 = region1

    def set_region2(self, region2):
        self._region2 = region2

    @property
    def strand(self):
        return self._strand

    @property
    def strand_str(self):
        return '+' if self._strand else '-'

    @property
    def info(self):
        return self._info

    @property
    def full_cigar(self):
        return self._full_cigar

    def set_full_cigar(self, full_cigar):
        self._full_cigar = full_cigar

    def set_cigar_from_info(self):
        full_cigar = self._info['CIGAR']
        del self._info['CIGAR']
        if full_cigar != '.':
            self._full_cigar = Cigar(full_cigar)
            self._full_cigar.init_index()

    @property
    def seq1(self):
        return self._seq1

    @property
    def seq2(self):
        return self._seq2

    def padded_seq1(self):
        start = self._region1.start - len(self._padding_seq[0])
        padded_seq = self._padding_seq[0] + self._seq1 + self._padding_seq[1]
        return padded_seq, start

    @staticmethod
    def _padded_subseq(left_seq, main_seq, right_seq, start, end):
        main_len = len(main_seq)
        assert start <= end and start <= main_len and end >= 0
        res = ''
        if start < 0:
            assert start >= -len(left_seq)
            res += left_seq[start:]
        res += main_seq[max(0, start) : end]
        if end > main_len:
            assert end - main_len <= len(right_seq)
            res += right_seq[:end - main_len]
        assert len(res) == end - start
        return res

    def padded_subseq1(self, start, end):
        return self._padded_subseq(self._padding_seq[0], self._seq1, self._padding_seq[1], start, end)

    def padded_subseq2(self, start, end):
        return self._padded_subseq(self._padding_seq[2], self._seq2, self._padding_seq[3], start, end)

    def set_sequences(self, *, genome=None, seq1=None, seq2=None):
        """
        If seq1 and seq2 are not None, sets duplication sequences from them.
        Otherwise, fetches sequences from the genome.
        """
        assert (seq1 is None) == (seq2 is None)
        if seq1 is not None:
            assert len(seq1) == len(self._region1)
            assert len(seq2) == len(self._region2)
            self._seq1 = seq1
            self._seq2 = seq2
        else:
            self._seq1 = self._region1.get_sequence(genome)
            self._seq2 = self._region2.get_sequence(genome, strand=self._strand)

    def set_padding_sequences(self, genome, padding_size):
        if not padding_size:
            return
        padding_seq = []
        for region, strand in zip(self.regions(), (True, self._strand)):
            if region.start > 0:
                left = Interval(region.chrom_id, max(0, region.start - padding_size), region.start)
                left_seq = left.get_sequence(genome)
                n_size = padding_size - len(left)
                if n_size:
                    left_seq = 'N' * n_size + left_seq
            else:
                left_seq = 'N' * padding_size

            if region.end < genome.chrom_len(region.chrom_id):
                right = Interval(region.chrom_id, region.end, region.end + padding_size)
                right.trim(genome)
                right_seq = right.get_sequence(genome)
                n_size = padding_size - len(right)
                if n_size:
                    right_seq += 'N' * n_size
            else:
                right_seq = 'N' * padding_size

            if strand:
                padding_seq.extend((left_seq, right_seq))
            else:
                padding_seq.extend((common.rev_comp(right_seq), common.rev_comp(left_seq)))
        self._padding_seq = tuple(padding_seq)

    def to_str(self, genome):
        res = '{}\t{}\t{}\t'.format(self._region1.to_bed(genome), self.strand_str, self._region2.to_bed(genome))
        res += ';'.join(map('%s=%s'.__mod__, self._info.items()))
        if 'CIGAR' not in self._info and self._full_cigar is not None:
            if self._info:
                res += ';'
            res += 'CIGAR='
            res += str(self._full_cigar)
        return res

    def to_str_pretty(self, genome):
        if 'CIGAR' in self._info and self._full_cigar is None:
            self.set_cigar_from_info()
        res = '{}\t{:,}\t{:,}\t'.format(self._region1.chrom_name(genome), self._region1.start, self._region1.end)
        res += '{}\t'.format(self.strand_str)
        res += '{}\t{:,}\t{:,}\t'.format(self._region2.chrom_name(genome), self._region2.start, self._region2.end)
        res += '\t'.join(map('%s: %s'.__mod__, self._info.items()))
        res += '\tCIGAR: '
        res += self._full_cigar.to_str()
        return res

    def __repr__(self):
        return '{!r} {} {!r}'.format(self._region1, self.strand_str, self._region2)

    def align_variant(self, variant):
        """
        Aligns variant to region1 and returns position in the region2: VariantPosition(region, strand, sequence).
        If variant is on a reverse strand, sequence is reverse-complemented relative to the region,
        meaning that if a variant does not overlap a PSV, sequence will be equal to variant.ref.
        """
        var_ref_len = len(variant.ref)
        start1 = variant.start - self._region1.start
        end1 = variant.start + var_ref_len - self._region1.start
        if start1 < 0 or end1 > len(self._region1):
            raise errors.VariantOutOfBounds()

        alt_size_diff = max(len(alt) - var_ref_len for alt in variant.alts)
        start2, end2 = self._full_cigar.aligned_region(start1, end1, alt_size_diff)
        assert start2 >= 0 and end2 <= len(self._region2)
        if end2 <= start2:
            raise ValueError('Align variant failed: variant {}:{} ({}): start1 {}, end1 {}, start2 {}, end2 {}'
                .format(variant.chrom, variant.pos, ','.join(variant.alleles), start1, end1, start2, end2))

        seq2 = self._seq2[start2:end2]
        dupl_start = self._region2.start + start2 if self._strand else self._region2.end - end2
        var_region2 = Interval(self._region2.chrom_id, dupl_start, dupl_start + len(seq2))
        return VariantPosition(var_region2, self._strand, seq2)

    def convert_variant(self, variant, genome, new_header, chrom_seq2, chrom_seq2_shift):
        """
        Converts the variant from the first copy to the second copy of the duplication.
        chrom_seq2_shift and chrom_seq2 store sequence of the region2.chrom
        It is needed when moving the variant to the left. The sequence does not need to be from the whole chromosome,
        just cover the duplication and a bit to the left and right.
        """
        print('Convert variant')
        var_ref_len = len(variant.ref)
        start1 = variant.start - self._region1.start
        end1 = variant.start + var_ref_len - self._region1.start
        if start1 < 0 or end1 > len(self._region1):
            raise errors.VariantOutOfBounds()

        is_indel = any(len(alt) != var_ref_len for alt in variant.alts)
        alt_size_diff = max(len(alt) - var_ref_len for alt in variant.alts) if is_indel else 0
        start2, end2 = self._full_cigar.aligned_region(start1, end1, alt_size_diff)
        print('Start1 {}   end1 {}   start2 {}   end2 {}'.format(start1, end1, start2, end2))
        print('Is indel {}   alt_size_diff {}'.format(is_indel, alt_size_diff))

        if not self._strand and (is_indel or start2 == end2):
            print('Not strand & smth')
            remove_prefix = os.path.commonprefix(variant.alleles)
            new_prefix = self.padded_subseq2(end2 - 1, end2)
        elif start2 == end2:
            print('Start2 == end2')
            remove_prefix = 0
            new_prefix = self.padded_subseq2(start2 - 1, start2)
        else:
            remove_prefix = 0
            new_prefix = ''

        new_ref = new_prefix + common.cond_rev_comp(self._seq2[start2 + remove_prefix : end2], strand=self._strand)
        new_alleles = [new_ref]
        old_to_new = []
        for allele in variant.alleles:
            new_allele = new_prefix + common.cond_rev_comp(allele[remove_prefix : ], strand=self._strand)
            if new_allele == new_ref:
                old_to_new.append(0)
            else:
                old_to_new.append(len(new_alleles))
                new_alleles.append(new_allele)

        new_variant = new_header.new_record()
        new_variant.chrom = self._region2.chrom_name(genome)
        if self._strand:
            new_variant.start = self._region2.start + start2 - len(new_prefix)
        else:
            new_variant.start = self._region2.end - end2 - len(new_prefix)
        new_variant.alleles = new_alleles
        new_variant.qual = variant.qual
        for value in variant.filter:
            new_variant.filter.add(value)

        if all(i == j for i, j in enumerate(old_to_new)):
            old_to_new = None
        variants_.copy_vcf_fields(variant, new_variant, old_to_new)

        if is_indel:
            move_res = variants_.move_left(new_variant.start, new_variant.ref, new_variant.alts,
                chrom_seq2, chrom_seq2_shift)
            if move_res is not None:
                new_variant.start = move_res[0]
                new_variant.alleles = move_res[1]
        return new_variant

    def store_stats(self, stats=None):
        if stats is None:
            stats = self._full_cigar.calculate_stats()
        self._info['ALENGTH'] = self._full_cigar.aligned_len
        stats.update_info(self._info)

    def estimate_complexity(self):
        """
        Estimates complexity of the duplication and if it contains a tandem repeat.
        Sequence1 must be defined.
        """
        k = 11
        kmers = list(map(operator.itemgetter(1), genome_.kmers(self._seq1, k)))
        n = len(kmers)
        kmers_counter = Counter(kmers)
        n_unique = len(kmers_counter)
        self._info['compl'] = '{:.3f}'.format(n_unique / min(n, 4 ** k))
        self._info['av_mult'] = '{:.2f}'.format(sum(kmers_counter.values()) / n_unique)

    @property
    def complexity(self):
        if 'compl' in self._info:
            return float(self._info['compl'])

    def align_read(self, read_seq, aln, weights, calc_score=True, full_cigar=False):
        """
        Moves the read from region2 to region1 and returns pair (read_seq, Alignment).
        Returned read_seq is the same as input read_seq, but is reverse-complimented if the duplication has - strand.
        """
        assert self._region2.intersects(aln.ref_interval)
        parasail_args = weights.parasail_args()
        read_cigar = aln.cigar

        if self._strand:
            start2 = aln.ref_interval.start - self._region2.start
            end2 = aln.ref_interval.end - self._region2.start
        else:
            start2 = self._region2.end - aln.ref_interval.end
            end2 = self._region2.end - aln.ref_interval.start
            read_cigar = read_cigar.reversed()
            read_seq = common.rev_comp(read_seq)

        region2_len = len(self._region2)
        if calc_score:
            ref_seq = self._seq2[max(0, start2) : end2]
            if start2 < 0:
                ref_seq = self._padding_seq[2][start2:] + ref_seq
            if end2 > region2_len:
                ref_seq += self._padding_seq[3][:end2 - region2_len]

            stats = read_cigar.calculate_stats_with_seqs(read_seq, ref_seq)
            stats.calculate_score(read_cigar.aligned_len, weights)
            aln.replace_stats(stats)
            no_gaps = stats.gaps == 0
            left_clip, right_clip = stats.clipping
        else:
            no_gaps = read_cigar.no_gaps()
            left_clip, right_clip = read_cigar.get_clipping()

        if no_gaps:
            if start2 < 0:
                left_clip -= start2
                start2 = 0
            if end2 > region2_len:
                right_clip += end2 - region2_len
                end2 = region2_len
            cigar, start1, end1 = self._align_perfect_read(read_seq, (left_clip, right_clip),
                start2, end2, parasail_args)
        else:
            cigar, start1, end1 = self._align_nonperfect_read(read_seq, read_cigar, start2, parasail_args)
        if cigar is not None:
            cigar = Cigar.from_tuples(cigar)

        ref_interval = Interval(self._region1.chrom_id, self._region1.start + start1, self._region1.start + end1)
        new_aln_strand = aln.strand if self._strand else not aln.strand

        if full_cigar:
            cigar = cigar.to_full(self._seq1[start1:end1], read_seq)
        stats = None
        if cigar:
            assert cigar.read_len == len(read_seq)
            assert cigar.ref_len == len(ref_interval)
            if calc_score:
                stats = cigar.calculate_stats_with_seqs(read_seq, self._seq1[start1:end1])
                stats.calculate_score(cigar.aligned_len, weights)
        return read_seq, Alignment(cigar, ref_interval, new_aln_strand, stats)

    def _align_nonperfect_read(self, read_seq, read_cigar, start2, parasail_args):
        """
        Returns tuple (cigar_tuples, start1, end1).
        """
        read_region2_pairs = [(rpos, pos2 + start2) for rpos, pos2 in read_cigar.aligned_pairs()]
        i = 0
        n_pairs = len(read_region2_pairs)
        region1_positions = [None] * n_pairs
        last_i = 0

        for pos2, pos1 in self._full_cigar.aligned_pairs(read_start=start2):
            while i < n_pairs and read_region2_pairs[i][1] < pos2:
                i += 1
            if i == n_pairs:
                break

            if read_region2_pairs[i][1] == pos2:
                region1_positions[i] = pos1
                i += 1
        try:
            first_i = next(i for i, el in enumerate(region1_positions) if el is not None)
        except StopIteration:
            return None, pos1, pos1 + 1

        rpos, pos2 = read_region2_pairs[first_i]
        pos1 = region1_positions[first_i]
        exp_rpos = rpos
        exp_pos1 = pos1

        new_cigar, start1 = self._add_left_clipping(read_seq, rpos, pos1, parasail_args)

        # Middle alignment
        for (rpos, pos2), pos1 in zip(read_region2_pairs, region1_positions):
            if pos1 is None:
                continue
            if rpos != exp_rpos or pos1 != exp_pos1:
                if rpos == exp_rpos:
                    Cigar.append(new_cigar, pos1 - exp_pos1, Operation.Deletion)
                elif pos1 == exp_pos1:
                    Cigar.append(new_cigar, rpos - exp_rpos, Operation.Insertion)
                else:
                    rseq = read_seq[exp_rpos : rpos]
                    seq1 = self._seq1[exp_pos1 : pos1]
                    # Run Needleman-Wunsch global alignment.
                    subaln = parasail.nw_trace_scan_sat(rseq, seq1, *parasail_args)
                    for value in subaln.cigar.seq:
                        length = value >> 4
                        op_num = value & 0xf
                        op = Operation.AlnMatch if op_num >= 7 else Operation(op_num)
                        Cigar.append(new_cigar, length, op)

            Cigar.append(new_cigar, 1, Operation.AlnMatch)
            exp_rpos = rpos + 1
            exp_pos1 = pos1 + 1

        seq1 = self._seq1[start1 : exp_pos1]

        new_cigar = _move_everything_left(new_cigar, start1=start1, seq1=seq1, start2=0, seq2=read_seq, move_dupl=False)
        new_cigar = _replace_del_ins_signature(new_cigar, read_seq, seq1, parasail_args)

        # Right clipping
        end1 = self._add_right_clipping(new_cigar, read_seq, len(read_seq) - exp_rpos, exp_pos1, parasail_args)
        return new_cigar, start1, end1

    def _align_perfect_read(self, read_seq, clipping, start2, end2, parasail_args):
        """
        Returns tuple (cigar_tuples, start1, end1).
        In the returned cigar, read and reference will switch places.

        If there is no alignment, the function returns (None, start1, end1),
        where start1 and end1 are approximate positions.
        """
        cigar_start, start1, pos2 = self._full_cigar.index.find_by_read(start2)
        new_cigar = []
        for length, op in self._full_cigar.iter_from(cigar_start):
            if op.consumes_both():
                out_length = min(end2, pos2 + length) - max(start2, pos2)
                if out_length > 0:
                    Cigar.append(new_cigar, out_length, Operation.AlnMatch)
                if start2 > pos2:
                    start1 += min(start2 - pos2, length)
                pos2 += length

            elif op.consumes_read():
                out_length = min(end2, pos2 + length) - max(start2, pos2)
                if out_length > 0:
                    new_op = Operation.Insertion if new_cigar else Operation.Soft
                    Cigar.append(new_cigar, out_length, new_op)
                pos2 += length

            else:
                assert op.consumes_ref()
                if new_cigar:
                    Cigar.append(new_cigar, length, Operation.Deletion)
                else:
                    start1 += length

            if pos2 >= end2:
                break
        # Cannot use new_cigar.read_len because new_cigar is not Cigar class yet.
        end1 = start1 + sum(length for length, op in new_cigar if op.consumes_ref())

        first_len, first_op = new_cigar[0]
        tmp_left_clip = first_len if first_op == Operation.Soft else 0
        left_clip = clipping[0] + tmp_left_clip
        if left_clip > 0:
            # Need to remove first soft operation, tmp_left_clip > 0
            tmp_cigar = new_cigar[1:] if tmp_left_clip else new_cigar
            if not tmp_cigar:
                # Reads completely inside an insertion, cannot align it to the other copy.
                return None, start1, start1 + 1

            new_cigar, start1 = self._add_left_clipping(read_seq, left_clip, start1, parasail_args)
            Cigar.extend(new_cigar, iter(tmp_cigar))

        last_len, last_op = new_cigar[-1]
        tmp_right_clip = last_len if not last_op.consumes_ref() else 0
        right_clip = clipping[1] + tmp_right_clip
        if right_clip > 0:
            # Need to remove last soft operation, tmp_right_clip > 0
            if tmp_right_clip:
                new_cigar.pop()
            end1 = self._add_right_clipping(new_cigar, read_seq, right_clip, end1, parasail_args)
        return new_cigar, start1, end1

    def _add_left_clipping(self, read_seq, clipping, first_pos1, parasail_args):
        """
        Returns (cigar tuples of the padding, start1: int).
        If `clipping` is zero, returns ([], first_pos1).
        """
        if clipping == 0:
            return [], first_pos1
        if first_pos1 == 0:
            return [(clipping, Operation.Soft)], first_pos1

        # Need to reverse strings to use in semi-global alignment.
        rseq = read_seq[clipping - 1::-1]
        start1 = max(0, first_pos1 - clipping - _SOFT_PADDING)
        seq1 = self._seq1[start1 : first_pos1]
        seq1 = seq1[::-1]

        new_cigar, pos1_add = _align_semi_global(rseq, seq1, parasail_args)
        return new_cigar[::-1], start1 + pos1_add

    def _add_right_clipping(self, new_cigar, read_seq, clipping, last_pos1, parasail_args):
        """
        Extends new_cigar and returns end1.
        """
        if clipping == 0:
            return last_pos1
        if last_pos1 == len(self._region1):
            Cigar.append(new_cigar, clipping, Operation.Soft)
            return last_pos1

        rseq = read_seq[-clipping : ]
        end1 = min(len(self._region1), last_pos1 + clipping + _SOFT_PADDING)
        seq1 = self._seq1[last_pos1 : end1]

        cigar_ext, pos1_sub = _align_semi_global(rseq, seq1, parasail_args)
        Cigar.extend(new_cigar, iter(cigar_ext))
        return end1 - pos1_sub

    def subregion2(self, start1, end1):
        """
        Return Interval - part of region2 that corresponds to the region1[start1:end1].
        Returns None if the subregion is empty.
        """
        reg_start1 = self._region1.start
        start2, end2 = self._full_cigar.read_region(start1 - reg_start1, end1 - reg_start1)
        if start2 == end2:
            return None
        if self._strand:
            return Interval(self._region2.chrom_id, self._region2.start + start2, self._region2.start + end2)
        else:
            return Interval(self._region2.chrom_id, self._region2.end - end2, self._region2.end - start2)

    def revert(self):
        """
        Returns duplication for which region1 in region2 are interchanged.
        """
        dupl = Duplication(self._region2, self._region1, self._strand)
        if self._full_cigar:
            dupl._full_cigar = self._full_cigar.revert(self._strand)
            dupl._full_cigar.init_index()

        if self._seq1:
            dupl.set_sequences(
                seq1=common.cond_rev_comp(self._seq2, strand=self._strand),
                seq2=common.cond_rev_comp(self._seq1, strand=self._strand))
        if self._padding_seq:
            if self._strand:
                dupl._padding_seq = (self._padding_seq[2], self._padding_seq[3],
                    self._padding_seq[0], self._padding_seq[1])
            else:
                dupl._padding_seq = (
                    common.rev_comp(self._padding_seq[3]), common.rev_comp(self._padding_seq[2]),
                    common.rev_comp(self._padding_seq[1]), common.rev_comp(self._padding_seq[0]))

        _copy_values(self._info, dupl._info, ['SEP', 'ALENGTH', 'SS', 'NM', 'compl', 'av_mult', 'clip'])
        if 'DIFF' in self._info:
            mism, deletions, insertions = self._info['DIFF'].split(',')
            dupl._info['DIFF'] = '{},{},{}'.format(mism, insertions, deletions)
        return dupl


def _replace_del_ins_signature(cigar, read_seq, ref_seq, parasail_args):
    """
    If CIGAR has a region aI bM cD or aD bM cI, where a>=b or c>=b, aligns this region using parasail.
    """
    i = 0
    n = len(cigar)

    read_pos = 0
    ref_pos = 0

    while i < n - 2:
        length, op = cigar[i]
        # Here, operations must be 0 (match), 1 (insertion), 2 (deletion) or 4 (soft clipping)
        # We need operations[i] + operations[i + 2] == 3 and operations[i + 1] == 0
        if op.value + cigar[i + 2][1].value + 10 * cigar[i + 1][1].value != 3 \
                or (length < cigar[i + 1][0] and cigar[i + 2][0] < cigar[i + 1][0]):
            if op.consumes_read():
                read_pos += length
            if op.consumes_ref():
                ref_pos += length
            i += 1
            continue

        if op.consumes_read():
            read_pos2 = read_pos + length
            ref_pos2 = ref_pos
        else:
            # Must be either deletion or insertion.
            read_pos2 = read_pos
            ref_pos2 = ref_pos + length

        # Must be match.
        length2 = cigar[i + 1][0]
        read_pos2 += length2
        ref_pos2 += length2

        length3, op3 = cigar[i + 2]
        if op3.consumes_read():
            read_pos2 += length3
        else:
            # Must be either deletion or insertion.
            ref_pos2 += length3

        new_cigar = cigar[:i]
        aln = parasail.nw_trace_diag_16(read_seq[read_pos : read_pos2], ref_seq[ref_pos : ref_pos2], *parasail_args)
        for value in aln.cigar.seq:
            new_length = value >> 4
            op_num = value & 0xf
            new_op = Operation.AlnMatch if op_num >= 7 else Operation(op_num)
            Cigar.append(new_cigar, new_length, new_op)

        Cigar.extend(new_cigar, iter(cigar[i + 3:]))
        cigar = new_cigar
        n = len(cigar)
        if i >= n:
            return cigar
        length, op = cigar[i]
        if op.consumes_read():
            read_pos += length
        if op.consumes_ref():
            ref_pos += length
        i += 1
    return cigar


def _copy_values(source, target, keys):
    for key in keys:
        if key in source:
            target[key] = source[key]


# When searching for the alignment in the ends of the reads, add this additional padding to the reference sequence.
_SOFT_PADDING = 2


def _get_regions_and_seqs(segments):
    """
    Returns strand, region1, region2, seq1, seq2.
    """
    first_segment = segments[0]
    strand = first_segment.strand
    chrom_id1 = first_segment.region1.chrom_id
    start1 = first_segment.region1.start
    end1 = first_segment.region1.end
    seq1 = first_segment.seq1

    chrom_id2 = first_segment.region2.chrom_id
    start2 = first_segment.region2.start if strand else -first_segment.region2.end
    end2 = first_segment.region2.end if strand else -first_segment.region2.start
    seq2 = first_segment.seq2

    for segment in itertools.islice(segments, 1, None):
        assert strand == segment.strand \
            and chrom_id1 == segment.region1.chrom_id and chrom_id2 == segment.region2.chrom_id
        curr_start1 = segment.region1.start
        curr_end1 = segment.region1.end
        curr_start2 = segment.region2.start if strand else -segment.region2.end
        curr_end2 = segment.region2.end if strand else -segment.region2.start
        if curr_start1 > end1 or curr_start2 > end2:
            for s in segments:
                common.log('{!r}   {!r}'.format(s.region1, s.region2))
        assert curr_start1 <= end1 and curr_start2 <= end2

        if end1 < curr_end1:
            seq1 += segment.seq1[-(curr_end1 - end1) : ]
            end1 = curr_end1
        if end2 < curr_end2:
            seq2 += segment.seq2[-(curr_end2 - end2) : ]
            end2 = curr_end2

    region1 = Interval(chrom_id1, start1, end1)
    if strand:
        region2 = Interval(chrom_id2, start2, end2)
    else:
        region2 = Interval(chrom_id2, -end2, -start2)

    assert len(seq1) == len(region1)
    assert len(seq2) == len(region2)
    return strand, region1, region2, seq1, seq2


def _glue_segments(cigar, aln_fun, seq1, seq2, shift1, shift2, a_end1, a_end2, b_start1, b_start2):
    if b_start1 > a_end1 and b_start2 > a_end2:
        assert a_end1 >= shift1 and a_end2 >= shift2
        inter_seq1 = seq1[a_end1 - shift1 : b_start1 - shift1]
        inter_seq2 = seq2[a_end2 - shift2 : b_start2 - shift2]
        aln = aln_fun(inter_seq2, inter_seq1)
        Cigar.extend(cigar, ((value >> 4, Operation(value & 0xf)) for value in aln.cigar.seq))
    elif b_start1 > a_end1:
        Cigar.append(cigar, b_start1 - a_end1, Operation.Deletion)
    elif b_start2 > a_end2:
        Cigar.append(cigar, b_start2 - a_end2, Operation.Insertion)


def _move_everything_left(cigar, start1, seq1, start2, seq2, move_dupl=True):
    pos1 = move_stop1 = shift1 = start1
    pos2 = shift2 = start2
    new_cigar = []
    match_op = Operation.SeqMatch if move_dupl else Operation.AlnMatch

    for length, op in cigar:
        if op.consumes_both():
            pos1 += length
            pos2 += length
            if move_dupl:
                if op != Operation.SeqMatch:
                    # -1 because we can allow insertions/deletions that start with a mismatch.
                    move_stop1 = pos1 - 1
            else:
                for i in range(1, length + 1):
                    if seq1[pos1 - i - shift1] != seq2[pos2 - i]:
                        move_stop1 = pos1 - i
                        break

            Cigar.append(new_cigar, length, op)
            continue

        new_start = None
        is_insertion = op.consumes_read()
        if pos1 > move_stop1 + 1:
            var_start1 = pos1 - 1
            assert var_start1 >= shift1
            var_end1 = pos1 if is_insertion else pos1 + length
            var_seq1 = seq1[var_start1 - shift1 : var_end1 - shift1]

            var_start2 = pos2 - 1
            assert var_start2 >= shift2
            var_end2 = pos2 + length if is_insertion else pos2
            var_seq2 = seq2[var_start2 - shift2 : var_end2 - shift2]
            new_start = variants_.move_left(var_start1, var_seq1, (var_seq2,), seq1, shift1,
                min_start=move_stop1, skip_alleles=True)

        if new_start is None:
            Cigar.append(new_cigar, length, op)
        else:
            shift = var_start1 - new_start
            last_len, last_op = new_cigar.pop()
            assert last_op == match_op and last_len >= shift > 0

            if last_len > shift:
                Cigar.append(new_cigar, last_len - shift, last_op)
            Cigar.append(new_cigar, length, op)
            Cigar.append(new_cigar, shift, match_op)

        if is_insertion:
            pos2 += length
        else:
            pos1 += length
        move_stop1 = pos1
    return new_cigar


def combine_segments(segments, aln_fun):
    """
    Segments: small Duplications.
    Segments must be sorted by both region1 and region2 (regions2 reverse if on reverse strand).
    Additionally, all cigars must start and end with a match.
    Sequences must be defined (call `.set_sequences(genome)`).

    Returns combined duplication.
    """
    strand, region1, region2, seq1, seq2 = _get_regions_and_seqs(segments)
    shift1 = pos1 = region1.start
    shift2 = pos2 = region2.start if strand else -region2.end

    new_cigar = []
    store_clipping = True
    left_clip = None
    right_clip = None
    for dupl_i, dupl in enumerate(segments):
        start1 = dupl.region1.start
        end1 = dupl.region1.end
        start2 = dupl.region2.start if strand else -dupl.region2.end
        end2 = dupl.region2.end if strand else -dupl.region2.start
        if end1 <= pos1 or end2 <= pos2:
            continue

        if dupl_i + 1 < len(segments):
            next_dupl = segments[dupl_i + 1]
            stop1 = max(shift1, (end1 + next_dupl.region1.start) // 2)
            stop2 = max(shift2, (end2 + (next_dupl.region2.start if strand else -dupl.region2.end)) // 2)
        else:
            stop1 = end1
            stop2 = end2

        cigar = dupl.full_cigar
        cigar_len = len(cigar)
        assert cigar[0][1].consumes_both()
        assert cigar[-1][1].consumes_both()

        cigar_ix = 0
        while cigar_ix < cigar_len and (pos1 > start1 or pos2 > start2):
            length, op = cigar[cigar_ix]
            cigar_ix += 1
            if op.consumes_both():
                diff = pos1 - start1
                if diff < length and diff == pos2 - start2:
                    ext = length - diff
                    Cigar.append(new_cigar, ext, op)
                    pos1 += ext
                    pos2 += ext
                start1 += length
                start2 += length

            elif op.consumes_ref():
                start1 += length
            elif op.consumes_read():
                start2 += length
        if cigar_ix == cigar_len:
            continue

        _glue_segments(new_cigar, aln_fun, seq1, seq2, shift1, shift2, pos1, pos2, start1, start2)
        pos1 = start1
        pos2 = start2

        while cigar_ix < cigar_len and (pos1 < stop1 and pos2 < stop2):
            length, op = cigar[cigar_ix]
            Cigar.append(new_cigar, length, op)
            cigar_ix += 1
            if op.consumes_both():
                pos1 += length
                pos2 += length
            elif op.consumes_ref():
                pos1 += length
            elif op.consumes_read():
                pos2 += length

        if 'clip' not in dupl.info:
            store_clipping = False
        elif store_clipping:
            # elif because store_clipping could be set False on previous iterations.
            clipping = dupl.info['clip'].split(',')
            if left_clip is None:
                left_clip = int(clipping[0])
            right_clip = int(clipping[1])

    new_cigar = _move_everything_left(new_cigar, shift1, seq1, shift2, seq2)

    region1 = Interval(region1.chrom_id, shift1, pos1)
    if strand:
        region2 = Interval(region2.chrom_id, shift2, pos2)
    else:
        region2 = Interval(region2.chrom_id, -pos2, -shift2)
    dupl = Duplication(region1, region2, strand)
    seq1 = seq1[:pos1 - shift1]
    seq2 = seq2[:pos2 - shift2]
    dupl.set_sequences(seq1=seq1, seq2=seq2)
    full_cigar = Cigar.from_tuples(new_cigar)
    full_cigar.init_index()

    assert full_cigar.ref_len == len(region1)
    assert full_cigar.read_len == len(region2)
    dupl._full_cigar = full_cigar

    if store_clipping:
        assert left_clip is not None
        dupl.info['clip'] = '{},{}'.format(left_clip, right_clip)
    return dupl


def _align_semi_global(seq1, seq2, parasail_args):
    """
    Currently, parasail does not have semi-global alignment without penalties in the ends/beginnings of both sequences.
    This function generates semi-global alignment without penalties in the end:
        - first we align two sequence using global NW alignment,
        - we find position we the best score,
        - use NW alignment to generate alignment of the corresponding subsequences.

    Returns pair `(cigar: list of (length, op), seq2_clipping: int)`.
    """
    # NW: global alignment, table: generate score matrix, scan: ??, sat: use 8 bits, switch to 16 if needed.
    aln = parasail.nw_table_scan_sat(seq1, seq2, *parasail_args)
    table = aln.score_table
    n, m = table.shape
    best_i = 0
    best_j = 0
    best = 0
    for i in range(n):
        for j in range(m):
            if table[i, j] > best:
                best = table[i, j]
                best_i = i + 1
                best_j = j + 1

    cigar = []
    if best_i != 0:
        aln2 = parasail.nw_trace_scan_sat(seq1[:best_i], seq2[:best_j], *parasail_args)
        for value in aln2.cigar.seq:
            length = value >> 4
            op_num = value & 0xf
            op = Operation.AlnMatch if op_num >= 7 else Operation(op_num)
            Cigar.append(cigar, length, op)

    if best_i < n:
        cigar.append((n - best_i, Operation.Soft))
    return (cigar, m - best_j)
