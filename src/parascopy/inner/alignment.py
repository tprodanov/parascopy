import itertools
import parasail

from . import common
from .genome import Interval
from . import cigar as cigar_


class Weights:
    def __init__(self, mismatch=4, open_gap=6, extend_gap=1):
        self._mismatch = 4
        self._open_gap = 6
        self._extend_gap = 1
        self._matrix = parasail.matrix_create("ACGT", 1, -self._mismatch)

    @property
    def mismatch(self):
        return self._mismatch

    @property
    def open_gap(self):
        return self._open_gap

    @property
    def extend_gap(self):
        return self._extend_gap

    def gap_penalty(self, length):
        return self._open_gap + length * self._extend_gap

    @property
    def matrix(self):
        return self._matrix

    def create_aln_fun(self):
        matrix = self._matrix.copy()
        open_gap = self._open_gap
        extend_gap = self._extend_gap

        def align(seq1, seq2):
            return parasail.nw_trace_diag_16(seq1, seq2, open_gap, extend_gap, matrix)
        return align

    def parasail_args(self):
        return (self._open_gap, self._extend_gap, self._matrix)


def _calculate_bwa_score(cigar, edit_dist, weights):
    """
    Calculates number of mismatches without the need to go through both sequences.
    Returns alignment score.
    """
    gap_penalties = 0
    total_gaps = 0
    n = len(cigar) - 1
    for i, (length, op) in enumerate(cigar):
        if i != 0 and i != n and not op.consumes_both():
            # Should be inner insertion or deletion.
            gap_penalties += weights.gap_penalty(length)
            total_gaps += length
    mismatches = edit_dist - total_gaps
    return cigar.aligned_len - weights.mismatch * mismatches - gap_penalties


class Alignment:
    def __init__(self, cigar, ref_interval, strand, stats=None):
        self._cigar = cigar
        self._ref_interval = ref_interval
        self._strand = strand
        self._stats = stats

    @classmethod
    def from_bwapy(cls, aln, genome, weights):
        cigar = cigar_.Cigar(aln.cigar)
        ref_id = genome.chrom_id(aln.rname)
        ref_start = aln.pos
        ref_end = ref_start + cigar.ref_len
        ref_interval = Interval(ref_id, ref_start, ref_end)

        strand = aln.orient == '+'
        edit_dist = aln.NM
        score = _calculate_bwa_score(cigar, edit_dist, weights)
        stats = ShortAlnStats(edit_dist=edit_dist, score=score, clipping=(0, 0))
        return cls(cigar, ref_interval, strand, stats)

    @classmethod
    def from_record(cls, record, genome):
        ref_id = genome.chrom_id(record.reference_name)
        if record.reference_start == record.reference_end:
            common.log('ERROR: read %s has alignment of zero-length (%s:%d-%d)'
                % (record.query_name, record.reference_name, record.reference_start + 1, record.reference_end))
            return None
        ref_interval = Interval(ref_id, record.reference_start, record.reference_end)
        cigar = cigar_.Cigar.from_pysam_tuples(record.cigartuples)
        strand = not record.is_reverse

        stats = ShortAlnStats.from_record(record, cigar)
        return cls(cigar, ref_interval, strand, stats)

    @property
    def cigar(self):
        return self._cigar

    @property
    def strand(self):
        return self._strand

    @property
    def strand_str(self):
        return '+' if self._strand else '-'

    @property
    def ref_interval(self):
        return self._ref_interval

    @property
    def stats(self):
        return self._stats

    def replace_stats(self, stats):
        self._stats = stats

    def to_str(self, genome):
        res = 'Aln({}; {}; {}'.format(self._ref_interval.to_str(genome), self.strand_str,
            '*' if self._cigar is None else self._cigar.to_str())
        if self._stats:
            stats_str = self._stats.to_str('; ')
            if stats_str:
                res += '; ' + stats_str
        return res + ')'

    def to_short_str(self, genome):
        res = '{}:{}'.format(self._ref_interval.to_str(genome), self.strand_str)
        if self._stats:
            stats_str = self._stats.to_str(';')
            if stats_str:
                res += ';' + stats_str
        return res

    @property
    def score(self):
        return self._stats and self._stats.score


class ShortAlnStats:
    def __init__(self, clipping=None, edit_dist=None, score=None):
        self._clipping = clipping
        self._edit_dist = edit_dist
        self._score = score
        self._seq_similarity = None

    @classmethod
    def from_record(cls, record, cigar):
        edit_dist = record.get_tag('NM') if record.has_tag('NM') else None
        score = record.get_tag('AS') if record.has_tag('AS') else None
        clipping = cigar.get_clipping()
        return cls(clipping=clipping, edit_dist=edit_dist, score=score)

    @property
    def clipping(self):
        return self._clipping

    @property
    def edit_dist(self):
        return self._edit_dist

    @property
    def score(self):
        return self._score

    def update_from_cigar(self, cigar):
        self._seq_similarity = 1.0 - self.edit_dist / cigar.aligned_len
        self._clipping = cigar.get_clipping()

    @property
    def seq_similarity(self):
        return self._seq_similarity

    def to_dict(self):
        d = {}
        if self.edit_dist is not None:
            d['NM'] = '%d' % self.edit_dist
        if self._score is not None:
            d['AS'] = self._score
        if self._clipping is not None:
            d['clip'] = '%d,%d' % self._clipping
        if self._seq_similarity is not None:
            d['SS'] = '%.3f' % self._seq_similarity
        return d

    def __str__(self):
        return self.to_str()

    def to_str(self, sep=', '):
        return sep.join('%s=%s' % t for t in self.to_dict().items())


class AlnStats(ShortAlnStats):
    def __init__(self):
        self._mismatches = 0
        self._insertions = []
        self._deletions = []

    @property
    def edit_dist(self):
        return self._mismatches + sum(self._insertions) + sum(self._deletions)

    @property
    def gaps(self):
        return sum(self._insertions) + sum(self._deletions)

    def add_mismatches(self, length):
        self._mismatches += length

    def add_insertions(self, length):
        self._insertions.append(length)

    def add_deletions(self, length):
        self._deletions.append(length)

    @property
    def diff(self):
        return '%d,%d,%d' % (self._mismatches, sum(self._deletions), sum(self._insertions))

    def calculate_score(self, aligned_len, weights):
        self._score = aligned_len - weights.mismatch * self._mismatches
        self._score -= sum(weights.open_gap + weights.extend_gap * length
            for length in itertools.chain(self._insertions, self._deletions))

    def update_info(self, info):
        info['SS'] = '%.3f' % self._seq_similarity
        info['DIFF'] = self.diff
        info['NM'] = self.edit_dist

    def to_dict(self):
        d = super().to_dict()
        d['DIFF'] = self.diff
        return d
