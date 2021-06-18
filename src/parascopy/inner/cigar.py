from enum import Enum
import operator
import itertools
import collections
import sys
import numpy as np
import re

from .alignment import AlnStats


class Operation(Enum):
    AlnMatch = 0
    Insertion = 1
    Deletion = 2
    Skip = 3
    Soft = 4
    Hard = 5
    Padding = 6
    SeqMatch = 7
    SeqMismatch = 8

    def consumes_read(self):
        return _consumes_read[self.value]

    def consumes_ref(self):
        return _consumes_ref[self.value]

    def consumes_both(self):
        return _consumes_both[self.value]

    def __str__(self):
        return _letters[self.value]


_letters = 'MIDNSHP=X'
_operations = dict((letter, Operation(num)) for num, letter in enumerate(_letters))
_consumes_read = [True, True, False, False, True, False, False, True, True]
_consumes_ref = [True, False, True, True, False, False, False, True, True]
_consumes_both = list(itertools.starmap(operator.and_, zip(_consumes_read, _consumes_ref)))


AlignedRegion = collections.namedtuple('AlignedRegion', 'start1 end1 start2 end2')


class Cigar:
    def __init__(self, line):
        self._tuples = []
        current = 0
        for char in line:
            if char.isdigit():
                current = current * 10 + int(char)
            else:
                self._tuples.append((current, _operations[char]))
                current = 0
        self._tuples = tuple(self._tuples)
        self.__init_lengths()
        self._index = None

    @classmethod
    def from_tuples(cls, tuples, lengths_from=None):
        self = cls.__new__(cls)
        self._tuples = tuple(tuples)
        if lengths_from:
            self._aligned_len = lengths_from._aligned_len
            self._ref_len = lengths_from._ref_len
            self._read_len = lengths_from._read_len
        else:
            self.__init_lengths()
        self._index = None
        return self

    @classmethod
    def from_pysam_tuples(cls, tuples):
        self = cls.__new__(cls)
        self._tuples = tuple((length, Operation(op)) for op, length in tuples)
        self.__init_lengths()
        self._index = None
        return self

    def init_index(self):
        self._index = CigarIndex(self)

    def __init_lengths(self):
        self._read_len = 0
        self._ref_len = 0
        self._aligned_len = 0
        for len, op in self._tuples:
            if op.consumes_read():
                self._read_len += len
            if op.consumes_ref():
                self._ref_len += len
            if op.consumes_both():
                self._aligned_len += len

    def reversed(self):
        """
        Returns reversed cigar.
        """
        return Cigar.from_tuples(self._tuples[::-1], self)

    def __iter__(self):
        return iter(self._tuples)

    def iter_from(self, start_ix):
        return itertools.islice(self._tuples, start_ix, len(self._tuples))

    @property
    def read_len(self):
        return self._read_len

    @property
    def ref_len(self):
        return self._ref_len

    @property
    def aligned_len(self):
        return self._aligned_len

    @property
    def index(self):
        return self._index

    """
    Returns i-th pair (length: int, operation: Operation).
    """
    def __getitem__(self, i):
        return self._tuples[i]

    """
    Finds aligned region in two sequences.

    Parameters:
    - ref_start: 0-based start of the region in the first sequence (should be >= 0),
    - ref_end: 0-based exclusive end of the region in the first sequence (should be <= ref_len),
    - alt_size_diff: difference between reference and alternative variants. Positive if alternative is longer.

    Returns pair `read_start, read_end`, output interval may be empty.

    TODO:
    Process case when the variant overlaps homopolymer: for example
        chr1:160696714-160696714        catatGtcatg
        chr22:18944546-18944546         catatGttcat
    can be stored as GT-, GTT or as G-T, GTT. We need to account for both cases.
    """
    def aligned_region(self, ref_start, ref_end, alt_size_diff):
        cigar_start, ref_pos, read_pos = self._index.find_by_ref(ref_start)
        read_start = None

        for cigar_ix in range(cigar_start, len(self._tuples)):
            length, op = self._tuples[cigar_ix]
            if op.consumes_ref():
                if read_start is None and ref_pos <= ref_start < ref_pos + length:
                    # Found start in the first sequence.
                    if op.consumes_read():
                        read_start = read_pos + ref_start - ref_pos
                    else:
                        # this position is deleted in the second sequence.
                        read_start = read_pos

                if ref_pos < ref_end <= ref_pos + length:
                    # Found end in the first sequence.
                    if op.consumes_read():
                        read_end = read_pos + ref_end - ref_pos
                    else:
                        read_end = read_pos

                    if alt_size_diff > 0 and ref_end - ref_pos == length and cigar_ix + 1 < len(self._tuples):
                        # Include case when the alternative variant is in the insertion in the second sequence.
                        next_len, next_op = self._tuples[cigar_ix + 1]
                        if not next_op.consumes_ref():
                            read_end += min(next_len, alt_size_diff)

                    return (read_start, read_end)
                ref_pos += length

            if op.consumes_read():
                read_pos += length
        assert False

    """
    Should be extended CIGAR: with X/= instead of M.
    Returns iterator, each element: `AlignedRegion`.
    """
    def find_differences(self):
        pos1 = 0
        pos2 = 0
        curr = None
        for length, op in self._tuples:
            if op.consumes_both():
                if op == Operation.SeqMismatch:
                    if curr:
                        yield curr
                    curr = AlignedRegion(start1=pos1, end1=pos1 + length, start2=pos2, end2=pos2 + length)
                elif op == Operation.AlnMatch:
                    raise ValueError('Cannot call find_differences() on non-extended CIGAR')

                pos1 += length
                pos2 += length

            # Both positions should be non-zero
            elif op.consumes_ref():
                assert pos1 and pos2
                if curr:
                    if curr.start1 < pos1 - 1:
                        yield curr
                yield AlignedRegion(start1=pos1 - 1, end1=pos1 + length, start2=pos2 - 1, end2=pos2)
                curr = None
                pos1 += length

            elif op.consumes_read():
                assert pos1 and pos2
                if curr:
                    if curr.start1 < pos1 - 1:
                        yield curr
                yield AlignedRegion(start1=pos1 - 1, end1=pos1, start2=pos2 - 1, end2=pos2 + length)
                curr = None
                pos2 += length
        if curr:
            yield curr

    def remove_clipping(self):
        """
        Returns tuple `(left_clipping, right_clipping, new_cigar)`.
        """
        left = 0
        right = 0
        new_tuples = []

        for length, op in self._tuples:
            if op == Operation.Soft or op == Operation.Hard:
                if new_tuples:
                    right += length
                else:
                    left += length
            else:
                new_tuples.append((length, op))
        return left, right, Cigar.from_tuples(new_tuples, lengths_from=self)

    def get_clipping(self):
        left_len, left_op = self._tuples[0]
        right_len, right_op = self._tuples[-1]
        # Do not expect to see hard clipping.
        assert left_op.consumes_read() and right_op.consumes_read()

        left = 0 if left_op.consumes_ref() else left_len
        right = 0 if right_op.consumes_ref() else right_len
        return left, right

    def __str__(self):
        return ''.join('%d%s' % t for t in self._tuples)

    def to_str(self, delim=' '):
        return delim.join('%d%s' % t for t in self._tuples)

    def __len__(self):
        return len(self._tuples)

    def to_full(self, ref_seq, read_seq):
        new_tuples = []
        pos1 = 0
        pos2 = 0

        for length, op in self._tuples:
            if op.consumes_both():
                curr_len = 0
                curr_op = Operation.SeqMatch
                for i in range(length):
                    new_op = Operation.SeqMatch if read_seq[pos1 + i] == ref_seq[pos2 + i] else Operation.SeqMismatch
                    if curr_op == new_op:
                        curr_len += 1
                    else:
                        if curr_len:
                            new_tuples.append((curr_len, curr_op))
                        curr_op = new_op
                        curr_len = 1
                new_tuples.append((curr_len, curr_op))
                pos1 += length
                pos2 += length

            elif op.consumes_read():
                new_tuples.append((length, op))
                pos1 += length
            elif op.consumes_ref():
                new_tuples.append((length, op))
                pos2 += length

        return Cigar.from_tuples(new_tuples, lengths_from=self)

    def to_extended_with_md(self, read_seq, md_tag):
        """
        Returns pair (extended CIGAR, reference_sequence).
        """
        new_tuples = []
        read_pos = 0
        ref_pos = 0
        ref_seq = ''
        # Use [A-Z^] instead of [0-9] because tag starts and ends with a number.
        # Parentheses needed to capture [A-Z^] as well as numbers.
        md_tag = re.split(r'([A-Z^]+)', md_tag)
        for i in range(0, len(md_tag), 2):
            md_tag[i] = int(md_tag[i])
        # Skip first, if it is == 0.
        md_ix = int(md_tag[0] == 0)
        md_shift = 0

        for length, op in self._tuples:
            if op.consumes_both():
                while length > 0:
                    # Sequence
                    if md_ix % 2:
                        subseq = md_tag[md_ix]
                        entry_len = len(subseq) - md_shift
                        pos_inc = min(length, entry_len)
                        new_op = Operation.SeqMismatch

                        ref_seq += subseq[md_shift : md_shift + pos_inc]
                        if entry_len > length:
                            md_shift += length
                        else:
                            md_ix += 1 + int(md_tag[md_ix + 1] == 0)
                            md_shift = 0

                    # Number
                    else:
                        new_op = Operation.SeqMatch
                        entry_len = md_tag[md_ix] - md_shift
                        pos_inc = min(length, entry_len)
                        ref_seq += read_seq[read_pos : read_pos + pos_inc]

                        if entry_len > length:
                            md_shift += length
                        else:
                            md_ix += 1
                            md_shift = 0

                    ref_pos += pos_inc
                    read_pos += pos_inc
                    if new_tuples and new_tuples[-1][1] == new_op:
                        new_tuples[-1] = (pos_inc + new_tuples[-1][0], new_op)
                    else:
                        new_tuples.append((pos_inc, new_op))
                    length -= pos_inc

            elif op.consumes_read():
                read_pos += length
                new_tuples.append((length, op))

            elif op.consumes_ref():
                assert md_tag[md_ix][0] == '^'
                ref_seq += md_tag[md_ix][1:]
                # Add 2 if next entry is 0. We should not get out of the list because it should end with a number.
                md_ix += 1 + int(md_tag[md_ix + 1] == 0)

                ref_pos += length
                new_tuples.append((length, op))

        assert md_ix == len(md_tag) and read_pos == self._read_len \
            and ref_pos == self._ref_len and len(ref_seq) == ref_pos
        return Cigar.from_tuples(new_tuples, lengths_from=self), ref_seq

    def calculate_stats(self):
        """
        Returns `AlnStats`. The Cigar should be full.
        """
        stats = AlnStats()
        for length, op in self._tuples:
            if op.consumes_both():
                if op == Operation.SeqMismatch:
                    stats.add_mismatches(length)
                else:
                    assert op == Operation.SeqMatch
            elif op.consumes_read() and op != Operation.Soft:
                stats.add_insertions(length)
            elif op.consumes_ref():
                stats.add_deletions(length)
        stats.update_from_cigar(self)
        return stats

    def calculate_stats_with_seqs(self, read_seq, ref_seq):
        """
        Returns `AlnStats`. The Cigar can contain M operations as well as X and =.
        """
        stats = AlnStats()
        pos1 = 0
        pos2 = 0
        for length, op in self._tuples:
            if op.consumes_both():
                stats.add_mismatches(sum(read_seq[pos1 + i] != ref_seq[pos2 + i] for i in range(length)))
                pos1 += length
                pos2 += length
            elif op.consumes_read():
                if op != Operation.Soft:
                    stats.add_insertions(length)
                pos1 += length
            elif op.consumes_ref():
                stats.add_deletions(length)
                pos2 += length
        stats.update_from_cigar(self)
        return stats

    def no_gaps(self):
        return all(op.consumes_both() for _, op in self._tuples[1:-1])

    def to_short(self):
        new_tuples = []
        curr_match = 0
        for length, op in self._tuples:
            if op.consumes_both():
                curr_match += length
            else:
                if curr_match:
                    new_tuples.append((curr_match, Operation.AlnMatch))
                    curr_match = 0
                new_tuples.append((length, op))
        if curr_match:
            new_tuples.append((curr_match, Operation.AlnMatch))
        return Cigar.from_tuples(new_tuples, lengths_from=self)

    def to_pysam_tuples(self):
        return [(op.value, len) for len, op in self._tuples]

    def aligned_seqs(self, read_seq, ref_seq):
        read_res = ''
        ref_res = ''

        pos1 = 0
        pos2 = 0
        for length, op in self._tuples:
            if op.consumes_both():
                subs1 = read_seq[pos1 : pos1 + length]
                subs2 = ref_seq[pos2 : pos2 + length]
                if op == Operation.SeqMismatch:
                    read_res += subs1.lower()
                    ref_res += subs2.lower()
                    if subs1 == subs2:
                        sys.stderr.write('Error: operation %dX at positions %d,%d corresponds to sequence match %s\n'
                            % (length, pos1, pos2, subs1))
                else:
                    read_res += subs1
                    ref_res += subs2
                    if op == Operation.SeqMatch and subs1 != subs2:
                        sys.stderr.write('Error: operation %d= at positions %d,%d corresponds to sequence '
                            'mismatch %s != %s\n' % (length, pos1, pos2, subs1, subs2))
                pos1 += length
                pos2 += length

            elif op.consumes_read():
                ref_res += '-' * length
                read_res += read_seq[pos1 : pos1 + length]
                pos1 += length
            elif op.consumes_ref():
                ref_res += ref_seq[pos2 : pos2 + length]
                read_res += '-' * length
                pos2 += length

        if pos1 != len(read_seq) or pos2 != len(ref_seq):
            sys.stderr.write('Error: CIGAR length does not match sequences length: '
                'read_len = %d, read_seq = %d, ref_len = %d, ref_seq = %d\n'
                % (pos1, len(read_seq), pos2, len(ref_seq)))
        return read_res, ref_res

    def aligned_pairs(self, ref_start=0, read_start=0):
        """
        Returns iterator over pairs of indices, aligned to each other.
        """
        assert not (ref_start and read_start)
        if ref_start:
            cigar_start, ref_pos, read_pos = self._index.find_by_ref(ref_start)
        elif read_start:
            cigar_start, ref_pos, read_pos = self._index.find_by_read(read_start)
        else:
            cigar_start = ref_pos = read_pos = 0

        for length, op in itertools.islice(self._tuples, cigar_start, len(self._tuples)):
            if op.consumes_both():
                for j in range(max(ref_start - ref_pos, 0), length):
                    yield (read_pos + j, ref_pos + j)
                read_pos += length
                ref_pos += length
            elif op.consumes_read():
                read_pos += length
            elif op.consumes_ref():
                ref_pos += length

    def read_region(self, ref_start, ref_end):
        cigar_start, ref_pos, read_pos = self._index.find_by_ref(ref_start)
        read_start = None

        for length, op in itertools.islice(self._tuples, cigar_start, len(self._tuples)):
            if ref_pos == ref_end:
                assert read_start is not None
                return read_start, read_pos

            cons_read = op.consumes_read()
            if op.consumes_ref():
                if ref_pos <= ref_start < ref_pos + length:
                    assert read_start is None
                    read_start = read_pos + (ref_start - ref_pos if cons_read else 1)
                if ref_pos <= ref_end < ref_pos + length:
                    assert read_start is not None
                    read_end = read_pos + (ref_end - ref_pos if cons_read else 1)
                    return read_start, read_end
                ref_pos += length

            if cons_read:
                read_pos += length

        assert ref_end == ref_pos
        assert read_start is not None
        return read_start, read_pos

    def subcigar(self, ref_start, ref_end):
        """
        Returns tuple (ref_start, ref_end, read_start, read_end, subcigar).
        Note, that if the first or last operation within the region of interest is not M,
        returned ref_start or ref_end may be different from input ref_start or ref_end.

        If the resulting alignment will be empty, returns None.
        """
        assert ref_start >= 0 and ref_end <= self._ref_len
        cigar_start, ref_pos, read_pos = self._index.find_by_ref(ref_start)
        read_start = None
        new_tuples = []

        for length, op in itertools.islice(self._tuples, cigar_start, len(self._tuples)):
            if ref_pos >= ref_end:
                assert ref_pos == ref_end
                break

            cons_ref = op.consumes_ref()
            cons_read = op.consumes_read()
            if cons_ref and cons_read:
                if ref_pos <= ref_start < ref_pos + length:
                    assert read_start is None
                    read_start = read_pos + ref_start - ref_pos
                intersection = min(ref_end, ref_pos + length) - max(ref_start, ref_pos)
                if intersection > 0:
                    new_tuples.append((intersection, op))
                if ref_pos < ref_end <= ref_pos + length:
                    read_pos += ref_end - ref_pos
                    break

                ref_pos += length
                read_pos += length

            elif cons_ref:
                if ref_pos <= ref_start < ref_pos + length:
                    # Alignment starts with Deletion.
                    ref_start = ref_pos + length
                    if ref_start >= ref_end:
                        return None
                elif ref_pos < ref_end <= ref_pos + length:
                    # Alignment ends with Deletion.
                    ref_end = ref_pos
                    break
                elif ref_pos >= ref_start:
                    new_tuples.append((length, op))
                ref_pos += length

            elif cons_read:
                if read_start is not None:
                    new_tuples.append((length, op))
                read_pos += length

        read_end = read_pos
        new_cigar = Cigar.from_tuples(new_tuples)
        assert new_cigar.ref_len == ref_end - ref_start
        assert new_cigar.read_len == read_end - read_start
        return ref_start, ref_end, read_start, read_end, new_cigar

    def revert(self, strand):
        new_tuples = []
        for length, op in self._tuples:
            if op.consumes_both():
                new_tuples.append((length, op))
            elif op == Operation.Insertion:
                new_tuples.append((length, Operation.Deletion))
            elif op == Operation.Deletion:
                new_tuples.append((length, Operation.Insertion))
            else:
                raise RuntimeError('Encountered operation %s while trying to revert cigar "%s"' % (op, self.to_str()))
        res = Cigar.from_tuples(new_tuples if strand else reversed(new_tuples), self)
        res._read_len = self._ref_len
        res._ref_len = self._read_len
        return res

    def __eq__(self, other):
        return self._tuples == other._tuples


class CigarIndex:
    def __init__(self, cigar, step_size=100):
        """
        Saves indices of CIGAR tuples to allow faster search within duplication.
        """
        cigar_indices = []
        ref_positions = []
        read_positions = []

        prev_ref_pos = ref_pos = 0
        prev_read_pos = read_pos = 0

        for cigar_ix, (length, op) in enumerate(cigar):
            if ref_pos > prev_ref_pos + step_size or read_pos > prev_read_pos + step_size:
                cigar_indices.append(cigar_ix)
                ref_positions.append(ref_pos)
                read_positions.append(read_pos)
                prev_ref_pos = ref_pos
                prev_read_pos = read_pos

            if op.consumes_ref():
                ref_pos += length
            if op.consumes_read():
                read_pos += length
        self._cigar_indices = tuple(cigar_indices)
        self._ref_positions = tuple(ref_positions)
        self._read_positions = tuple(read_positions)

    def find_by_ref(self, ref_pos):
        """
        Returns tuple (cigar_ix, ref_pos', read_pos') with the biggest cigar_ix such that ref_pos' <= ref_pos.
        """
        i = np.searchsorted(self._ref_positions, ref_pos, side='right') - 1
        if i >= 0:
            return (self._cigar_indices[i], self._ref_positions[i], self._read_positions[i])
        return (0, 0, 0)

    def find_by_read(self, read_pos):
        """
        Returns tuple (cigar_ix, ref_pos', read_pos') with the biggest cigar_ix such that read_pos' <= read_pos.
        """
        i = np.searchsorted(self._read_positions, read_pos, side='right') - 1
        if i >= 0:
            return (self._cigar_indices[i], self._ref_positions[i], self._read_positions[i])
        return (0, 0, 0)
