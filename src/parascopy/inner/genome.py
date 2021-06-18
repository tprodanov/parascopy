import pysam
import re
import sys

from . import common


class OnlyChromNames:
    def __init__(self, names):
        self._names = names

    def chrom_name(self, chrom_id):
        return self._names[chrom_id]


class ChromNames:
    def __init__(self, names, lengths):
        self._names = names
        self._ids = { name: i for i, name in enumerate(names) }
        self._lengths = lengths

    @classmethod
    def from_table(cls, table):
        prefix = '# Chromosomes: '
        for line in table.header:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                names = []
                lengths = []
                for entry in line.split(','):
                    name, length = entry.split(':')
                    names.append(name)
                    lengths.append(int(length))
                return cls(names, lengths)
        raise ValueError('Cannot find line with prefix "%s" in the header' % prefix)

    def chrom_id(self, chrom_name):
        return self._ids[chrom_name]

    def has_chrom(self, chrom_name):
        return chrom_name in self._ids

    def chrom_name(self, chrom_id):
        return self._names[chrom_id]

    def chrom_len(self, chrom_id):
        return self._lengths[chrom_id]

    @property
    def has_lengths(self):
        return bool(self._lengths)

    @property
    def chrom_names(self):
        return self._names

    @property
    def chrom_lengths(self):
        return self._lengths

    @property
    def names_lengths(self):
        return zip(self._names, self._lengths)

    @property
    def n_chromosomes(self):
        return len(self._names)

    @classmethod
    def from_pysam(cls, obj):
        """
        From pysam object with fields `references` and `lengths` (for example `FastaFile` or `AlignmentFile`).
        """
        return cls(obj.references, obj.lengths)

    def generate_bam_header(self):
        return '\n'.join(map('@SQ\tSN:%s\tLN:%d'.__mod__, zip(self._names, self._lengths)))

    def chrom_interval(self, chrom_id):
        return Interval(chrom_id, 0, self._lengths[chrom_id])


class Genome(ChromNames):
    def __init__(self, filename):
        self._filename = filename
        self._fasta_file = pysam.FastaFile(filename)
        super().__init__(self._fasta_file.references, self._fasta_file.lengths)

    @property
    def filename(self):
        return self._filename

    def fetch_interval(self, interval):
        return self._fasta_file.fetch(self.chrom_name(interval.chrom_id), interval.start, interval.end).upper()

    def close(self):
        self._fasta_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def only_chrom_names(self):
        return OnlyChromNames(self._fasta_file.references)


class Interval:
    _interval_pattern = re.compile(r'^([^:]+):([0-9,]+)-([0-9,]+)$')

    def __init__(self, chrom_id, start, end):
        """
        Construct genomic interval.
        Parameters:
            - chrom_id: int, 0-based chromosome number,
            - start: int, 0-based start (inclusive),
            - end: int, 0-based end (exclusive).
        """
        self._chrom_id = chrom_id
        self._start = start
        self._end = end
        if self._end <= self._start:
            raise ValueError('Cannot construct an empty interval: start0 = %d, end = %d' % (self._start, self._end))

    @classmethod
    def parse(cls, string, genome):
        """
        Parse interval from string name:start-end, where start-end is 1-based closed interval.
        `genome` should be an object with method `chrom_id(chrom_name)`. This can be `genome.Genome`.
        """

        m = re.match(Interval._interval_pattern, string)
        if m is None:
            raise ValueError('Cannot parse "%s"' % string)
        chrom_id = genome.chrom_id(m.group(1))
        start = int(m.group(2).replace(',', '')) - 1
        end = int(m.group(3).replace(',', ''))
        return cls(chrom_id, start, end)

    @property
    def chrom_id(self):
        return self._chrom_id

    def chrom_name(self, genome):
        return genome.chrom_name(self._chrom_id)

    @property
    def start(self):
        return self._start

    @property
    def start_1(self):
        """
        Returns 1-based start.
        """
        return self._start + 1

    @property
    def end(self):
        return self._end

    def to_str(self, genome):
        """
        Returns string "chr:start-end", where start-end is 1-based inclusive interval.
        """
        return '%s:%d-%d' % (genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_str_comma(self, genome):
        """
        Returns string "chr:start-end", where start-end is 1-based inclusive interval.
        """
        return '{}:{:,}-{:,}'.format(genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_str_path(self, genome):
        return '%s_%d-%d' % (genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_range_str(self, genome):
        """
        Returns string "chr:start..end", start-end is 0-based (!) semi-inclusive interval.
        """
        return '%s:%d..%d' % (genome.chrom_name(self._chrom_id), self._start, self._end)

    def to_bed(self, genome):
        """
        Returns string "chr\tstart\tend", where start-end is 0-based semi-exclusive interval.
        """
        return '%s\t%d\t%d' % (genome.chrom_name(self._chrom_id), self._start, self._end)

    def get_sequence(self, genome, strand=True):
        """
        Returns genomic sequence.
        `strand` can be True (forward strand) or False (reverse strand).
        """
        seq = genome.fetch_interval(self)
        if strand:
            return seq
        else:
            return common.rev_comp(seq)

    def __len__(self):
        return self._end - self._start

    def intersects(self, other):
        return self._chrom_id == other._chrom_id and self._start < other._end and self._end > other._start

    def intersection(self, other):
        assert self.intersects(other)
        return Interval(self._chrom_id, max(self._start, other._start), min(self._end, other._end))

    def intersection_size(self, other):
        if self._chrom_id != other._chrom_id:
            return 0
        return max(0, min(self._end, other._end) - max(self._start, other._start))

    def forward_order(self, other, strict_order):
        if self._chrom_id != other._chrom_id:
            return False
        if strict_order:
            return self._end <= other._start
        return self._start <= other._end

    def with_max_start(self, max_start):
        """
        Returns new Interval with start = max(self.start, max_start).
        """
        if self._start >= max_start:
            return self
        return Interval(self._chrom_id, max_start, self._end)

    def with_min_end(self, min_end):
        """
        Returns new Interval with end = min(self.end, min_end).
        """
        if self._end <= min_end:
            return self
        return Interval(self._chrom_id, self._start, min_end)

    def out_of_bounds(self, genome):
        return self._end > genome.chrom_len(self._chrom_id)

    def trim(self, genome):
        """
        Trim end if it is bigger than the chromosome length.
        """
        chrom_len = genome.chrom_len(self._chrom_id)
        if self._start >= chrom_len:
            raise ValueError('Interval %s is out of bounds: chromosome length is %d'
                % (self.to_str(genome), chrom_len))
        self._end = min(chrom_len, self._end)

    def start_distance(self, other):
        """
        Returns distance between starts. Returns -1 if intervals lie on different chromosomes.
        """
        if self._chrom_id != other._chrom_id:
            return -1
        return abs(self._start - other._start)

    def distance(self, other):
        """
        Returns distance between closest points of two duplications.
        Returns sys.maxsize if intervals lie on different chromosomes.
        """
        if self._chrom_id != other._chrom_id:
            return sys.maxsize
        return max(0, self._start - other._end + 1, other._start - self._end + 1)

    def combine(self, other):
        """
        Returns combined interval (min of starts, max of ends). Should have the same chromosomes.
        """
        assert self._chrom_id == other._chrom_id
        return Interval(self._chrom_id, min(self._start, other._start), max(self._end, other._end))

    def add_padding(self, padding):
        return Interval(self._chrom_id, max(self._start - padding, 0), self._end + padding)

    def to_tuple(self):
        return (self._chrom_id, self._start, self._end)

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __eq__(self, other):
        return self._chrom_id == other._chrom_id and self._start == other._start and self._end == other._end

    def __hash__(self):
        return hash(self.to_tuple())

    def contains(self, other):
        return self._chrom_id == other._chrom_id and self._start <= other._start and self._end >= other._end

    def contains_point(self, chrom_id, pos):
        return self._chrom_id == chrom_id and self._start <= pos < self._end

    def __repr__(self):
        return 'Region(chrom_id={}, start={:,}, end={:,})'.format(self._chrom_id, self._start, self._end)

    @staticmethod
    def combine_overlapping(intervals, max_dist=0):
        res = []
        for interval in sorted(intervals):
            if not res or res[-1].distance(interval) > max_dist:
                res.append(interval)
            else:
                res[-1] = res[-1].combine(interval)
        return res


class NamedInterval(Interval):
    def __init__(self, chrom_id, start, end, genome, name=None):
        super().__init__(chrom_id, start, end)
        self._name_provided = name is not None
        if name is None:
            self._name = self.to_str_comma(genome)
            self._os_name = self.to_str(genome)
        else:
            self._name = name
            self._os_name = re.sub(r'[^0-9a-zA-Z_:-]', '_', name)

    @classmethod
    def from_region(cls, region, name):
        assert name is not None
        return cls(region.chrom_id, region.start, region.end, None, name)

    @property
    def name_provided(self):
        return self._name_provided

    @property
    def name(self):
        return self._name

    @property
    def os_name(self):
        """
        Returns name, which can be used as file names.
        """
        return self._os_name

    def full_name(self, genome):
        if self._name_provided:
            return '%s (%s)' % (super().to_str_comma(genome), self._name)
        return self._name

    @classmethod
    def parse(cls, string, genome):
        interval = Interval.parse(string, genome)
        return cls(interval.chrom_id, interval.start, interval.end, genome)


_nucleotides = 'ACGT'
_nucleotide_index = { nt: i for i, nt in enumerate(_nucleotides) }


def kmers(seq, k):
    """
    Returns iterator over pairs `(index: int, kmer: int)`.
    `kmer` can be transformed into sequence using `kmer_sequence(...)`.
    """
    k1 = k - 1
    cut_old = (1 << 2 * k1) - 1

    kmer = 0
    til_yield = k1
    for i, nt in enumerate(seq):
        nt_index = _nucleotide_index.get(nt)
        if nt_index is None:
            kmer = 0
            til_yield = k1
            continue

        kmer = ((kmer & cut_old) << 2) + nt_index
        if til_yield:
            til_yield -= 1
        else:
            yield (i - k1, kmer)


def kmer_sequence(int_kmer, kmer_len):
    res = ''
    for i in range(kmer_len):
        res += _nucleotides[value % 4]
        int_kmer = int_kmer >> 2
    return ''.join(reversed(res))
