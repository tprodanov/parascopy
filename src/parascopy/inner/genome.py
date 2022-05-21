import pysam
import re
import sys

from . import common


class ChromNames:
    def __init__(self, names, lengths):
        self._names = names
        self._ids = { name: i for i, name in enumerate(names) }
        if len(self._names) != len(self._ids):
            dupl_name = next(name for i, name in enumerate(self._names) if self._ids[name] != i)
            raise ValueError('Genome has duplicated chromosome names, for example chromosome "{}"'.format(dupl_name))
        self._lengths = lengths

    def table_header(self):
        res = '# Chromosomes: '
        res += ','.join(map('%s:%d'.__mod__, zip(self._names, self._lengths)))
        return res + '\n'

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
        raise ValueError('Cannot find line with prefix "{}" in the header'.format(prefix))

    def matches_header(self, header):
        chromosomes = header.split(':', 1)[1].strip().split(',')
        if len(chromosomes) != len(self._names):
            return False
        for i, entry in enumerate(chromosomes):
            name, length = entry.split(':')
            if name != self._names[i] or int(length) != self._lengths[i]:
                return False
        return True

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
        return '\n'.join(map('@SQ\tSN:%s\tLN:%d'.__mod__, self.names_lengths()))

    def chrom_interval(self, chrom_id, named=True):
        if named:
            return NamedInterval(chrom_id, 0, self._lengths[chrom_id], name=self._names[chrom_id])
        return Interval(chrom_id, 0, self._lengths[chrom_id])

    def all_chrom_intervals(self, named=True):
        for chrom_id in range(self.n_chromosomes):
            yield self.chrom_interval(chrom_id, named=named)

    def complement_intervals(self, intervals, include_full_chroms=True):
        curr_chrom_id = 0
        curr_end = 0
        res = []
        for interval in intervals:
            if interval.chrom_id > curr_chrom_id:
                curr_len = self._lengths[curr_chrom_id]
                if curr_len > curr_end and (curr_end > 0 or include_full_chroms):
                    res.append(Interval(curr_chrom_id, curr_end, curr_len))
                if include_full_chroms:
                    for chrom_id in range(curr_chrom_id + 1, interval.chrom_id):
                        res.append(Interval(chrom_id, 0, self._lengths[chrom_id]))
                curr_chrom_id = interval.chrom_id
                curr_end = 0

            assert interval.chrom_id == curr_chrom_id and interval.start >= curr_end
            if interval.start > curr_end:
                res.append(Interval(curr_chrom_id, curr_end, interval.start))
            curr_end = interval.end

        curr_len = self._lengths[curr_chrom_id]
        if curr_len > curr_end and (curr_end > 0 or include_full_chroms):
            res.append(Interval(curr_chrom_id, curr_end, curr_len))
        if include_full_chroms:
            for chrom_id in range(curr_chrom_id + 1, len(self._lengths)):
                res.append(Interval(chrom_id, 0, self._lengths[chrom_id]))
        return res


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

    def to_chrom_names(self):
        return ChromNames(list(self._names), list(self._lengths))

    @property
    def is_merged(self):
        return False

    def compare_with_other(self, genome2, filename2):
        set1 = set(self._names)
        set2 = set(genome2.chrom_names)

        PRINT_MAX = 5
        if set1 != set2:
            common.log('WARNING: Files "{}" and "{}" have different sets of contigs. This may raise errors later!'
                .format(self.filename, filename2))
            extra1 = sorted(set1 - set2)
            if len(extra1) > PRINT_MAX:
                extra1[PRINT_MAX] = '...'
            if extra1:
                common.log('File "{}" has {} extra contigs ({})'.format(
                    self.filename, len(extra1), ', '.join(extra1[:PRINT_MAX + 1])))

            extra2 = sorted(set2 - set1)
            if len(extra2) > PRINT_MAX:
                extra2[PRINT_MAX] = '...'
            if extra2:
                common.log('File "{}" has {} extra contigs ({})'.format(
                    filename2, len(extra2), ', '.join(extra2[:PRINT_MAX + 1])))
            common.log('')

        for name in set1 & set2:
            len1 = self.chrom_len(self.chrom_id(name))
            len2 = genome2.chrom_len(genome2.chrom_id(name))
            if len1 != len2:
                common.log('WARNING: Files "{}" and "{}" have different contig lengths:'
                    .format(self.filename, filename2))
                common.log('For example contig "{}" has lengths {:,} and {:,}'.format(name, len1, len2))
                common.log('')
            break


class GenomeMerge(ChromNames):
    def __init__(self, genome1, genome2):
        assert isinstance(genome1, Genome) and isinstance(genome2, Genome)
        self._genome1 = genome1
        self._genome2 = genome2
        self._fasta1 = genome1._fasta_file
        self._fasta2 = genome2._fasta_file
        super().__init__([], [])

        for genome in (genome1, genome2):
            for name, length in genome.names_lengths():
                new_chrom_id = self._ids.get(name)
                if new_chrom_id is None:
                    self._ids[name] = len(self._names)
                    self._names.append(name)
                    self._lengths.append(length)
                elif length != self._lengths[new_chrom_id]:
                    raise ValueError('Cannot merge genomes: chromosome "{}" has different lengths: {} and {}'
                        .format(name, length, self._lengths[new_chrom_id]))

        self._genome1_has_chrom = tuple(map(genome1.has_chrom, self._names))
        self._genome2_has_chrom = tuple(map(genome2.has_chrom, self._names))

    @staticmethod
    def from_filenames(*filenames):
        if len(filenames) < 1 or len(filenames) > 2:
            raise ValueError('Cannot create GenomeMerge from {} filenames'.format(len(filenames)))
        if len(filenames) == 1 or filenames[1] is None:
            return Genome(filenames[0])
        return GenomeMerge(Genome(filenames[0]), Genome(filenames[1]))

    @property
    def filename(self):
        return None

    def fetch_interval(self, interval):
        chrom_id = interval.chrom_id
        name = self._names[chrom_id]

        seq1 = self._fasta1.fetch(name, interval.start, interval.end).upper() \
            if self._genome1_has_chrom[chrom_id] else None
        seq2 = self._fasta2.fetch(name, interval.start, interval.end).upper() \
            if self._genome2_has_chrom[chrom_id] else None
        if seq1 and seq2 and seq1 != seq2:
            raise ValueError('Region "{}" is present in both input genomes, but contains different sequences'
                .format(interval.to_str_comma(self)))
        res = seq1 or seq2
        assert res is not None
        return res

    def to_chrom_names(self):
        return ChromNames(list(self._names), list(self._lengths))

    def close(self):
        self.genome1.close()
        self.genome2.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def is_merged(self):
        return True

    @property
    def genome1(self):
        return self._genome1

    @property
    def genome2(self):
        return self._genome2


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
            raise ValueError('Cannot construct an empty interval: start0 = {:,}, end = {:,}'
                .format(self._start, self._end))

    def copy(self):
        return Interval(self._chrom_id, self._start, self._end)

    @classmethod
    def parse(cls, string, genome):
        """
        Parse interval from string name:start-end, where start-end is 1-based closed interval.
        `genome` should be an object with method `chrom_id(chrom_name)`. This can be `genome.Genome`.
        """
        m = re.match(Interval._interval_pattern, string)
        if m is None:
            raise ValueError('Cannot parse "{}"'.format(string))
        chrom_id = genome.chrom_id(m.group(1))
        start = int(m.group(2).replace(',', '')) - 1
        end = int(m.group(3).replace(',', ''))
        return cls(chrom_id, start, end)

    @classmethod
    def parse_with_strand(cls, string, genome):
        interval, strand = string.rsplit(':', 1)
        assert strand == '+' or strand == '-'
        return cls.parse(interval, genome), strand == '+'

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
        return '{}:{}-{}'.format(genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_str_comma(self, genome):
        """
        Returns string "chr:start-end", where start-end is 1-based inclusive interval.
        """
        return '{}:{:,}-{:,}'.format(genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_str_path(self, genome):
        return '{}_{}-{}'.format(genome.chrom_name(self._chrom_id), self.start_1, self._end)

    def to_str0(self, genome):
        return '{}:{}..{}'.format(genome.chrom_name(self._chrom_id), self._start, self._end)

    def to_bed(self, genome):
        """
        Returns string "chr\tstart\tend", where start-end is 0-based semi-exclusive interval.
        """
        return '{}\t{}\t{}'.format(genome.chrom_name(self._chrom_id), self._start, self._end)

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
            raise ValueError('Interval {} is out of bounds: chromosome length is {}'
                .format(self.to_str(genome), chrom_len))
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

    @classmethod
    def get_disjoint_subregions(cls, intervals):
        """
        Returns a minimal set of subregions,
        s.t. each subregion is either contained or does not overlap any input intervals.
        Additionally, returns tuple of indices for each subregion (which input intervals cover the subregion).
        Input intervals must be sorted.
        """
        endpoints = []
        for i, interval in enumerate(intervals):
            endpoints.append((interval.chrom_id, interval.start, i))
            endpoints.append((interval.chrom_id, interval.end, ~i))
        endpoints.sort()

        subregions = []
        curr_ixs = set()
        for i, (chrom_id, pos, ix) in enumerate(endpoints):
            if curr_ixs:
                chrom_id2, prev_pos, _ = endpoints[i - 1]
                assert chrom_id == chrom_id2
                if prev_pos < pos:
                    subregions.append((cls(chrom_id, prev_pos, pos), tuple(curr_ixs)))

            if ix < 0:
                curr_ixs.remove(~ix)
            else:
                curr_ixs.add(ix)
        assert not curr_ixs
        return subregions


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
    def from_region(cls, region, genome, name):
        return cls(region.chrom_id, region.start, region.end, genome, name)

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
            return '{} ({})'.format(super().to_str_comma(genome), self._name)
        return self._name

    @classmethod
    def parse(cls, string, genome):
        if '::' in string:
            s_interval, name = string.split('::', 1)
        else:
            s_interval = string
            name = None

        if ':' in s_interval:
            interval = Interval.parse(s_interval, genome)
            interval.trim(genome)
        else:
            chrom_id = genome.chrom_id(s_interval)
            interval = Interval(chrom_id, 0, genome.chrom_len(chrom_id))
            if name is None:
                name = s_interval
        return cls.from_region(interval, genome, name)


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
