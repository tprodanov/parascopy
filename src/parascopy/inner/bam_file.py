import re
import struct
from enum import Enum
import tempfile
import os
import numpy as np
from collections import defaultdict
from intervaltree import IntervalTree

from .duplication import Duplication
from .genome import Interval
from . import common
from .cigar import Cigar


def checked_fetch(bam_file, chrom, start, end):
    try:
        return bam_file.fetch(chrom, start, end)
    except ValueError as e:
        common.log('ERROR: Cannot fetch {}:{}-{} from {}. Possibly chromosome {} is not in the alignment file.'
            .format(chrom, start + 1, end, bam_file.filename.decode(), chrom))
        return iter(())


def fetch(bam_file, region, genome):
    return checked_fetch(bam_file, region.chrom_name(genome), region.start, region.end)


def get_read_groups(bam_file):
    """
    Returns list of pairs (group_id, sample).
    """
    read_groups = []
    for line in str(bam_file.header).splitlines():
        if line.startswith('@RG'):
            has_rg = True

            id_m = re.search(r'ID:([ -~]+)', line)
            sample_m = re.search(r'SM:([ -~]+)', line)
            if id_m is None or sample_m is None:
                common.log('ERROR: Cannot find ID or SM field in the header line: "%s"' % line)
                exit(1)
            read_groups.append((id_m.group(1), sample_m.group(1)))
    return read_groups


def get_comment_items(bam_file):
    """
    Searches BAM header for the comments of type "@CO    key=value" and returns a corresponding dictionary.
    """
    res = {}
    for line in str(bam_file.header).splitlines():
        if line.startswith('@CO'):
            comment = line.split('\t', 1)[1]
            if '=' in comment:
                key, value = comment.split('=', 1)
                res[key] = value
    return res


class Samples:
    def __init__(self, samples):
        self._samples = sorted(samples)
        self._sample_ids = { sample: id for id, sample in enumerate(self._samples) }

    @classmethod
    def from_bam_wrappers(cls, bam_wrappers):
        samples = set()
        for bam_wrapper in bam_wrappers:
            samples.update(bam_wrapper.present_samples())
        return cls(samples)

    def __contains__(self, sample_name):
        return sample_name in self._sample_ids

    def __getitem__(self, sample_id):
        return self._samples[sample_id]

    def id(self, sample_name):
        return self._sample_ids[sample_name]

    def __iter__(self):
        return iter(self._samples)

    def as_list(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def __bool__(self):
        return bool(self._samples)


def string_hash_fnv1(s: str, flag: bool):
    with np.errstate(over='ignore'):
        fnv_prime = np.uint64(0x100000001b3)

        hash = np.uint64(0xcbf29ce484222325)
        for ch in s.encode():
            hash *= fnv_prime
            hash ^= np.uint8(ch)
        return (hash << np.uint8(1)) | np.uint8(flag)


class ReadStatus(Enum):
    # Read was not realigned (on the main copy of the duplication).
    SameLoc = 0
    # Read was realigned (on a second copy of the duplication).
    Realigned = 1
    # Read mate outside of the duplication.
    ReadMate = 2

    def has_orig_aln(self):
        return self == ReadStatus.Realigned


class AllDuplTree:
    def __init__(self, table, genome):
        self.table = table
        self.genome = genome
        self.dupl_tree = defaultdict(IntervalTree)
        self.query_tree = defaultdict(IntervalTree)

    def query_region_contained(self, region):
        for interval in self.query_tree[region.chrom_id].overlap(region.start, region.end):
            if interval.begin <= region.start and region.end <= interval.end:
                return True
        return False

    def _checked_add_dupl_region(self, region):
        tree = self.dupl_tree[region.chrom_id]
        if not tree.containsi(region.start, region.end, None):
            tree.addi(region.start, region.end, None)

    def add_query_region(self, region):
        chrom_id = region.chrom_id
        self.query_tree[chrom_id].addi(region.start, region.end, None)
        for tup in self.table.fetch(region.chrom_name(self.genome), region.start, region.end):
            dupl = Duplication.from_tuple(tup, self.genome)
            self._checked_add_dupl_region(dupl.region1)
            if not dupl.is_tangled_region:
                self._checked_add_dupl_region(dupl.region2)

    def unique_tail_size(self, region, padding_right=1000):
        start = region.start
        end = region.end
        if not self.query_region_contained(region):
            self.add_query_region(Interval(region.chrom_id, start, end + padding_right))

        min_tail = end - start
        for interval in self.dupl_tree[region.chrom_id].overlap(start, end):
            # Interval completely contains the region.
            min_tail = min(min_tail,
                max(0, interval.begin - start, end - interval.end))
            # No reason to write a break here, because dupl_tree.overlap is a set, not a lazy iterator.
        return min_tail


def _aln_b_is_true_position(cigar_a, cigar_b, aln_region_b, strand_b, baseq, dupl_tree, min_tail):
    has_clipping_a = True if cigar_a is None else cigar_a.has_true_clipping(baseq)
    has_clipping_b = cigar_b.has_true_clipping(common.cond_reverse(baseq, strand=strand_b))
    #  Same as (not has_clipping_a or has_clipping_b).
    if has_clipping_a <= has_clipping_b:
        return False
    return dupl_tree.unique_tail_size(aln_region_b) >= min_tail


class RecordCoord:
    STATUS_MASK = 0b00000011
    MAPPED_UNIQ = 0b10000000
    IS_PAIRED   = 0b01000000
    MAX_U16 = 0xffff
    byte_struct = None

    @staticmethod
    def init_byte_struct():
        """
        Binary format takes 19 bytes per read:
        Name          Bytes    Comment
        hash            8      Hash of the read name, last bit is 1 if this is the first mate.
        seq_len         2
        chrom_id        2
        aln_start       4      0-based position
        aln_len         2
        flag            1
        """
        if RecordCoord.byte_struct is None:
            RecordCoord.byte_struct = struct.Struct('=QHHIHB')

    def __init__(self):
        RecordCoord.init_byte_struct()

        self.read_hash = None
        self.seq_len = None
        self.aln_region = None
        self.status = None
        self.mapped_uniquely = False
        self.is_paired = False

    @classmethod
    def from_pooled_record(cls, record, dupl_tree, genome, min_mapq=50, min_unique_tail=50):
        self = cls()
        self.read_hash = string_hash_fnv1(record.query_name, record.is_read1)
        # print('Hash {} {}'.format(record.query_name, self.read_hash))
        self.seq_len = len(record.query_sequence)
        self.is_paired = record.is_paired

        if not record.is_unmapped:
            chrom_a = genome.chrom_id(record.reference_name)
            self.aln_region = aln_region_a = Interval(chrom_a, record.reference_start, record.reference_end)
            cigar_a = Cigar.from_pysam_tuples(record.cigartuples)
            mapq_a = record.mapping_quality
        else:
            cigar_a = None

        self.status = ReadStatus(record.get_tag('st'))
        if self.status.has_orig_aln():
            oa_tag = record.get_tag('OA').split(',')
            chrom_b = genome.chrom_id(oa_tag[0])
            start_b = int(oa_tag[1]) - 1
            strand_b = oa_tag[2] == '+'
            cigar_b = Cigar(oa_tag[3])
            mapq_b = int(oa_tag[4])

            aln_region_b = Interval(chrom_b, start_b, start_b + cigar_b.ref_len)
            if self.aln_region is None:
                self.aln_region = aln_region_b

            # Do not check mapq_a because it is always 60.
            if mapq_b >= min_mapq and _aln_b_is_true_position(
                    cigar_a, cigar_b, aln_region_b, strand_b, record.query_qualities, dupl_tree, min_unique_tail):
                self.aln_region = aln_region_b
                self.mapped_uniquely = True

        elif mapq_a >= min_mapq and not cigar_a.has_true_clipping(record.query_qualities) and \
                dupl_tree.unique_tail_size(aln_region_a) >= min_unique_tail:
            self.aln_region = aln_region_a
            self.mapped_uniquely = True

        assert self.aln_region is not None
        return self

    def write_binary(self, out):
        flag = self.status.value
        if self.is_paired:
            flag |= RecordCoord.IS_PAIRED
        if self.mapped_uniquely:
            flag |= RecordCoord.MAPPED_UNIQ

        out.write(RecordCoord.byte_struct.pack(
            self.read_hash,
            min(self.seq_len, RecordCoord.MAX_U16),
            self.aln_region.chrom_id,
            self.aln_region.start,
            min(len(self.aln_region), RecordCoord.MAX_U16),
            flag,
        ))

    @classmethod
    def from_binary(cls, buffer, offset=0):
        self = cls()
        (read_hash, seq_len, chrom_id, start, region_len, flag) = RecordCoord.byte_struct.unpack_from(buffer, offset)
        self.read_hash = np.uint64(read_hash)
        self.seq_len = seq_len
        self.aln_region = Interval(chrom_id, start, start + region_len)
        self.status = ReadStatus(flag & RecordCoord.STATUS_MASK)
        self.is_paired = bool(flag & RecordCoord.IS_PAIRED)
        self.mapped_uniquely = bool(flag & RecordCoord.MAPPED_UNIQ)
        return self

    def to_str(self, genome=None):
        return 'Hash {}: status {}, length {}, alignment {}, mapped uniquely: {}'.format(self.read_hash,
            self.status.name, self.seq_len, self.aln_region.to_str(genome) if genome else repr(self.aln_region),
            self.mapped_uniquely)


def write_record_coordinates(in_bam, samples, dupl_tree, genome, out_filename):
    n_samples = len(samples)
    read_groups = {}
    for read_group, sample in get_read_groups(in_bam):
        read_groups[read_group] = samples.id(sample)
    comment_dict = get_comment_items(in_bam)

    with tempfile.TemporaryDirectory(prefix='parascopy') as wdir:
        tmp_files = []
        try:
            for i in range(n_samples):
                tmp_files.append(open(os.path.join(wdir, str(i)), 'wb'))
            for record in in_bam:
                coord = RecordCoord.from_pooled_record(record, dupl_tree, genome)
                sample_id = read_groups[record.get_tag('RG')]
                coord.write_binary(tmp_files[sample_id])
        finally:
            for f in tmp_files:
                f.close()

        with open(out_filename, 'wb') as out:
            index_str = [genome.table_header()]
            for key, val in comment_dict.items():
                index_str.append('# {}={}\n'.format(key, val))
            offset = 0
            for i in range(n_samples):
                with open(os.path.join(wdir, str(i)), 'rb') as tmp_file:
                    data = tmp_file.read()
                out.write(data)
                new_offset = offset + len(data)
                index_str.append('{}\t{}\t{}\n'.format(samples[i], offset, new_offset))
                offset = new_offset
        with open(CoordinatesIndex.index_name(out_filename), 'w') as out_index:
            out_index.writelines(index_str)


class CoordinatesIndex:
    @staticmethod
    def index_name(path):
        return path + '.ix'

    def __init__(self, path, samples, genome):
        self.index = [None] * len(samples)
        comment_dict = {}

        with open(self.index_name(path)) as in_index:
            if not genome.matches_header(next(in_index)):
                raise ValueError('Input reference fasta does not match read coordinates file {}.\n'.format(path) +
                    'Consider deleting it or run parascopy with --rerun full.')
            for line in in_index:
                if line.startswith('# ') and '=' in line:
                    key, val = line[2:].strip().split('=', 1)
                    comment_dict[key] = val
                    continue
                sample, start, end = line.strip().split('\t')
                self.index[samples.id(sample)] = (int(start), int(end))

        self.max_mate_dist = int(comment_dict['max_mate_dist'])
        self.path = path
        self.file = None
        RecordCoord.init_byte_struct()

    def load(self, sample_id):
        start, end = self.index[sample_id]
        length = end - start
        self.file.seek(start)
        buffer = self.file.read(length)
        struct_size = RecordCoord.byte_struct.size

        coordinates = {}
        for offset in range(0, length, struct_size):
            coord = RecordCoord.from_binary(buffer, offset)
            if coord.read_hash in coordinates:
                curr_key = (coord.mapped_uniquely, len(coord.aln_region))
                prev = coordinates[coord.read_hash]
                prev_key = (prev.mapped_uniquely, len(prev.aln_region))
                if curr_key > prev_key:
                    coordinates[coord.read_hash] = coord
            else:
                coordinates[coord.read_hash] = coord
        return coordinates

    def open(self):
        self.file = open(self.path, 'rb')

    def close(self):
        self.file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()
