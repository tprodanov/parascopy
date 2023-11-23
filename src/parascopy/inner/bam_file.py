import re
import io
import struct
from enum import Enum
import tempfile
import os
import numpy as np
from collections import defaultdict
from intervaltree import IntervalTree
import construct

from .duplication import Duplication
from .genome import Interval
from . import common
from .cigar import Cigar


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


def compare_contigs(bam_wrappers, genome):
    genome_set = set(genome.chrom_names)
    n = len(bam_wrappers)

    missing_n_contigs = np.zeros(n)
    missing_contigs = set()
    extra_n_contigs = np.zeros(n)
    extra_contigs = set()

    for i, bam_wrapper in enumerate(bam_wrappers):
        curr_contigs = set(bam_wrapper.contigs)
        curr_missing = genome_set - curr_contigs
        missing_n_contigs[i] = len(curr_missing)
        missing_contigs |= curr_missing

        curr_extra = curr_contigs - genome_set
        extra_n_contigs[i] = len(curr_extra)
        extra_contigs |= curr_extra

    PRINT_MAX = 5
    if missing_contigs or extra_contigs:
        common.log('WARN: Reference genome does not match BAM/CRAM files completely.')
    if missing_contigs:
        common.log('{} / {} alignment files have missing contigs.'.format(np.sum(missing_n_contigs > 0), n))
        common.log('On average, {:.3f} contigs are missing per file. In total, {} contigs are missing.'
            .format(np.mean(missing_n_contigs), len(missing_contigs)))

        j = np.where(missing_n_contigs > 0)[0][0]
        curr_missing = sorted(genome_set - set(bam_wrappers[j].contigs))
        if len(curr_missing) > PRINT_MAX:
            curr_missing[PRINT_MAX] = '...'
        common.log('For example, "{}" has {:.0f} missing contigs ({})'
            .format(bam_wrappers[j].filename, missing_n_contigs[j], ', '.join(curr_missing[:PRINT_MAX + 1])))
        common.log('')

    if extra_contigs:
        common.log('{} / {} alignment files have extra contigs.'.format(np.sum(extra_n_contigs > 0), n))
        common.log('On average, there are {:.3f} extra contigs per file. In total, there are {} extra contigs.'
            .format(np.mean(extra_n_contigs), len(extra_contigs)))

        j = np.where(extra_n_contigs > 0)[0][0]
        curr_extra = sorted(set(bam_wrappers[j].contigs) - genome_set)
        if len(curr_extra) > PRINT_MAX:
            curr_extra[PRINT_MAX] = '...'
        common.log('For example, "{}" has {:.0f} extra contigs ({})'
            .format(bam_wrappers[j].filename, extra_n_contigs[j], ', '.join(curr_extra[:PRINT_MAX + 1])))
        common.log('')


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

    def id_or_none(self, sample_name):
        return self._sample_ids.get(sample_name)

    def __iter__(self):
        return iter(self._samples)

    def as_list(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def __bool__(self):
        return bool(self._samples)

    def __eq__(self, oth):
        return self._samples == oth._samples


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


def _old_aln_is_true(cigar1, cigar2, aln_region2, strand2, baseq, unique_tree, min_tail):
    """
    Checks if the old alignment (*2) represents the true location of the read.
    """
    has_clipping1 = True if cigar1 is None else cigar1.has_true_clipping(baseq)
    has_clipping2 = cigar2.has_true_clipping(common.cond_reverse(baseq, strand=strand2))
    #  Same as (not has_clipping1 or has_clipping2).
    if has_clipping1 <= has_clipping2:
        return False
    return unique_tree.intersection_size(aln_region2) >= min_tail


class RecordCoord:
    IS_PAIRED    = 0b00000001
    IS_REVERSE   = 0b00000010
    # There is no new alignment.
    NEW_UNMAPPED = 0b00000100
    # There is no need to store old location, it is equal to the new.
    OLD_EQ_NEW   = 0b00001000
    LOC_INFO_SHIFT = 6

    #
    # Name          Bytes    Comment
    # hash            8      Hash of the read name, last bit is 1 if this is the first mate.
    # flag            1      Information about the read: strand, SE/PE, location info.
    # seq_len         V
    # new_chrom_id    V      New alignment location. Only present if !NEW_UNMAPPED.
    # new_start       V      0-based start
    # new_len         V
    # old_chrom_id    V      Original alignment location. Only present if !OLD_EQ_NEW.
    # old_start       V      0-based start
    # old_len         V
    #
    RecordStruct = construct.Struct(
        'hash' / construct.Int64un, # un = Unsigned native,
        'flag' / construct.Int8un,
        'seq_len' / construct.VarInt,
        'new_region' / construct.IfThenElse(construct.this.flag & NEW_UNMAPPED,
            construct.Pass, construct.VarInt[3]),
        'old_region' / construct.IfThenElse(construct.this.flag & OLD_EQ_NEW,
            construct.Pass, construct.VarInt[3]),
    )

    class LocationInfo(Enum):
        Unknown = 0
        # New location is certainly incorrect
        NewIncorrect = 1
        # Old location was certainly correct
        OldCorrect = 2

    def __init__(self):
        self.read_hash = None
        self.is_paired = None
        self.is_reverse = None
        self.location_info = RecordCoord.LocationInfo.Unknown
        self.seq_len = None
        # New and old alignment regions (can be the same).
        self.new_region = None
        self.old_region = None
        # It is possible that there are many realignments of the same read.
        # Here, we will refer all of them to each other.
        self.other_entries = None

    def add_entry(self, other):
        if self.other_entries:
            self.other_entries.append(other)
        else:
            self.other_entries = [other]

    def get_true_location(self):
        """
        Returns a true location, if it is certainly known. Otherwise, returns None.
        """
        return self.old_region if self.location_info == RecordCoord.LocationInfo.OldCorrect else None

    def get_forbidden_locations(self):
        """
        Returns all forbidden locations, if they are known.
        """
        res = []
        if self.location_info == RecordCoord.LocationInfo.NewIncorrect:
            res.append(self.new_region)
        for entry in self.other_entries or ():
            if entry.location_info == RecordCoord.LocationInfo.NewIncorrect:
                res.append(entry.new_region)
        return tuple(res)

    @classmethod
    def from_pooled_record(cls, record, unique_tree, genome, min_mapq=50, min_unique_tail=15):
        self = cls()
        self.read_hash = string_hash_fnv1(record.query_name, record.is_read1)
        # print('From pooled record: hash {} {}'.format(record.query_name, self.read_hash))
        self.seq_len = len(record.query_sequence)
        self.is_paired = record.is_paired
        self.is_reverse = record.is_reverse

        if record.is_unmapped:
            cigar1 = None
        else:
            chrom1 = genome.chrom_id(record.reference_name)
            self.new_region = Interval(chrom1, record.reference_start, record.reference_end)
            cigar1 = Cigar.from_pysam_tuples(record.cigartuples)
            mapq1 = record.mapping_quality

        # Parsing old alignment.
        oa_tag = record.get_tag('OA').split(',')
        chrom2 = genome.chrom_id(oa_tag[0])
        start2 = int(oa_tag[1]) - 1
        strand2 = oa_tag[2] == '+'
        cigar2 = Cigar(oa_tag[3])
        mapq2 = int(oa_tag[4])
        self.old_region = Interval(chrom2, start2, start2 + cigar2.ref_len)

        if self.new_region is not None and self.new_region == self.old_region \
                and not cigar1.has_true_clipping(record.query_qualities) \
                and unique_tree.intersection_size(self.new_region) >= min_unique_tail:
            self.location_info = RecordCoord.LocationInfo.OldCorrect

        elif mapq2 >= min_mapq and _old_aln_is_true(cigar1, cigar2, self.old_region, strand2,
                record.query_qualities, unique_tree, min_unique_tail):
            self.location_info = RecordCoord.LocationInfo.OldCorrect

        elif cigar1 is not None and cigar2.aligned_len > cigar1.aligned_len + min_unique_tail:
            # Redundant to add NewIncorrect if cigar1 is None.
            self.location_info = RecordCoord.LocationInfo.NewIncorrect
        return self

    def write_binary(self, out):
        data = construct.Container(hash=self.read_hash, seq_len=self.seq_len, new_region=None, old_region=None)
        flag = 0
        if self.is_paired:
            flag |= RecordCoord.IS_PAIRED
        if self.is_reverse:
            flag |= RecordCoord.IS_REVERSE

        if self.new_region is None:
            flag |= RecordCoord.NEW_UNMAPPED
        else:
            data.new_region = (self.new_region.chrom_id, self.new_region.start, len(self.new_region))

        if self.new_region is not None and self.new_region == self.old_region:
            flag |= RecordCoord.OLD_EQ_NEW
        else:
            data.old_region = (self.old_region.chrom_id, self.old_region.start, len(self.old_region))

        data.flag = flag | (self.location_info.value << RecordCoord.LOC_INFO_SHIFT)
        out.write(RecordCoord.RecordStruct.build(data))

    @staticmethod
    def parse_interval(seq):
        if seq is None:
            return None
        chrom, start, length = seq
        return Interval(chrom, start, start + length)

    @classmethod
    def from_binary(cls, stream, offset=0):
        self = cls()
        struct = RecordCoord.RecordStruct.parse_stream(stream)
        self.read_hash = np.uint64(struct.hash)
        flag = struct.flag
        self.is_paired = bool(flag & RecordCoord.IS_PAIRED)
        self.is_reverse = bool(flag & RecordCoord.IS_REVERSE)
        self.location_info = RecordCoord.LocationInfo(flag >> RecordCoord.LOC_INFO_SHIFT)

        self.seq_len = struct.seq_len
        self.new_region = RecordCoord.parse_interval(struct.new_region)
        self.old_region = RecordCoord.parse_interval(struct.old_region)
        assert self.new_region is not None or (flag & RecordCoord.NEW_UNMAPPED)
        if self.old_region is None:
            assert flag & RecordCoord.OLD_EQ_NEW
            self.old_region = self.new_region
        return self

    def to_str(self, genome=None):
        return '{:x}: length {}, new {}, old {}, {}'.format(
            self.read_hash, self.seq_len,
            '*' if self.new_region is None else (self.new_region.to_str(genome) if genome else repr(self.new_region)),
            '*' if self.old_region is None else (self.old_region.to_str(genome) if genome else repr(self.old_region)),
            self.location_info)


def write_record_coordinates(in_bam, samples, unique_tree, genome, out_filename, comment_dict):
    n_samples = len(samples)
    read_groups = {}
    for read_group, sample in get_read_groups(in_bam):
        read_groups[read_group] = samples.id_or_none(sample)

    with tempfile.TemporaryDirectory(prefix='parascopy') as wdir:
        n_entries = [0] * len(samples)
        tmp_filenames = [os.path.join(wdir, str(i)) for i in range(n_samples)]
        try:
            tmp_files = []
            for tmp_filename in tmp_filenames:
                tmp_files.append(open(tmp_filename, 'wb'))
            for record in in_bam:
                coord = RecordCoord.from_pooled_record(record, unique_tree, genome)
                sample_id = read_groups[record.get_tag('RG')]
                if sample_id is not None:
                    coord.write_binary(tmp_files[sample_id])
                    n_entries[sample_id] += 1
        finally:
            for f in tmp_files:
                f.close()

        with open(out_filename, 'wb') as out:
            # Header begins with the version of the coordinates file.
            index_str = ['#v2\n', genome.table_header()]
            for key, val in comment_dict.items():
                index_str.append('# {}={}\n'.format(key, val))
            index_str.append('#sample\tstart_offset\tend_offset\tn_entries\n')
            offset = 0
            for sample, tmp_filename, sample_entries in zip(samples, tmp_filenames, n_entries):
                with open(tmp_filename, 'rb') as tmp_file:
                    data = tmp_file.read()
                out.write(data)
                new_offset = offset + len(data)
                index_str.append('{}\t{}\t{}\t{}\n'.format(sample, offset, new_offset, sample_entries))
                offset = new_offset

    # Write full index at once so that we have no situation, where both files exist, but are incomplete.
    with open(CoordinatesIndex.index_name(out_filename), 'w') as out_index:
        out_index.writelines(index_str)


class CoordinatesIndex:
    @staticmethod
    def index_name(path):
        return path + '.ix'

    def __init__(self, path, samples, genome):
        self.version = 0
        self.index = [None] * len(samples)
        comment_dict = {}

        with open(CoordinatesIndex.index_name(path)) as f:
            line = next(f)
            if line.startswith('#v'):
                self.version = int(line[2:])
                line = next(f)
            if not genome.matches_header(line):
                raise ValueError('Input reference fasta does not match read coordinates file {}.\n'.format(path) +
                    'Consider deleting it or run parascopy with --rerun full.')

            for line in f:
                if line.startswith('#'):
                    if '=' in line:
                        key, val = line[2:].strip().split('=', 1)
                        comment_dict[key] = val
                    continue
                sample, start, end, n_entries = line.strip().split('\t')
                self.index[samples.id(sample)] = (int(start), int(end), int(n_entries))

        self.max_mate_dist = int(comment_dict['max_mate_dist'])
        self.path = path
        self.file = None

    def load(self, sample_id):
        start, end, n_entries = self.index[sample_id]
        length = end - start
        self.file.seek(start)
        stream = io.BytesIO(self.file.read(length))

        coordinates = {}
        for _ in range(n_entries):
            coord = RecordCoord.from_binary(stream)
            if coord.read_hash in coordinates:
                coordinates[coord.read_hash].add_entry(coord)
            else:
                coordinates[coord.read_hash] = coord
        try:
            if stream.__getstate__()[1] != length:
                common.log('WARN: Possibly, not all entries were read from the coordinates file')
        except AttributeError:
            pass
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
