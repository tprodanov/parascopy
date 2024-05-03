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

    #####
    @classmethod
    def check_vmr(cls, bam_wrappers, low_vmrs):
        new_bam_wrappers = []
        new_samples = set()
        for bam_wrapper in bam_wrappers:
            if bam_wrapper.present_samples() & set(low_vmrs):
                new_bam_wrappers.append(bam_wrapper)
                new_samples.update(bam_wrapper.present_samples())
        return new_bam_wrappers, cls(new_samples)
    #####

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


def _aln_b_is_true_position(cigar_a, cigar_b, aln_region_b, strand_b, baseq, unique_tree, min_tail):
    has_clipping_a = True if cigar_a is None else cigar_a.has_true_clipping(baseq)
    has_clipping_b = cigar_b.has_true_clipping(common.cond_reverse(baseq, strand=strand_b))
    #  Same as (not has_clipping_a or has_clipping_b).
    if has_clipping_a <= has_clipping_b:
        return False
    return unique_tree.intersection_size(aln_region_b) >= min_tail


class RecordCoord:
    class LocationInfo(Enum):
        """
        Alignment region can have unknown status (unknown if correct or incorrect),
        certainly correct (it is known that self.aln_region is the true location),
        or certainly incorrect (self.aln_region is not the true location, however the true location is unknown).
        """
        Unknown = 0
        CertIncorrect = 1
        CertCorrect = 2

    IS_PAIRED   = 0b00000001
    IS_REVERSE  = 0b00000010
    LOC_INFO_SHIFT = 6
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
        self.location_info = RecordCoord.LocationInfo.Unknown
        self.is_paired = False
        self.is_reverse = False
        self.other_entries = None

    def add_entry(self, other):
        if self.other_entries:
            self.other_entries.append(other)
        else:
            self.other_entries = [other]

    def get_certainly_correct_location(self):
        return self.aln_region if self.location_info == RecordCoord.LocationInfo.CertCorrect else None

    def get_certainly_incorrect_locations(self):
        res = []
        if self.location_info == RecordCoord.LocationInfo.CertIncorrect:
            res.append(self.aln_region)
        if self.other_entries:
            for entry in self.other_entries:
                if entry.location_info == RecordCoord.LocationInfo.CertIncorrect:
                    res.append(entry.aln_region)
        return tuple(res)

    @classmethod
    def from_pooled_record(cls, record, unique_tree, genome, min_mapq=50, min_unique_tail=15):
        self = cls()
        self.read_hash = string_hash_fnv1(record.query_name, record.is_read1)
        # print('From pooled record: hash {} {}'.format(record.query_name, self.read_hash))
        self.seq_len = len(record.query_sequence)
        self.is_paired = record.is_paired
        self.is_reverse = record.is_reverse

        if not record.is_unmapped:
            chrom_a = genome.chrom_id(record.reference_name)
            self.aln_region = aln_region_a = Interval(chrom_a, record.reference_start, record.reference_end)
            cigar_a = Cigar.from_pysam_tuples(record.cigartuples)
            mapq_a = record.mapping_quality
            # print('    Alignment A:    {}   {}   {}'.format(aln_region_a.to_str(genome), cigar_a.to_str('.'), mapq_a))
            # print('    unique tail A = {}'.format(unique_tree.intersection_size(aln_region_a)))
        else:
            cigar_a = None

        if record.has_tag('OA'):
            oa_tag = record.get_tag('OA').split(',')
            chrom_b = genome.chrom_id(oa_tag[0])
            start_b = int(oa_tag[1]) - 1
            strand_b = oa_tag[2] == '+'
            cigar_b = Cigar(oa_tag[3])
            mapq_b = int(oa_tag[4])

            aln_region_b = Interval(chrom_b, start_b, start_b + cigar_b.ref_len)
            if self.aln_region is None:
                self.aln_region = aln_region_b
            # print('    Alignment B:    {}   {}   {}'.format(aln_region_b.to_str(genome), cigar_b.to_str('.'), mapq_b))
            # print('    unique tail B = {}'.format(unique_tree.intersection_size(aln_region_b)))

            # Do not check mapq_a because it is always 60.
            if mapq_b >= min_mapq and _aln_b_is_true_position(
                    cigar_a, cigar_b, aln_region_b, strand_b, record.query_qualities, unique_tree, min_unique_tail):
                self.aln_region = aln_region_b
                self.location_info = RecordCoord.LocationInfo.CertCorrect
            elif cigar_a is None or cigar_b.aligned_len > cigar_a.aligned_len + min_unique_tail:
                self.location_info = RecordCoord.LocationInfo.CertIncorrect

        elif mapq_a >= min_mapq and not cigar_a.has_true_clipping(record.query_qualities) and \
                unique_tree.intersection_size(aln_region_a) >= min_unique_tail:
            self.aln_region = aln_region_a
            self.location_info = RecordCoord.LocationInfo.CertCorrect
        # print('    -> location info = {}'.format(self.location_info))

        assert self.aln_region is not None
        return self

    def write_binary(self, out):
        flag = 0
        if self.is_paired:
            flag |= RecordCoord.IS_PAIRED
        if self.is_reverse:
            flag |= RecordCoord.IS_REVERSE
        flag |= self.location_info.value << RecordCoord.LOC_INFO_SHIFT

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
        self.is_paired = bool(flag & RecordCoord.IS_PAIRED)
        self.is_reverse = bool(flag & RecordCoord.IS_REVERSE)
        self.location_info = RecordCoord.LocationInfo(flag >> RecordCoord.LOC_INFO_SHIFT)
        return self

    def to_str(self, genome=None):
        return 'Hash {}: status {}, length {}, alignment {}, {}'.format(self.read_hash, self.status.name, self.seq_len,
            self.aln_region.to_str(genome) if genome else repr(self.aln_region), self.location_info)


def write_record_coordinates(in_bam, samples, unique_tree, genome, out_filename, comment_dict):
    n_samples = len(samples)
    read_groups = {}
    for read_group, sample in get_read_groups(in_bam):
        read_groups[read_group] = samples.id_or_none(sample)

    with tempfile.TemporaryDirectory(prefix='parascopy') as wdir:
        tmp_files = []
        try:
            for i in range(n_samples):
                tmp_files.append(open(os.path.join(wdir, str(i)), 'wb'))
            for record in in_bam:
                coord = RecordCoord.from_pooled_record(record, unique_tree, genome)
                sample_id = read_groups[record.get_tag('RG')]
                if sample_id is not None:
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
                coordinates[coord.read_hash].add_entry(coord)
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
