import re

from . import common


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


class Samples:
    def __init__(self, samples):
        self._samples = sorted(samples)
        self._sample_ids = { sample: id for id, sample in enumerate(self._samples) }

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
