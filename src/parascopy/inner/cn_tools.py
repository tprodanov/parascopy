import collections
import numpy as np
import itertools
import os
from scipy.special import logsumexp

from . import common
from . import itree
from .genome import Interval


def regions2_str(regions2, genome, use_comma=False, sep=','):
    if regions2 is None or len(regions2) == 0:
        return '*'
    if genome is None:
        return sep.join('{!r}:{}'.format(region2, '+' if strand2 else '-') for region2, strand2 in regions2)
    fn = Interval.to_str_comma if use_comma else Interval.to_str
    return sep.join('{}:{}'.format(fn(region2, genome), '+' if strand2 else '-') for region2, strand2 in regions2)


class DuplRegion:
    def __init__(self, ix, region1, regions2):
        self._ix = ix
        self._region1 = region1
        # List of pairs (region, strand: bool)
        self._regions2 = regions2
        self._copy_num = 2 + 2 * len(self.regions2) if self.regions2 is not None else 2

    @property
    def ix(self):
        return self._ix

    @property
    def region1(self):
        return self._region1

    @property
    def regions2(self):
        return self._regions2

    def regions2_str(self, *args, **kwargs):
        return regions2_str(self._regions2, *args, **kwargs)

    @property
    def cn(self):
        return self._copy_num

    def __str__(self):
        prefix = self.__class__.__name__
        if self._ix is not None:
            prefix += '[{}]'.format(self._ix)
        return '{}({!r}  CN = {})'.format(prefix, self._region1, self._copy_num)

    def continue_if_possible(self, other, max_dist, strict_order=True):
        """
        Creates a region with start at self.region1.start and end at other.region1.end.
        If possible, returns a pair `region`, `regions2`. Otherwise, returns None.
        """
        if self._copy_num != other._copy_num or self._region1.distance(other._region1) > max_dist:
            return None

        assert self._region1.forward_order(other._region1, strict_order)
        new_regions2 = []
        b_used = np.zeros(len(other._regions2), dtype=np.bool)
        for i, (a_reg, a_strand) in enumerate(self._regions2):
            for j, (b_reg, b_strand) in enumerate(other._regions2):
                if b_used[j] or a_strand != b_strand or a_reg.chrom_id != b_reg.chrom_id:
                    continue
                if not a_strand:
                    a_reg, b_reg = b_reg, a_reg

                if a_reg.forward_order(b_reg, strict_order) and b_reg.start - a_reg.end < max_dist:
                    b_used[j] = True
                    new_region = Interval(a_reg.chrom_id, a_reg.start, b_reg.end)
                    new_regions2.append((new_region, a_strand))
                    break
            else:
                # Cannot continue one of the second regions.
                return None

        new_region1 = Interval(self.region1.chrom_id, self.region1.start, other.region1.end)
        return new_region1, new_regions2

    def __lt__(self, other):
        return self._region1.__lt__(other._region1)


class Window(DuplRegion):
    def __init__(self, ix, region1, regions2, middles2, gc_content, const_region_ix):
        super().__init__(ix, region1, regions2)
        self._gc_content = gc_content
        self._const_region_ix = const_region_ix
        self.in_hmm = True
        self.multiplier = None
        self._middles2 = middles2

    @property
    def gc_content(self):
        return self._gc_content

    @property
    def const_region_ix(self):
        return self._const_region_ix

    def half_subregion1(self, left_side: bool):
        middle1 = (self._region1.start + self._region1.end) // 2
        if left_side:
            return self._region1.with_min_end(middle1)
        return self._region1.with_max_start(middle1)

    def half_subregion2(self, subregion_ix, left_side):
        subregion2, strand2 = self._regions2[subregion_ix]
        middle2 = self._middles2[subregion_ix]

        if strand2 == left_side:
            half_subregion2 = subregion2.with_min_end(middle2)
        else:
            half_subregion2 = subregion2.with_max_start(middle2)
        return half_subregion2, strand2

    def half_dupl_region(self, left_side: bool):
        region1 = self.half_subregion1(left_side)
        regions2 = []
        for i in range(len(self.regions2)):
            regions2.append(self.half_subregion2(i, left_side))
        return DuplRegion(None, region1, regions2)


class ConstRegion(DuplRegion):
    def __init__(self, ix, region1, regions2, skip, dupl_ixs):
        """
        Duplication indices should have the same order as regions2.
        """
        super().__init__(ix, region1, regions2)
        self.skip = skip
        self._dupl_ixs = dupl_ixs
        self._group_name = None

    @property
    def dupl_ixs(self):
        return self._dupl_ixs

    def set_group_name(self, group_name):
        self._group_name = group_name

    @property
    def group_name(self):
        return self._group_name


class RegionGroup(DuplRegion):
    def __init__(self, name, const_region, psv_ixs):
        super().__init__(name, const_region.region1, const_region.regions2)
        self._region_ixs = [const_region.ix]
        const_region.set_group_name(name)
        self._psv_ixs = psv_ixs
        self._window_ixs = []

    @property
    def name(self):
        return self._ix

    def continue_group(self, const_region, psv_ixs, max_dist):
        assert self.cn == const_region.cn
        continuation = self.continue_if_possible(const_region, max_dist)
        if continuation is None:
            return False

        const_region.set_group_name(self.name)
        self._region1 = continuation[0]
        self._regions2 = continuation[1]
        self._region_ixs.append(const_region.ix)
        self._psv_ixs.extend(psv_ixs)
        return True

    @property
    def region_ixs(self):
        return self._region_ixs

    @property
    def psv_ixs(self):
        return self._psv_ixs

    def append_window(self, window_ix):
        self._window_ixs.append(window_ix)

    @property
    def window_ixs(self):
        return self._window_ixs


class CopyNumPrediction(DuplRegion):
    def __init__(self, region1, regions2, pred_cn, pred_cn_str, pred_cn_qual):
        super().__init__(None, region1, regions2)
        self._pred_cn = pred_cn
        self._pred_cn_str = pred_cn_str
        self._qual = pred_cn_qual
        self._probable_cns = ((pred_cn, pred_cn_str, 0),)
        self._info = {}

    @classmethod
    def create(cls, region1, regions2, pred_cns, cn_probs, model, update_agcn_qual):
        pred_cn = pred_cns[0]
        pred_cn_qual = common.phred_qual(cn_probs, best_ix=0)
        self = cls(region1, regions2, pred_cn, model.format_cn(pred_cn), pred_cn_qual)
        if pred_cn_qual < update_agcn_qual:
            thresh_prob = (-0.1 * update_agcn_qual - 1) * common.LOG10
            ixs = np.where(cn_probs >= thresh_prob)[0]
            if len(ixs) < 2:
                ixs = np.argsort(-cn_probs)[:2]
            if len(ixs) > 1:
                self._probable_cns = tuple((pred_cns[i], model.format_cn(pred_cns[i]), cn_probs[i]) for i in ixs)
                self.info['agCN_probs'] = \
                    ','.join('{}:{:.4g}'.format(cn_str, np.exp(prob)) for _, cn_str, prob in self._probable_cns)
        return self

    def update_pred_cn(self, cn, cn_str, cn_qual):
        if self._pred_cn == cn and self._qual >= cn_qual:
            return
        self._info['init_agCN'] = '{}:{:.0f}'.format(self._pred_cn_str, self._qual)
        self._pred_cn = cn
        self._pred_cn_str = cn_str
        self._qual = cn_qual

    @property
    def pred_cn(self):
        return self._pred_cn

    @property
    def pred_cn_str(self):
        return self._pred_cn_str

    @property
    def qual(self):
        return self._qual

    @property
    def probable_cns(self):
        return self._probable_cns

    @property
    def region_ix(self):
        return self._region_ix

    @property
    def cn_is_known(self):
        return self._pred_cn is not None and self._pred_cn_str[0].isdigit()

    @property
    def info(self):
        return self._info


def find_const_regions(duplications, interval, skip_regions, genome, min_size, max_ref_cn):
    """
    Parameters:
    * duplications - list of duplications on the same chromosome. Duplications should have a defined CIGAR.
        Will not work for iterator of duplications.
    Returns list of `ConstRegion`s.
    """
    assert isinstance(duplications, list)
    result = []
    reg_chrom_id = None
    # List of (pos: int, is_start: bool, index: int). If index is -1, the region is skipped.
    endpoints = []
    for i, dupl in enumerate(duplications):
        if reg_chrom_id is None:
            reg_chrom_id = dupl.region1.chrom_id
        else:
            assert reg_chrom_id == dupl.region1.chrom_id

        endpoints.append((dupl.region1.start, True, i))
        endpoints.append((dupl.region1.end, False, i))
    for region in skip_regions:
        endpoints.append((region.start, True, -1))
        endpoints.append((region.end, False, -1))
    endpoints.sort()
    if not endpoints or endpoints[-1][0] < interval.end:
        endpoints.append((interval.end, True, -1))

    prev_pos = min(interval.start, endpoints[0][0])
    curr_dupl = set()
    curr_skip_amount = 0

    for pos, is_start, index in endpoints:
        if pos > prev_pos:
            region1 = Interval(interval.chrom_id, prev_pos, pos)
            if curr_skip_amount == 0:
                regions2 = []
                dupl_ixs = []
                for i in curr_dupl:
                    reg2 = duplications[i].subregion2(prev_pos, pos)
                    if reg2:
                        regions2.append((reg2, duplications[i].strand))
                        dupl_ixs.append(i)
                sort_ixs = sorted(range(len(regions2)), key=lambda i: regions2[i])
                regions2 = [regions2[i] for i in sort_ixs]
                dupl_ixs = [dupl_ixs[i] for i in sort_ixs]
                const_reg = ConstRegion(len(result), region1, regions2,
                    skip=len(region1) < min_size and len(regions2) < max_ref_cn // 2 and interval.contains(region1),
                    dupl_ixs=dupl_ixs)
            else:
                const_reg = ConstRegion(len(result), region1, None, skip=True, dupl_ixs=None)
                const_reg._copy_num = 2 * (len(curr_dupl) + 1 + curr_skip_amount)
            result.append(const_reg)
            prev_pos = pos

        if is_start:
            if index >= 0:
                curr_dupl.add(index)
            else:
                curr_skip_amount += 1
        else:
            if index >= 0:
                curr_dupl.remove(index)
            else:
                curr_skip_amount -= 1

    assert not curr_dupl
    # Because of the additional fake endpoint at the end.
    assert curr_skip_amount <= 1
    return result


def _extend_windows(windows, const_region, interval_start, interval_seq, duplications, window_size,
        padding=50, window_len_ratio=0.9):
    if const_region.skip or len(const_region.region1) < window_size - 2 * padding:
        return
    region_dupls = [duplications[i] for i in const_region.dupl_ixs]
    region1 = const_region.region1

    remainder = (len(region1) - 2 * padding) % window_size
    left = remainder // 2
    right = remainder - left
    prev_ends2 = [None] * len(region_dupls)

    for window_start in range(region1.start + left + padding, region1.end - right - padding, window_size):
        window_end = window_start + window_size
        assert window_end <= region1.end
        subregion1 = Interval(region1.chrom_id, window_start, window_end)
        middle1 = (window_start + window_end) // 2
        subregions2 = []
        middles2 = []

        big_difference = False
        # One of the duplications has a large deletion in this position.
        skip_this_window = False
        for j, dupl in enumerate(region_dupls):
            subregion2 = dupl.aligned_region(subregion1)
            if subregion2 is None:
                skip_this_window = True
                # continue because we need to update all prev_ends.
                continue
            assert const_region.regions2[j][0].contains(subregion2)
            if dupl.strand:
                len2 = len(subregion2) if prev_ends2[j] is None else subregion2.end - prev_ends2[j]
                prev_ends2[j] = subregion2.end
            else:
                len2 = len(subregion2) if prev_ends2[j] is None else prev_ends2[j] - subregion2.start
                prev_ends2[j] = subregion2.start

            if len2 < window_len_ratio * window_size or window_size < window_len_ratio * len2:
                big_difference = True
            subregions2.append((subregion2, dupl.strand))
            # +1 if negative strand. This is because we need middle of the window, not exact match to middle1.
            middles2.append(dupl.aligned_pos(middle1) + int(not dupl.strand))
        if skip_this_window:
            continue

        gc_content = int(common.gc_content(interval_seq[window_start - interval_start : window_end - interval_start]))
        window = Window(len(windows), subregion1, subregions2, middles2, gc_content, const_region.ix)
        if big_difference:
            window.in_hmm = False
        windows.append(window)


def _psv_matches_const_region(const_region, psv, genome):
    psv_allele_lengths = list(map(len, psv.alleles))
    if psv.start < const_region.region1.start or const_region.region1.end < psv.start + psv_allele_lengths[0]:
        return False

    pos2 = psv.info['pos2']
    if len(pos2) != len(const_region.regions2):
        common.log('WARN: PSV {}:{:,} does not match with region {}'.format(psv.chrom, psv.pos, const_region.ix))
        return False

    for (region2, reg_strand), pos2_entry in zip(const_region.regions2, pos2):
        pos2_entry = pos2_entry.split(':')
        curr_chrom_id = genome.chrom_id(pos2_entry[0])
        curr_start = int(pos2_entry[1]) - 1
        curr_strand = pos2_entry[2] == '+'
        allele = 1 if len(pos2_entry) < 4 else int(pos2_entry[3])
        curr_end = curr_start + psv_allele_lengths[allele]

        if curr_chrom_id != region2.chrom_id or reg_strand != curr_strand:
            common.log('WARN: PSV {}:{:,} does not match with region {}'.format(psv.chrom, psv.pos, const_region.ix))
            return False
        if curr_start < region2.start or region2.end < curr_end:
            if curr_start < region2.start - 100 or curr_end > region2.end + 100:
                common.log('WARN: PSV {}:{:,} does not match with region {}'.format(psv.chrom, psv.pos, const_region.ix))
            return False
    return True


def _create_region_groups(const_regions, psvs, genome, max_dist=1000):
    """
    Groups closeby const regions if they have the same CN even if there is a region with a different CN between them.
    """
    psv_ix = 0
    n_psvs = len(psvs)

    # Key: copy_num, value: list of PloidyRegionsGroups.
    groups = collections.defaultdict(list)
    for const_region in const_regions:
        if const_region.skip:
            continue

        region1_end = const_region.region1.end
        curr_psvs = []
        while psv_ix < n_psvs and psvs[psv_ix].start < region1_end:
            if _psv_matches_const_region(const_region, psvs[psv_ix], genome):
                curr_psvs.append(psv_ix)
            psv_ix += 1

        copy_num = const_region.cn // 2
        # If copy_num was already encountered, we try to continue last group with this copy_num.
        if copy_num not in groups or not groups[copy_num][-1].continue_group(const_region, curr_psvs, max_dist):
            group_name ='{:02d}-{:02d}'.format(copy_num, len(groups[copy_num]) + 1)
            groups[copy_num].append(RegionGroup(group_name, const_region, curr_psvs))
    return sorted(itertools.chain(*groups.values()))


class DuplHierarchy:
    """
    Hierarchy of duplicated regions:
        psvs < windows < const_regions < region_groups.
    """
    def __init__(self, interval, psvs, const_regions, genome, duplications, window_size, max_ref_cn):
        self.interval = interval
        self._psvs = psvs
        self._psv_searcher = itree.NonOverlTree(self._psvs, itree.start, itree.variant_end)
        self._const_regions = const_regions

        interval_seq = interval.get_sequence(genome)
        interval_start = interval.start

        self._windows = []
        for const_region in const_regions:
            if const_region.cn > max_ref_cn:
                continue
            _extend_windows(self._windows, const_region, interval_start, interval_seq, duplications, window_size)

        self._region_groups = _create_region_groups(const_regions, psvs, genome)
        self._group_by_name = { group.name: i for i, group in enumerate(self._region_groups) }
        self._init()

    def _init(self):
        self._store_group_windows()
        self._check_indices()

    def _check_indices(self):
        for i, window in enumerate(self._windows):
            assert i == window.ix
        for i, const_region in enumerate(self._const_regions):
            assert i == const_region.ix
        for i, group in enumerate(self.region_groups):
            assert i == self._group_by_name[group.name]

    def _store_group_windows(self):
        for window in self._windows:
            self.get_group(self.window_group_name(window)).window_ixs.append(window.ix)

    @property
    def psvs(self):
        return self._psvs

    @property
    def psv_searcher(self):
        return self._psv_searcher

    @property
    def windows(self):
        return self._windows

    @property
    def const_regions(self):
        return self._const_regions

    @property
    def region_groups(self):
        return self._region_groups

    def get_group(self, group_name):
        return self._region_groups[self._group_by_name[group_name]]

    def window_group_name(self, window):
        return self._const_regions[window.const_region_ix].group_name

    def summarize_region_groups(self, genome, out, min_windows):
        for region_group in self._region_groups:
            out.write('Region group {}\n'.format(region_group.name))
            out.write('    Sum length {:,} bp{}\n'.format(len(region_group.region1),
                ' (including breaks)' if len(region_group.region_ixs) > 1 else ''))
            out.write('    Copy number:  {}\n'.format(region_group.cn))
            out.write('    PSVs:    {}\n'.format(len(region_group.psv_ixs)))

            total_windows = len(region_group.window_ixs)
            windows_in_hmm = sum(self._windows[i].in_hmm for i in region_group.window_ixs)
            out.write('    Windows: {} (suitable for HMM: {})\n'.format(total_windows, windows_in_hmm))
            out.write('    Constant regions: {}\n'.format(len(region_group.region_ixs)))
            for region_ix in region_group.region_ixs:
                subregion = self._const_regions[region_ix]
                out.write('    ====\n')
                out.write('    {} ({:,} bp)\n'.format(subregion.region1.to_str_comma(genome), len(subregion.region1)))
                out.write('    {}\n'.format(subregion.regions2_str(genome, use_comma=True, sep='\n    ')))
            out.write('    ====\n')

            if min_windows > 1 and total_windows >= min_windows * 1.5 and windows_in_hmm == 0:
                common.log(('WARN: [{}  {}] cannot use any windows for the agCN HMM (out of total {} windows).\n' +
                    '           If there are many messages like this, consider setting --window-filtering > 1')
                    .format(self.interval.name, region_group.name, total_windows))


class OutputFiles:
    def __init__(self, out_dir, filenames):
        """
        filenames: dictionary { key: path }, for example
        ```
        out = OutputFiles('dir', dict(a='a.csv'))
        out.open()
        out.a # dir/a.csv
        out.close()
        ```
        """
        self._out_dir = out_dir
        self._filenames = filenames
        self._files = None

    def open(self):
        assert self._files is None
        self._files = {}
        for key, filename in self._filenames.items():
            self._files[key] = open(os.path.join(self._out_dir, filename), 'w')

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getattr__(self, key):
        return self._files[key]

    def flush(self):
        for f in self._files.values():
            f.flush()

    def checked_write(self, key, *text):
        if key in self._files:
            for line in text:
                self._files[key].write(line)


class RegionGroupExtra:
    """
    Structure that stores all necessary information for a single region group, as well as Viterbi results.
    """
    def __init__(self, dupl_hierarchy, region_group, psv_read_counts, n_samples, genome):
        from . import paralog_cn

        self._dupl_hierarchy = dupl_hierarchy
        self._region_group = region_group
        windows = dupl_hierarchy.windows
        self._group_windows = [windows[i] for i in region_group.window_ixs]
        self._hmm_windows = [window for window in self._group_windows if window.in_hmm]
        self._windows_searcher = itree.NonOverlTree(self._group_windows, itree.region1_start, itree.region1_end)
        self._hmm_windows_searcher = itree.NonOverlTree(self._hmm_windows, itree.region1_start, itree.region1_end)

        psvs = dupl_hierarchy.psvs
        self._region_psvs = [psvs[i] for i in region_group.psv_ixs]
        self._psv_searcher = itree.NonOverlTree(self._region_psvs, itree.start, itree.variant_end)
        self._psv_read_counts = [psv_read_counts[i] for i in region_group.psv_ixs]

        self._sample_const_regions = None
        self._sample_reliable_regions = None

        self._psv_infos = paralog_cn.create_psv_infos(self._region_psvs, self._region_group, n_samples, genome)
        self._psv_f_values = None

    @property
    def dupl_hierarchy(self):
        return self._dupl_hierarchy

    @property
    def region_group(self):
        return self._region_group

    @property
    def group_windows(self):
        return self._group_windows

    @property
    def group_windows_searcher(self):
        return self._windows_searcher

    @property
    def hmm_windows(self):
        return self._hmm_windows

    @property
    def hmm_windows_searcher(self):
        return self._hmm_windows_searcher

    @property
    def psvs(self):
        return self._region_psvs

    @property
    def psv_searcher(self):
        return self._psv_searcher

    @property
    def psv_read_counts(self):
        return self._psv_read_counts

    def set_viterbi_res(self, sample_const_regions, sample_reliable_regions):
        self._sample_const_regions = sample_const_regions
        self._sample_reliable_regions = sample_reliable_regions

    @property
    def sample_const_regions(self):
        return self._sample_const_regions

    @property
    def sample_reliable_regions(self):
        return self._sample_reliable_regions

    def set_f_values(self, f_values):
        self._psv_f_values = f_values

    def set_from_model_params(self, model_params, n_samples):
        copy_num = self._region_group.cn // 2
        self._psv_f_values = model_params.load_psv_f_values(self._psv_infos, copy_num)
        if self._psv_f_values is None:
            return
        for psv_info in self._psv_infos:
            psv_info.set_use_samples(np.zeros(n_samples, dtype=np.bool))

    def update_psv_records(self, reliable_thresholds):
        from . import paralog_cn

        if not self.has_f_values:
            return
        semirel_threshold, reliable_threshold = reliable_thresholds

        for psv, fval, psv_info in zip(self._region_psvs, self._psv_f_values, self._psv_infos):
            psv.info['fval'] = tuple(fval)
            psv.info['info'] = psv_info.info_content
            min_fval = min(fval)
            psv_type = 'u'
            if min_fval >= reliable_threshold:
                psv_type = 'r'
            elif min_fval >= semirel_threshold:
                psv_type = 's'
            psv.info['rel'] = psv_type

            for sample_id, sample_info in enumerate(psv_info.sample_infos):
                sample_cn = sample_info and sample_info.best_cn
                if sample_cn is None or sample_info.psv_gt_probs[sample_cn] is None:
                    continue
                fmt = psv.samples[sample_id]
                psv_gt_probs = sample_info.psv_gt_probs[sample_cn]
                precomp_data = psv_info.precomp_datas[sample_cn]
                best_gt = np.argmax(psv_gt_probs)

                # For example PSV genotype (4,2) -> 0/0/0/0/1/1
                psv_gt = precomp_data.psv_genotypes[best_gt]
                psv_gt = np.repeat(np.arange(len(psv_gt)), psv_gt)
                fmt['GT'] = tuple(psv_gt)
                fmt['GQ'] = int(common.phred_qual(psv_gt_probs, best_gt))

                support_row = sample_info.support_rows.get(sample_cn)
                if support_row is None:
                    continue
                n_copies = len(precomp_data.sample_genotypes[0])
                _, pscn, pscn_qual = paralog_cn.calculate_marginal_probs(precomp_data.sample_genotypes, support_row,
                    n_copies, sample_cn)
                pscn, pscn_qual, any_known = paralog_cn.paralog_cn_str(pscn, pscn_qual)
                if any_known:
                    fmt['psCN'] = pscn
                    fmt['psCNq'] = pscn_qual

    @property
    def has_f_values(self):
        return self._psv_f_values is not None

    @property
    def psv_f_values(self):
        return self._psv_f_values

    @property
    def psv_infos(self):
        return self._psv_infos

    @property
    def n_samples(self):
        return len(self._sample_const_regions)

    def set_reliable_psvs(self, semirel_threshold, reliable_threshold):
        if self.has_f_values:
            self.psv_in_em = np.array([psv_info.in_em for psv_info in self.psv_infos])
            with np.errstate(invalid='ignore'):
                self.psv_is_reliable = np.all(self.psv_f_values >= reliable_threshold, axis=1) & self.psv_in_em
                self.psv_is_semirel = np.all(self.psv_f_values >= semirel_threshold, axis=1) & self.psv_in_em
        else:
            n_psvs = len(self._psv_infos)
            self.psv_in_em = np.zeros(n_psvs, dtype=np.bool)
            self.psv_is_reliable = np.zeros(n_psvs, dtype=np.bool)
            self.psv_is_semirel = np.zeros(n_psvs, dtype=np.bool)
