from enum import Enum
import re
import numpy as np
import itertools
from collections import defaultdict
import gzip
import os
import copy

from .genome import Interval
from . import cn_tools
from . import common
from . import duplication as duplication_
from .. import __version__


class SinglePosition:
    def __init__(self, chrom_id, start):
        self.chrom_id = chrom_id
        self.start = start

    @classmethod
    def from_variant(cls, variant, genome):
        chrom_id = genome.chrom_id(variant.chrom)
        return cls(chrom_id, variant.start)

    @property
    def end(self):
        return self.start + 1

    def to_str(self, genome):
        return '{}:{}'.format(genome.chrom_name(self.chrom_id), self.start + 1)

    @classmethod
    def parse(cls, s, genome):
        chrom_name, pos = s.rsplit(':', 1)
        chrom_id = genome.chrom_id(chrom_name)
        return cls(chrom_id, int(pos) - 1)


class EntryType(Enum):
    Main = 0
    Duplication = 1
    SkipRegion = 2

    RegionGroup = 10
    ConstRegion = 15
    Window = 20
    HmmPaths = 30
    PSV = 40

    def __str__(self):
        if self == EntryType.Main:
            return 'main'
        if self == EntryType.Duplication:
            return 'duplication'
        if self == EntryType.SkipRegion:
            return 'skip_region'
        if self == EntryType.RegionGroup:
            return 'region_group'
        if self == EntryType.ConstRegion:
            return 'const_region'
        if self == EntryType.Window:
            return 'window'
        if self == EntryType.HmmPaths:
            return 'hmm_paths'
        if self == EntryType.PSV:
            return 'psv'
        raise ValueError('Unexpected EntryType')

    @classmethod
    def parse(cls, s):
        s = s.strip()
        if s == 'main':
            return cls.Main
        if s == 'duplication':
            return cls.Duplication
        if s == 'skip_region':
            return cls.SkipRegion
        if s == 'region_group':
            return cls.RegionGroup
        if s == 'const_region':
            return cls.ConstRegion
        if s == 'window':
            return cls.Window
        if s == 'hmm_paths':
            return cls.HmmPaths
        if s == 'psv':
            return cls.PSV
        raise ValueError('Unexpected EntryType {}'.format(s))


class Entry:
    _float_regex = None

    def __init__(self, ty, region1, regions2=None):
        self._ty = ty
        self._region1 = region1
        self._regions2 = regions2
        self._info = {}

    def dupl_region_mismatch(self, dupl_region, genome):
        regions2_match = True
        if self._regions2 or dupl_region.regions2:
            regions2_match = self._regions2 == dupl_region.regions2

        if self._region1 == dupl_region.region1 and regions2_match:
            return None
        s = 'From model parameters:\n'
        s += '    {}\n    '.format(self._region1.to_str_comma(genome))
        s += cn_tools.regions2_str(self._regions2, genome, use_comma=True, sep='\n    ')
        s += '\nFrom this run:\n'
        s += '    {}\n    '.format(dupl_region.region1.to_str_comma(genome))
        s += dupl_region.regions2_str(genome, use_comma=True, sep='\n    ')
        return s

    @property
    def ty(self):
        return self._ty

    @property
    def region1(self):
        return self._region1

    @property
    def regions2(self):
        return self._regions2

    def to_str(self, genome):
        res = '{}\t{}\t'.format(self._ty, self._region1.to_str(genome))
        if self._regions2:
            res += cn_tools.regions2_str(self._regions2, genome, sep=' ')
            res += '\t'
        else:
            res += '*\t'

        if self._info:
            res += ';'.join(map('%s=%s'.__mod__, self._info.items()))
        else:
            res += '*'
        return res

    @classmethod
    def parse(cls, s, genome):
        if Entry._float_regex is None:
            Entry._float_regex = re.compile(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')

        s = s.strip()
        try:
            s_split = s.split('\t')
            s_ty, s_region1, s_regions2, s_info = s_split
            ty = EntryType.parse(s_ty)

            region_cls = Interval if '-' in s_region1 else SinglePosition
            region1 = region_cls.parse(s_region1, genome)
            if s_regions2 == '*':
                regions2 = None
            else:
                regions2 = []
                for entry in s_regions2.split(' '):
                    region, strand = entry.rsplit(':', 1)
                    regions2.append((region_cls.parse(region, genome), strand == '+'))

            self = cls(ty, region1, regions2)
            if s_info != '*':
                for entry in s_info.split(';'):
                    key, value = entry.split('=', 1)
                    if re.match(Entry._float_regex, value):
                        value = (int if value.isdigit() else float)(value)
                    self._info[key] = value
            return self

        except (IndexError, ValueError):
            raise ValueError('Cannot parse entry "{}"'.format(s))

    def __lt__(self, other):
        return (self._region1.start, self._ty.value) < (other._region1.start, other._ty.value)

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        for key in keys:
            assert isinstance(key, str)
            if key in self._info:
                return self._info[key]

        info_str = ', '.join(map('%s: %s'.__mod__, self._info.items()))
        raise KeyError('Keys "{}" not found in the entry {{ {} }}'.format('", "'.join(keys), info_str))

    def __setitem__(self, key, val):
        assert isinstance(key, str)
        self._info[key] = val

    def get(self, key, default=None):
        return self._info.get(key, default)

    def info_items(self):
        return self._info.items()


class ModelParams:
    def __init__(self, main_interval, n_samples, is_loaded):
        self.entries = defaultdict(list)
        if main_interval is not None:
            main_entry = Entry(EntryType.Main, main_interval)
            if main_interval.name_provided:
                main_entry['name'] = main_interval.name
            main_entry['version'] = __version__
            main_entry['n_samples'] = n_samples
            self.add_entry(main_entry)
        self._is_loaded = is_loaded
        self._hmm_entries = None
        self._psv_dict = None

    def save_args(self, args):
        main_entry = self.main_entry
        main_entry['max_ref_cn'] = args.max_ref_cn
        main_entry['agcn_range'] = '{},{}'.format(*args.agcn_range)
        main_entry['agcn_jump'] = args.agcn_jump

        assert args.reliable_threshold[0] <= args.reliable_threshold[1]
        main_entry['reliable_threshold'] = '{:.5g},{:.5g}'.format(*args.reliable_threshold)
        main_entry['pscn_bound'] = '{},{}'.format(*args.pscn_bound)
        main_entry['transition_prob'] = '{:.5g}'.format(args.transition_prob)

    def load_args(self, args):
        main_entry = self.main_entry

        args = copy.copy(args)
        args.min_windows = 1
        args.min_samples = None
        args.agcn_range = tuple(map(int, main_entry[('agcn_range', 'copy_num_range')].split(',')))
        args.max_ref_cn = int(main_entry[('max_ref_cn', 'max_copy_num')])
        args.agcn_jump = int(main_entry[('agcn_jump', 'copy_num_jump')])

        if args.reliable_threshold is None:
            args.reliable_threshold = tuple(map(float, main_entry['reliable_threshold'].split(',')))
        args.pscn_bound = tuple(map(int, main_entry[('pscn_bound', 'copy_num_bound')].split(',')))
        args.transition_prob = float(main_entry['transition_prob'])
        return args

    @property
    def is_loaded(self):
        return self._is_loaded

    def add_entry(self, entry):
        self.entries[entry.ty].append(entry)

    def add_duplication(self, ix, dupl):
        regions2 = ((dupl.region2, dupl.strand),)
        entry = Entry(EntryType.Duplication, dupl.region1, regions2)
        entry['ix'] = ix
        self.add_entry(entry)

    def get_duplications(self, table, interval, genome):
        duplications = []
        for entry in self.entries[EntryType.Duplication]:
            region1 = entry.region1
            assert entry.regions2 and len(entry.regions2) == 1
            region2, strand2 = entry.regions2[0]

            for tup in table.fetch(region1.chrom_name(genome), region1.start, region1.start + 1):
                if int(tup[1]) != region1.start or int(tup[2]) != region1.end:
                    continue
                dupl = duplication_.Duplication.from_tuple(tup, genome)
                if dupl.is_tangled_region:
                    continue
                if dupl.strand != strand2 or dupl.region2 != region2:
                    continue
                ix = int(entry['ix'])
                if len(duplications) <= ix:
                    duplications.extend((None,) * (ix - len(duplications) + 1))
                duplications[ix] = dupl
                break
            else:
                msg = self.mismatch_warning(genome)
                msg += '\nCannot find duplication in the table that would match duplication {}  {}:{}'.format(
                    region1.to_str_comma(genome), region2.to_str_comma(genome), '+' if strand2 else '-')
                raise RuntimeError(msg)

        assert all(dupl is not None for dupl in duplications)
        return duplications

    def set_skip_regions(self, skip_regions):
        assert not self._is_loaded
        for region in skip_regions:
            self.add_entry(Entry(EntryType.SkipRegion, region))

    def set_dupl_hierarchy(self, dupl_hierarchy):
        assert not self._is_loaded
        for window in dupl_hierarchy.windows:
            entry = Entry(EntryType.Window, window.region1, window.regions2)
            entry['ix'] = window.ix
            entry['const_region'] = window.const_region_ix
            entry['gc_content'] = '{:.0f}'.format(window.gc_content)
            entry['in_hmm'] = 'T' if window.in_hmm else 'F'
            self.add_entry(entry)

        for const_region in dupl_hierarchy.const_regions:
            entry = Entry(EntryType.ConstRegion, const_region.region1, const_region.regions2)
            entry['ix'] = const_region.ix
            entry['group'] = const_region.group_name or '*'
            entry['skip'] = 'T' if const_region.skip else 'F'
            self.add_entry(entry)

        for region_group in dupl_hierarchy.region_groups:
            entry = Entry(EntryType.RegionGroup, region_group.region1, region_group.regions2)
            entry['name'] = region_group.name
            self.add_entry(entry)

    def check_dupl_hierarchy(self, dupl_hierarchy, genome):
        window_entries = self.entries[EntryType.Window]
        if len(window_entries) != len(dupl_hierarchy.windows):
            return 'Region has {} windows while loaded model parameters have {} windows'.format(
                len(dupl_hierarchy.windows), len(window_enties))
        for i, entry in enumerate(window_entries):
            window = dupl_hierarchy.windows[i]
            dupl_region_mism = entry.dupl_region_mismatch(window, genome)
            if window.ix != i or dupl_region_mism \
                    or int(entry['const_region']) != window.const_region_ix \
                    or int(entry['gc_content']) != window.gc_content:
                return 'Window {} {} does not match with corresponding window in the model parameters:\n{}' \
                    .format(window.ix, window.region1.to_str(genome), dupl_region_mism)
            window.in_hmm = entry[('in_hmm', 'in_viterbi')] == 'T'

        region_entries = self.entries[EntryType.ConstRegion]
        if len(region_entries) != len(dupl_hierarchy.const_regions):
            return 'Region has {} constant regions while loaded model parameters have {} constant regions'.format(
                len(dupl_hierarchy.const_regions), len(region_enties))
        for i, entry in enumerate(region_entries):
            const_region = dupl_hierarchy.const_regions[i]
            entry_skip = entry['skip'] == 'T'
            entry_group = entry['group']
            if entry_group == '*':
                entry_group = None
            dupl_region_mism = entry.dupl_region_mismatch(const_region, genome)
            if const_region.ix != int(entry['ix']) or const_region.group_name != entry_group \
                    or const_region.skip != entry_skip or dupl_region_mism:
                return 'Constant region {} {} does not match with corresponding const region in the model parameters:\n{}' \
                    .format(const_region.ix, const_region.region1.to_str(genome), dupl_region_mism)

        group_entries = self.entries[EntryType.RegionGroup]
        if len(group_entries) != len(dupl_hierarchy.region_groups):
            return 'Region has {} region groups while model parameters have {} region groups'.format(
                len(dupl_hierarchy.region_groups), len(group_entries))
        for entry in group_entries:
            try:
                region_group = dupl_hierarchy.get_group(entry['name'])
            except KeyError:
                return 'Model parameters has unmatched group {} {}' \
                    .format(entry['name'], entry.region1.to_str(genome))
            dupl_region_mism = entry.dupl_region_mismatch(region_group, genome)
            if dupl_region_mism:
                return 'Region group {} {} does not match with corresponding region group in the model parameters:\n{}' \
                    .format(region_group.name, region_group.region1.to_str(genome), dupl_region_mism)

    def set_hmm_results(self, region_group, window_ixs, model, multipliers, paths):
        jump_probs = model.middle_state_transitions()
        jumps_len = jump_probs.shape[0]
        windows_entries = self.entries[EntryType.Window]

        for i, window_ix in enumerate(window_ixs):
            entry = windows_entries[window_ix]
            assert entry['ix'] == window_ix
            entry['mult'] = '{:.8f}'.format(multipliers[i])
            if i < jumps_len:
                entry['jumps'] = ','.join(map('{:.8f}'.format, -jump_probs[i]))

        paths_entry = Entry(EntryType.HmmPaths, region_group.region1)
        paths_entry['group'] = region_group.name
        min_cn = model.get_copy_num(0)
        max_cn = model.get_copy_num(model.n_hidden - 1)
        paths_entry['copy_nums'] = '{},{}'.format(min_cn, max_cn)
        paths_entry['initial'] = ','.join(map('{:.8f}'.format, -model.initial))

        for path in paths.values():
            path_str = ','.join('{}-{}-{}'.format(segment.start_ix, segment.end_ix, segment.cn) for segment in path)
            paths_entry['_' + path.name] = '{:.8f},{}'.format(path.weight, path_str)
        self.add_entry(paths_entry)

    def get_hmm_entry(self, group_name):
        if self._hmm_entries is None:
            self._hmm_entries = {}
            for entry in self.entries[EntryType.HmmPaths]:
                self._hmm_entries[entry['group']] = entry
        return self._hmm_entries[group_name]

    def get_hmm_paths(self, group_name):
        entry = self.get_hmm_entry(group_name)
        for key, value in entry.info_items():
            if key.startswith('_'):
                yield key[1:], value

    def get_hmm_data(self, window_ixs, max_state_dist):
        n_observations = len(window_ixs)
        multipliers = np.zeros(n_observations)
        jump_probs = np.zeros((n_observations - 1, 3))

        window_entries = self.entries[EntryType.Window]
        for i, window_ix in enumerate(window_ixs):
            entry = window_entries[window_ix]
            assert entry['ix'] == window_ix
            multipliers[i] = float(entry['mult'])
            if i < n_observations - 1:
                jump_probs[i] = -np.array(list(map(float, entry['jumps'].split(','))))
        return multipliers, jump_probs

    def set_psv_f_values(self, group_extra, genome):
        group_name = group_extra.region_group.name
        for i, psv in enumerate(group_extra.psvs):
            entry = Entry(EntryType.PSV, SinglePosition.from_variant(psv, genome))
            entry['group'] = group_name
            if group_extra.has_f_values:
                psv_info = group_extra.psv_infos[i]
                entry['in_em'] = 'T' if psv_info.in_em else 'F'
                entry['info'] = '{:.8f}'.format(psv_info.info_content)
                entry['fval'] = ','.join(map('{:.8f}'.format, group_extra.psv_f_values[i]))
            else:
                entry['in_em'] = 'F'
                entry['info'] = '*'
                entry['fval'] = '*'
            self.add_entry(entry)

    def write_to(self, out, genome):
        for entry in sorted(itertools.chain(*self.entries.values())):
            out.write(entry.to_str(genome))
            out.write('\n')

    @classmethod
    def load(cls, inp, genome):
        self = cls(None, None, True)
        for line in inp:
            if line.startswith('#'):
                continue
            self.add_entry(Entry.parse(line, genome))
        return self

    @property
    def main_entry(self):
        return self.entries[EntryType.Main][0]

    def mismatch_warning(self, genome):
        main_entry = self.main_entry
        version = main_entry['version']
        name = main_entry.get('name')
        if name is None:
            name = main_entry.region1.to_str(genome)

        msg = 'Error: Model parameters for region {} ({}) does not match current run ({}).\n'.format(
            name, version, __version__)
        msg += '    Possible explanations:\n'
        if version != __version__:
            msg += '        - Version mismatch,\n'
        msg += '        - Different arguments,\n'
        msg += '        - Different duplications table.'
        return msg

    def get_skip_regions(self, skip_regions, genome):
        skip_entries = self.entries[EntryType.SkipRegion]
        skip_entries.sort()
        i = 0
        res = []
        for skip_entry in skip_entries:
            region = skip_entry.region1
            res.append(region)
            while i < len(skip_regions) and skip_regions[i].intersects(region):
                if not skip_regions[i].contains(region):
                    common.log('WARN: Skipped region {} is outside of the skipped regions from model parameters.'
                        .format(skip_regions[i].to_str(genome)) +
                        ' Only skipped regions from model parameters would be used.')
                i += 1
        return res

    def load_psv_f_values(self, psv_infos, copy_num):
        psv_entries = self.entries[EntryType.PSV]
        if self._psv_dict is None:
            self._psv_dict = { psv.region1.start: i for i, psv in enumerate(psv_entries) }

        n_psvs = len(psv_infos)
        psv_f_values = np.full((n_psvs, copy_num), np.nan)
        for psv_info in psv_infos:
            try:
                psv_entry = psv_entries[self._psv_dict[psv_info.start]]
            except (KeyError, IndexError):
                raise RuntimeError('Cannot find PSV {}:{} in model parameters'
                    .format(psv_info.chrom, psv_info.start + 1))
            if psv_entry['info'] == '*':
                continue
            psv_info.in_em = psv_entry['in_em'] == 'T'
            psv_info.info_content = float(psv_entry['info'])
            psv_f_values[psv_info.psv_ix] = list(map(float, psv_entry['fval'].split(',')))

        if not any(psv_info.in_em for psv_info in psv_infos):
            return None
        return psv_f_values


def load_all(input, genome):
    models = []
    for path in input:
        if os.path.isdir(path):
            filenames = [os.path.join(path, filename) for filename in os.listdir(path) if filename.endswith('.gz')]
        elif path.endswith('.gz'):
            filenames = (path,)
        else:
            curr_dir = os.path.dirname(path)
            with open(path) as inp:
                filenames = [os.path.join(curr_dir, line.strip()) for line in inp]

        for filename in filenames:
            try:
                with gzip.open(filename, 'rt') as curr_inp:
                    model_params = ModelParams.load(curr_inp, genome)
                _test = model_params.main_entry
                models.append(model_params)
            except (ValueError, IndexError) as e:
                common.log('ERROR: Cannot parse model file "{}":\n{}'.format(filename, e))
    return models
