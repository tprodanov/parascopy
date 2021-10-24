import itertools
import numpy as np
from intervaltree import IntervalTree


class NonOverlTree:
    def __init__(self, objects, start_getter, end_getter):
        self._starts = np.array(list(map(start_getter, objects)))
        self._ends = np.array(list(map(end_getter, objects)))
        assert len(self._starts) == len(self._ends) and np.all(self._ends[:-1] <= self._starts[1:])
        self._objects = objects

    def overlap_ixs(self, start, end):
        start_ix = self._ends.searchsorted(start, side='right')
        end_ix = self._starts.searchsorted(end, side='left')
        return start_ix, end_ix

    def overlap_size(self, start, end):
        start_ix, end_ix = self.overlap_ixs(start, end)
        return end_ix - start_ix

    def overlap_iter(self, start, end):
        start_ix, end_ix = self.overlap_ixs(start, end)
        return itertools.islice(self._objects, int(start_ix), int(end_ix))

    def contained_ixs(self, start, end):
        start_ix = self._starts.searchsorted(start, side='left')
        end_ix = self._ends.searchsorted(end, side='right')
        return start_ix, end_ix


def start(obj):
    return obj.start

def end(obj):
    return obj.end

def variant_end(variant):
    return variant.start + len(variant.ref)

def region1_start(dupl_region):
    return dupl_region.region1.start

def region1_end(dupl_region):
    return dupl_region.region1.end


def create_interval_tree(objects, start_getter, end_getter, store_indices=False):
    tree = IntervalTree()
    for i, obj in enumerate(objects):
        start = start_getter(obj)
        end = end_getter(obj)
        tree.addi(start, end, i if store_indices else obj)
    return tree
