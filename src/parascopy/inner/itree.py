import itertools
import numpy as np
from collections import defaultdict
from intervaltree import IntervalTree


class NonOverlTree:
    def __init__(self, objects, start_getter, end_getter):
        self._starts = np.array(list(map(start_getter, objects)))
        self._ends = np.array(list(map(end_getter, objects)))
        assert len(self._starts) == len(self._ends) and np.all(self._ends[:-1] <= self._starts[1:])
        self._objects = objects

    @classmethod
    def empty(cls):
        return cls((), None, None)

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

    def __len__(self):
        return len(self._starts)


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

def identity(obj):
    return obj


def create_interval_tree(objects, start_getter, end_getter, store_indices=False):
    tree = IntervalTree()
    for i, obj in enumerate(objects):
        start = start_getter(obj)
        end = end_getter(obj)
        tree.addi(start, end, i if store_indices else obj)
    return tree


class MultiChromTree:
    def __init__(self):
        self._trees = defaultdict(IntervalTree)
        self._objects = []

    def add(self, region, obj):
        self._trees[region.chrom_id].addi(region.start, region.end, len(self._objects))
        self._objects.append(obj)

    def overlap_ixs(self, region):
        tree = self._trees.get(region.chrom_id)
        if tree is None:
            return
        for overlap in tree.overlap(region.start, region.end):
            yield overlap.data

    def overlap_size(self, region):
        tree = self._trees.get(region.chrom_id)
        return 0 if tree is None else len(tree.overlap(region.start, region.end))


class MultiNonOverlTree:
    def __init__(self, objects, region_getter=identity):
        start_getter = lambda obj: region_getter(obj).start
        end_getter = lambda obj: region_getter(obj).end

        self._trees = []
        curr_objects = []
        curr_chrom_id = None
        for obj in objects:
            chrom_id = region_getter(obj).chrom_id
            if curr_chrom_id is not None and curr_chrom_id == chrom_id:
                curr_objects.append(obj)
            else:
                if curr_objects:
                    for _ in range(len(self._trees), curr_chrom_id):
                        self._trees.append(None)
                    self._trees.append(NonOverlTree(list(curr_objects), start_getter, end_getter))
                    curr_objects.clear()

                curr_objects.append(obj)
                curr_chrom_id = chrom_id

        if curr_objects:
            for _ in range(len(self._trees), curr_chrom_id):
                self._trees.append(None)
            self._trees.append(NonOverlTree(list(curr_objects), start_getter, end_getter))

    def overlap_iter(self, region):
        tree = None if region.chrom_id >= len(self._trees) else self._trees[region.chrom_id]
        if tree is None:
            return iter(())
        return tree.overlap_iter(region.start, region.end)

    def overlap_size(self, region):
        tree = None if region.chrom_id >= len(self._trees) else self._trees[region.chrom_id]
        return 0 if tree is None else tree.overlap_size(region.start, region.end)

    def intersection_size(self, region):
        return sum(interval.intersection_size(region) for interval in self.overlap_iter(region))
