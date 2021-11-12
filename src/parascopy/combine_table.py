#!/usr/bin/env python3

import argparse
import sys
import operator
import pysam
import os
import numpy as np
import itertools
import multiprocessing
import traceback

from . import pretable
from .view import parse_expression
from .inner.genome import Genome, Interval
from .inner.duplication import Duplication, TangledRegion, combine_segments
from .inner import common
from .inner.alignment import Weights
from . import long_version


def _start_distance_reg2(dupl1, dupl2):
    """
    Returns distance between starts of the regions2.
    It is assumed that the duplications are on the same chromosome.
    """
    return dupl2.region2.start - dupl1.region2.start if dupl1.strand else dupl1.region2.end - dupl2.region2.end


def _extension_sizes(dupl1, dupl2):
    """
    If dupl2 cannot extend dupl1 (different locations or strands), returns None.
    Otherwise, returns two numbers: extension1 and extension2, which represent the sizes of new regions covered by dupl2.

    It is assumed that dupl1.region1.intersects(dupl2.region1).
    """
    if dupl1.strand != dupl2.strand or not dupl1.region2.intersects(dupl2.region2):
        return None

    if dupl1.strand:
        if dupl1.region2.start > dupl2.region2.start:
            return None
        extension2 = dupl2.region2.end - dupl1.region2.end
    else:
        if dupl2.region2.end > dupl1.region2.end:
            return None
        extension2 = dupl1.region2.start - dupl2.region2.start

    extension1 = dupl2.region1.end - dupl1.region1.end
    return extension1, extension2


def _select_best_edges(distances, duplications, curr_dupl):
    """
    Returns iterator over indices of edges, for which both distances are similar and not too high.
    """
    curr_len1 = len(curr_dupl.region1)
    curr_len2 = len(curr_dupl.region2)

    for i, (dist1, dist2) in enumerate(distances):
        dupl = duplications[i]
        len1 = min(curr_len1, len(dupl.region1))
        len2 = min(curr_len2, len(dupl.region2))
        if dist1 <= len1 / 2 and dist2 <= len2 / 2 and min(dist1, dist2) >= max(dist1, dist2) / 2:
            yield i


class _Graph:
    def __init__(self):
        self._nodes = []
        # Out/In edges: adjacency vector (same length as nodes, each element: set of indices).
        self._out_edges = []
        self._in_edges = []
        self._tangled_nodes = None
        self._removed_edges = set()

    def add_node(self, value):
        """
        Returns node id.
        """
        self._nodes.append(value)
        self._out_edges.append(set())
        self._in_edges.append(set())
        return len(self._nodes) - 1

    def remove_last(self):
        """
        Removes last node.
        """
        self._nodes.pop()
        node_b = len(self._nodes)
        for node_a in self._in_edges[node_b]:
            self._out_edges[node_a].remove(node_b)
        for node_c in self._out_edges[node_b]:
            self._in_edges[node_c].remove(node_b)

        assert self._tangled_nodes is None
        self._out_edges.pop()
        self._in_edges.pop()

    def get_value(self, node_id):
        return self._nodes[node_id]

    def add_edge(self, node_a, node_b):
        self._out_edges[node_a].add(node_b)
        self._in_edges[node_b].add(node_a)

    def try_remove_edge(self, node_a, node_b):
        try:
            self._out_edges[node_a].remove(node_b)
            self._in_edges[node_b].remove(node_a)
        except KeyError:
            pass

    def remove_edge_record(self, node_a, node_b):
        self._out_edges[node_a].remove(node_b)
        self._in_edges[node_b].remove(node_a)
        self._removed_edges.add((node_a, node_b))

    def clear(self):
        self._nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()

    def write_dot(self, outp, genome, colored_nodes=None):
        """
        colored_nodes: list of bools.
        """
        outp.write('Digraph {\n')
        for node_a, dupl in enumerate(self._nodes):
            out_edges = self._out_edges[node_a]
            if colored_nodes is not None and colored_nodes[node_a]:
                prop = ';fillcolor=red;style=filled'
            else:
                prop = ''

            if dupl.is_tangled_region:
                outp.write('    {} [label="{}\\nTangled region"{}]\n'
                    .format(node_a, dupl.region1.to_str_comma(genome), prop))
            else:
                outp.write('    {} [label="{}\\n{}\\nlen={}, {} strand"{}]\n'
                    .format(node_a, dupl.region1.to_str_comma(genome), dupl.region2.to_str_comma(genome),
                    len(dupl.region1), dupl.strand_str, prop))

            for node_b in out_edges:
                prop = ''
                if (node_a, node_b) in self._removed_edges:
                    prop = ' [color=red;arrowhead=noneicurvecurve;arrowsize=1.5]'
                outp.write('    {} -> {}{}\n'.format(node_a, node_b, prop))
        outp.write('}\n')

    def transitive_reduction(self):
        """
        Removes all transitive edges.
        """
        to_remove = []
        for node_a, out_a in enumerate(self._out_edges):
            to_remove.clear()
            for node_b in out_a:
                out_b = self._out_edges[node_b]
                to_remove.extend(out_a & out_b)

            for node_c in to_remove:
                self.try_remove_edge(node_a, node_c)

    def _prune_multiple_input_edges(self, curr_node):
        curr_dupl = self._nodes[curr_node]
        curr_region1 = curr_dupl.region1

        prev_nodes = list(self._in_edges[curr_node])
        prev_dupls = [self._nodes[i] for i in prev_nodes]
        # List of pairs (dist1, dist2), both numbers represent distance between the previous and current dupl,
        # one for regions1, and one for regions2.
        distances = [
            (curr_region1.start - dupl.region1.start, _start_distance_reg2(dupl, curr_dupl))
            for dupl in prev_dupls]

        best = set(_select_best_edges(distances, prev_dupls, curr_dupl))
        if 0 < len(best) < len(prev_dupls):
            for i, node_id in enumerate(prev_nodes):
                if i not in best:
                    self.remove_edge_record(node_id, curr_node)

    def _prune_multiple_output_edges(self, curr_node):
        curr_dupl = self._nodes[curr_node]
        curr_region1 = curr_dupl.region1

        next_nodes = list(self._out_edges[curr_node])
        next_dupls = [self._nodes[i] for i in next_nodes]
        # List of pairs (dist1, dist2), both numbers represent distance between the previous and current dupl,
        # one for regions1, and one for regions2.
        distances = [
            (dupl.region1.start - curr_region1.start, _start_distance_reg2(curr_dupl, dupl))
            for dupl in next_dupls]

        best = set(_select_best_edges(distances, next_dupls, curr_dupl))
        if 0 < len(best) < len(next_dupls):
            for i, node_id in enumerate(next_nodes):
                if i not in best:
                    self.remove_edge_record(curr_node, node_id)

    def remove_redundant(self):
        """
        For nodes with several input or output edges, tries to remove some of them.
        """
        for curr_node in range(len(self._nodes)):
            if len(self._in_edges[curr_node]) > 1:
                self._prune_multiple_input_edges(curr_node)
            if len(self._out_edges[curr_node]) > 1:
                self._prune_multiple_output_edges(curr_node)

    def remove_tangled_edges(self):
        for curr in np.where(self._tangled_nodes)[0]:
            for prev in tuple(self._in_edges[curr]):
                self.remove_edge_record(prev, curr)
            for next in tuple(self._out_edges[curr]):
                self.remove_edge_record(curr, next)

    def __len__(self):
        return len(self._nodes)

    def __bool__(self):
        return bool(self._nodes)

    def has_multidegree_nodes(self):
        return any(len(edges) > 1 for edges in self._in_edges) or any(len(edges) > 1 for edges in self._out_edges)

    def find_self_overlapping(self):
        total = 0
        for i, dupl in enumerate(self._nodes):
            # Do we care about strand?
            if not self._tangled_nodes[i] and dupl.region1.intersects(dupl.region2):
                self._tangled_nodes[i] = True
                total += 1
        return total

    def _region_from_path(self, path):
        start_region = self.get_value(path[0]).region1
        end_region = self.get_value(path[-1]).region1
        return Interval(start_region.chrom_id, start_region.start, end_region.end)

    def find_bounding_regions(self, paths, node_in_path):
        # Get path from node id.
        path_rev = [None] * len(self)
        for i, path in enumerate(paths):
            for node_id in path:
                path_rev[node_id] = i

        bounding_regions = [None] * len(paths)
        for i, path in enumerate(paths):
            start = path[0]
            for prev in self._in_edges[start]:
                if not node_in_path[prev]:
                    continue
                assert not self._tangled_nodes[prev]

                # There is an edge between two paths.
                prev_i = path_rev[prev]
                prev_path = paths[prev_i]
                assert prev_path[-1] == prev

                if bounding_regions[prev_i] is None:
                    bounding_regions[prev_i] = self._region_from_path(prev_path)
                if bounding_regions[i] is None:
                    bounding_regions[i] = self._region_from_path(path)

                prev_region = self._nodes[prev].region1
                start_region = self._nodes[start].region1
                assert prev_region.end > start_region.start
                split_point = (start_region.start + prev_region.end) // 2
                bounding_regions[prev_i] = bounding_regions[prev_i].with_min_end(split_point)
                bounding_regions[i] = bounding_regions[i].with_max_start(split_point)
                self._removed_edges.add((prev, start))

            # Do not need to check path[-1] -> next, because it will be checked in the start of another path.
        return bounding_regions

    def find_simple_paths(self, min_nodes):
        """
        Finds straight lines in the graph (all nodes with one in- and out-edge).
        The path should represent a full connected component or contain at least `min_nodes`.

        Returns iterator over lists with node indices.
        """
        for start_id, start_in_edges in enumerate(self._in_edges):
            if self._tangled_nodes[start_id]:
                assert not start_in_edges and not self._out_edges[start_id]
                continue

            if len(start_in_edges) == 1:
                prev_id = next(iter(start_in_edges))
                if len(self._out_edges[prev_id]) == 1:
                    continue

            path = [start_id]
            curr_id = start_id
            while len(self._out_edges[curr_id]) == 1:
                next_id = next(iter(self._out_edges[curr_id]))
                if len(self._in_edges[next_id]) != 1:
                    break
                curr_id = next_id
                path.append(curr_id)

            if len(path) >= min_nodes or (not start_in_edges and not self._out_edges[curr_id]):
                for subpath in _split_self_overlapping_path(path, self):
                    yield subpath

    def add_tangled_nodes(self, tangled_predicate, genome):
        self._tangled_nodes = np.zeros(len(self._nodes), dtype=np.bool)
        for i, dupl in enumerate(self._nodes):
            if dupl.is_tangled_region or tangled_predicate(dupl, genome):
                self._tangled_nodes[i] = True

    def extend_tangled_regions(self, node_in_path, tangled_regions):
        ixs = np.where(self._tangled_nodes | ~node_in_path)[0]
        for i in ixs:
            tangled_regions.append(self._nodes[i].region1)

    @property
    def tangled_nodes(self):
        return self._tangled_nodes


def _split_self_overlapping_path(path, graph):
    """
    Returns iterator over lists (new paths).
    """
    n = len(path)
    dupl0 = graph.get_value(path[0])
    if dupl0.region1.chrom_id != dupl0.region2.chrom_id:
        yield path
        return

    region1 = dupl0.region1
    region2 = dupl0.region2
    start_i = 0
    for i in range(1, n):
        curr_region1 = graph.get_value(path[i]).region1
        curr_region2 = graph.get_value(path[i]).region2

        next_region1 = region1.combine(curr_region1)
        next_region2 = region2.combine(curr_region2)

        if next_region1.intersects(next_region2):
            assert i > start_i

            yield path[start_i:i]
            start_i = i
            region1 = curr_region1
            region2 = curr_region2
        else:
            region1 = next_region1
            region2 = next_region2
    if start_i < n:
        yield path[start_i:n]


def _path_segments(graph, paths, bounding_regions, tangled_regions):
    for path, bound in zip(paths, bounding_regions):
        segments = []
        for i, node_id in enumerate(path):
            segment = graph.get_value(node_id)
            if bound is not None:
                segment = segment.sub_duplication(bound)
                if i:
                    segment_reg1 = segment.region1
                    prev = graph.get_value(path[i - 1])
                    prev_reg1 = prev.region1
                    if prev_reg1.contains(segment_reg1) or prev.region2.contains(segment.region2):
                        # After bounding the segment, it no longer brings new information.
                        continue
                    if not prev_reg1.intersects(segment_reg1) or not prev.region2.intersects(segment.region2):
                        # For some reason, after cropping regions, they stopped overlapping.
                        # Likely, this means there is a short duplicated pattern
                        yield segments
                        segments = []
                        tangled_start = min(prev_reg1.start, segment_reg1.start)
                        tangled_end = min(prev_reg1.end, segment_reg1.end)
                        tangled_regions.append(Interval(prev_reg1.chrom_id, tangled_start, tangled_end))
            segments.append(segment)
        if segments:
            yield segments


def _save_component(graph, chrom_names, component_id, graph_dir):
    """
    Simplifies duplications graph, extracts simple paths from it, and constructs combined duplications.
    Returns sorted list of combined duplications.
    """
    aln_fun = Weights().create_aln_fun()
    n_self_overl = graph.find_self_overlapping()
    graph.remove_tangled_edges()
    graph.transitive_reduction()
    if graph.has_multidegree_nodes():
        graph.remove_redundant()

    MIN_NODES = 4
    paths = list(graph.find_simple_paths(MIN_NODES))
    total_nodes = len(graph)
    node_in_path = np.zeros(total_nodes, dtype=np.bool)
    for node_id in itertools.chain(*paths):
        node_in_path[node_id] = True
    n_in_path = sum(node_in_path)
    bounding_regions = graph.find_bounding_regions(paths, node_in_path)

    if graph_dir:
        n_tangled = total_nodes - n_in_path - n_self_overl
        if n_tangled:
            common.log('    Component {}: {} tangled nodes'.format(component_id, n_tangled))
        if n_self_overl:
            common.log('    Component {}: {} self-overlapping nodes'.format(component_id, n_self_overl))
        with open(os.path.join(graph_dir, '{:05d}.dot'.format(component_id)), 'w') as graph_out:
            graph.write_dot(graph_out, chrom_names)

    combined = []
    tangled_regions = []
    for segments in _path_segments(graph, paths, bounding_regions, tangled_regions):
        comb_dupl = combine_segments(segments, aln_fun)
        comb_dupl.info['entries'] = len(segments)
        comb_dupl.store_stats()
        comb_dupl.estimate_complexity()
        combined.append(comb_dupl)

    graph.extend_tangled_regions(node_in_path, tangled_regions)
    tangled_regions = Interval.combine_overlapping(tangled_regions, max_dist=100)
    for region in tangled_regions:
        combined.append(TangledRegion(region))
    combined.sort(key=operator.attrgetter('region1'))
    return combined


def _save_component_wrapper(tup):
    try:
        return _save_component(*tup)
    except:
        common.log('CATCH')
        fmt_exc = traceback.format_exc()
        common.log('ERROR:\n{}'.format(fmt_exc))
        return None


def _iterate_components(duplications, tangled_predicate, chrom_names, graph_dir):
    """
    Returns iterator of tuples (graph, chrom_names, component_id, graph_dir).
    Such strange format is needed to pass this iterator to Pool.imap.
    """
    graph = _Graph()
    component_id = 0
    reachable = set()
    for i, dupl2 in enumerate(duplications):
        dupl_region = dupl2.region1
        node2 = None

        # Use list(reachable) because we may remove nodes during iteration.
        for node1 in list(reachable):
            dupl1 = graph.get_value(node1)
            if not dupl1.region1.intersects(dupl_region):
                reachable.remove(node1)
                continue
            if dupl1.is_tangled_region or dupl2.is_tangled_region:
                continue

            extension = _extension_sizes(dupl1, dupl2)
            if extension is not None:
                ext1, ext2 = extension
                if ext1 <= 0 and ext2 <= 0:
                    # Both regions are contained within previous duplication,
                    # this duplication does not have any new information.
                    if node2 is not None:
                        assert node2 == len(graph) - 1
                        graph.remove_last()
                    dupl2 = None
                    break

                if ext1 <= 0 or ext2 <= 0:
                    # Does not have new information on at least one of the regions.
                    continue

                if node2 is None:
                    node2 = graph.add_node(dupl2)
                graph.add_edge(node1, node2)

        if dupl2 is None:
            continue
        if not reachable and graph:
            graph.add_tangled_nodes(tangled_predicate, chrom_names)
            yield (graph, chrom_names, component_id, graph_dir)
            graph = _Graph()
            component_id += 1

        if node2 is None:
            node2 = graph.add_node(dupl2)
        reachable.add(node2)

    if graph:
        graph.add_tangled_nodes(tangled_predicate, chrom_names)
        yield (graph, chrom_names, component_id, graph_dir)


def combine(duplications, tangled_predicate, chrom_names, graph_dir, threads):
    """
    Parameters:
        - duplications: iterator over lines, each line stores a single Duplication (must be sorted by region1).
        - graph_dir: directory, where .dot files with duplications graph will be saved (optional, may be None).
    Returns a list of merged duplications, sorted by region1.
    """
    components_iter = _iterate_components(duplications, tangled_predicate, chrom_names, graph_dir)
    if threads == 1:
        res = map(_save_component_wrapper, components_iter)
        pool = None
    else:
        pool = multiprocessing.Pool(threads)
        res = pool.imap(_save_component_wrapper, components_iter, chunksize=1)

    for dupl_group in res:
        if dupl_group is None:
            os._exit(1)
        for dupl in dupl_group:
            yield dupl

    if pool is not None:
        pool.terminate()


def _duplications_iterator(table, args, genome):
    exclude_dupl = parse_expression(args.exclude)
    for tup in common.fetch_iterator(table, args, genome):
        dupl = Duplication.from_tuple(tup, genome)
        if dupl.is_tangled_region:
            yield dupl
        elif not exclude_dupl(dupl, genome):
            dupl.set_cigar_from_info()
            dupl.set_sequences(genome=genome)
            yield dupl


def main(prog_name=None, in_args=None):
    prog_name = prog_name or '%(prog)s'
    parser = argparse.ArgumentParser(
        description='Convert homology pre-table into homology table.\n'
            'This command combines overlapping homologous regions into longer duplications.',
        formatter_class=common.SingleMetavar, add_help=False,
        usage='{} -i <pretable> -f <fasta> -o <table> [arguments]'.format(prog_name))
    io_args = parser.add_argument_group('Input/output arguments')
    io_args.add_argument('-i', '--input', metavar='<file>', required=True,
        help='Input indexed bed.gz homology pre-table.')
    io_args.add_argument('-f', '--fasta-ref', metavar='<file> [<file>]', required=True,
        help='Input reference fasta file.')
    io_args.add_argument('-o', '--output', metavar='<file>', required=True,
        help='Output table bed[.gz] file with homology table.')
    io_args.add_argument('-g', '--graph', metavar='<file>',
        help='Optional: output directory with duplication graphs.')

    reg_args = parser.add_argument_group('Region arguments (optional)')
    reg_args.add_argument('-r', '--regions', nargs='+', metavar='<region>',
        help='Region(s) in format "chr" or "chr:start-end").\n'
            'Start and end are 1-based inclusive. Commas are ignored.')
    reg_args.add_argument('-R', '--regions-file', nargs='+', metavar='<file>',
        help='Input bed[.gz] file(s) containing regions (tab-separated, 0-based semi-exclusive).')

    filt_args = parser.add_argument_group('Duplications filtering arguments')
    filt_args.add_argument('-e', '--exclude', metavar='<expr>',
        default='length < 500 && seq_sim < 0.97',
        help='Exclude duplications for which the expression is true\n[default: %(default)s].')
    filt_args.add_argument('-t', '--tangled', metavar='<expr>',
        default='sep < 5000 && av_mult > 2',
        help='Exclude duplications for which the expression is true and mark regions\n'
            'as "tangled". These regions will be discarded from downstream analysis.\n'
            'Regions in args.exclude will be discarded first.\n'
            'By default, this discards tandem duplications with high multiplicity\n'
            '(low complexity). [default: %(default)s].')

    opt_args = parser.add_argument_group('Optional arguments')
    opt_args.add_argument('--tabix', metavar='<path>', default='tabix',
        help='Path to "tabix" executable [default: %(default)s].')
    opt_args.add_argument('-@', '--threads', type=int, metavar='<int>', default=4,
        help='Use <int> threads [default: %(default)s].')

    oth_args = parser.add_argument_group('Other arguments')
    oth_args.add_argument('-h', '--help', action='help', help='Show this help message')
    oth_args.add_argument('-V', '--version', action='version',
        version=long_version(), help='Show version.')
    args = parser.parse_args(in_args)

    common.log('Using {} threads'.format(args.threads))
    if args.graph is not None:
        common.mkdir(args.graph)

    tangled_predicate = parse_expression(args.tangled)
    with pysam.TabixFile(args.input, parser=pysam.asTuple()) as table, Genome(args.fasta_ref) as genome, \
            common.open_possible_gzip(args.output, 'w', bgzip=True) as outp:
        pretable.write_header(genome, outp, sys.argv)
        duplications = _duplications_iterator(table, args, genome)

        chrom_names = genome.to_chrom_names()
        i = 0
        for dupl in combine(duplications, tangled_predicate, chrom_names, args.graph, args.threads):
            i += 1
            if i % 1000 == 0:
                region = dupl.region1
                common.log('    {}:{:12,}    Combined {:9,} duplications.'
                    .format(region.chrom_name(chrom_names), region.start + 1, i))
            outp.write(dupl.to_str(chrom_names))
            outp.write('\n')

    if args.output.endswith('.gz'):
        common.log('Index output with tabix')
        common.Process([args.tabix, '-p', 'bed', args.output]).finish()
    common.log('Success')


if __name__ == '__main__':
    main()
