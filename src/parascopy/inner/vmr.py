# =============================================================================
# vmr.py
# -  Identifies samples with low and high variance/mean ratio based on a
#    specified ratio threshold.
# Sang Yoon Byun & Timofey Prodanov
# =============================================================================

import os
import numpy as np
import csv
from collections import defaultdict

from . import common


def select_samples(in_samples, threshold, depth_dirs):
    """
    Select a subset of samples with appropriate variance/mean ratio.
    Returned set of samples.
    """
    GC_CONTENT = 40

    in_samples = set(in_samples)
    # Sample -> list of ratios
    ratios_dict = defaultdict(list)

    for depth_dir in depth_dirs:
        with open(os.path.join(depth_dir, 'depth.csv')) as f:
            fieldnames = None
            for line in f:
                if not line.startswith('#'):
                    fieldnames = line.strip().split('\t')
                    break
            assert fieldnames is not None

            reader = csv.DictReader(f, delimiter='\t', fieldnames=fieldnames)
            for row in reader:
                if float(row['gc_content']) != GC_CONTENT or row['read_end'] != '1':
                    continue
                sample = row['sample']
                if sample not in in_samples:
                    continue
                ratios_dict[sample].append(float(row['var']) / float(row['mean']))

    samples = []
    ratios = []
    for sample, sample_ratios in ratios_dict.items():
        samples.append(sample)
        ratios.append(np.mean(sample_ratios))
    ratios = np.array(ratios)

    if threshold.endswith('%'):
        percentile = 0.01 * float(threshold[:-1])
        assert 0.0 < percentile <= 1.0, f'Invalid VMR threshold {threshold}'
        actual_threshold = max(np.percentile(ratios, percentile), min(ratios))
    else:
        actual_threshold = float(threshold)
        assert 0.0 < actual_threshold, f'Invalid VMR threshold {threshold}'

    ixs = np.where(ratios <= actual_threshold)[0]
    common.log('Use {} / {} samples with VMR under {:.3f} for computing copy number profiles'
        .format(len(ixs), len(samples), actual_threshold))
    if len(ixs) == 0:
        raise RuntimeError('No samples remaining after VMR filter')
    return { samples[i] for i in ixs }
