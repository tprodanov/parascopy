# =============================================================================
# vmr.py
# -  Identifies samples with low and high variance/mean ratio based on a
#    specified ratio threshold.
# =============================================================================

import os
import pandas as pd
import numpy as np
import argparse
    
from .inner import common

def compute_vmr(inp_depth_dir, threshold_data):
    
    # Read depth.csv
    inp_depth_fp = os.path.join(inp_depth_dir, "depth.csv")
    depth_df = pd.read_csv(inp_depth_fp, sep="\t", comment='#')
    
    # Unpack threshold arguments
    threshold_value, threshold_percentile = threshold_data

    # Compute variance/mean ratio per sample
    depth_vm_df = depth_df.loc[depth_df.gc_content==50].groupby(by="sample")[['mean', 'var']].mean()
    depth_vm_df['ratio'] = depth_vm_df['var'] / depth_vm_df['mean']

    # Set threshold for determining high vs low vmr
    if threshold_value:
        VMR_THRESHOLD = float(threshold_value)
    if threshold_percentile:
        percentile = float(threshold_percentile)
        VMR_THRESHOLD = np.percentile(depth_vm_df.ratio, percentile)

    # Determine which samples to keep vs. discard based on vm-ratio
    low_vm_df = depth_vm_df.loc[depth_vm_df.ratio < VMR_THRESHOLD].reset_index()
    high_vm_df = depth_vm_df.loc[depth_vm_df.ratio >= VMR_THRESHOLD].reset_index()
    
    print(f"VMR THRESHOLD: {VMR_THRESHOLD}")
    print(f"Total of {len(high_vm_df)} samples with high variance/mean ratio will be removed.")
    print(f"{len(low_vm_df)} samples had low variance/mean ratio and will be used for computing copy-number profiles.")
   
    # Save intermediate files in depth directory
    print(f"Saving VMR files to {os.path.join(inp_depth_dir, 'vmr')}")
    save_intermediaries(depth_vm_df, low_vm_df, high_vm_df, inp_depth_dir)

    return low_vm_df['sample'].to_list()


def plot_histogram(depth_vm_df, thresh, outp_directory):

    plt.figure(figsize=(12,8))

    ax = plt.hist(depth_vm_df.ratio, bins=100)
    plt.axvline(x=thresh, ymin=0, ymax=1, color='red')

    plt.xlabel("Variance/Mean Ratio")
    plt.ylabel("Count")

    plt.text(thresh+0.01, ax[0].max(), 
             f'Threshold: {thresh}', color = 'red', 
             horizontalalignment='left', verticalalignment='top')
    
    plt.savefig(f"{outp_directory}/vmr_histogram.png", dpi=300)
    print("Saved histogram.")


def save_intermediaries(depth_vm_df, low_vm_df, high_vm_df, outp_directory):
    
    # Make output directory
    directory = os.path.join(outp_directory, 'vmr')
    common.mkdir(directory)
    
    # Save intermediate files regarding raw dataframe of vmr values
    low_vm_df = low_vm_df.sort_values(by="ratio").round(5)
    low_vm_df.to_csv(f"{directory}/low.vmr.samples", index=False, sep="\t")

    high_vm_df = high_vm_df.sort_values(by="ratio", ascending=False).round(5)
    high_vm_df.to_csv(f"{directory}/high.vmr.samples", index=False, sep="\t")

    # Save intermediate files regarding percentiles
    ratio_percentiles = [(p, np.percentile(depth_vm_df.ratio, p)) for p in range(5,100)]

    with open(f"{directory}/vmr.percentiles", "w") as file:
        file.writelines("percentile\tthreshold\tnum_samples_to_remove\n")
        for p,val in ratio_percentiles:
            # total number of samples that must be removed.
            remove = len(depth_vm_df.loc[depth_vm_df.ratio >= val])
            file.writelines(f"{p}\t{val}\t{remove}\n")

