### Functions for tests (TODO: @wknd37: Move these to a tests folder)
import numpy as np
import pandas as pd


def get_window_level_local_ancestry_inference_from_msp_file(path_to_msp_file):
    msp_df = pd.read_csv(path_to_msp_file, sep="\t", skiprows=[0])
    inferred_indices = (np.array(msp_df)[:,6:].T).astype(int)
    with open(path_to_msp_file, "r") as f:
        index_to_population_map = np.array([p.split("=")[0] for p in f.readline().split()[2:]])
    inferred_ancestry_labels = index_to_population_map[inferred_indices]
    return inferred_ancestry_labels

def get_labels_from_sample_map_file(sample_map_file, dtype=None):
    true_global_ancestry_labels_single_haplotype_df = pd.read_csv(sample_map_file, sep="\t")
    true_global_ancestry_labels_single_haplotype = true_global_ancestry_labels_single_haplotype_df["Population"].to_numpy()
    if dtype is not None:
        true_global_ancestry_labels_single_haplotype = true_global_ancestry_labels_single_haplotype.astype(dtype)
    true_gloabl_ancestry_labels_single_dimension = np.repeat(true_global_ancestry_labels_single_haplotype, 2)
    true_gloabl_ancestry_labels = true_gloabl_ancestry_labels_single_dimension.reshape(-1, 1)
    return true_gloabl_ancestry_labels