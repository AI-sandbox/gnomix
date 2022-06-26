import allel
import numpy as np
import pandas as pd
from scipy import stats

"""
Pre-processing pipeline.
Functions to load data, generate labels based on window size.
"""
def load_np_data(files, verb=False):

    data = []
    for f in files:
        if verb:
            print("Reading " + f + " ...")
        data.append(np.load(f).astype(np.int16))

    data = np.concatenate(data,axis=0)
    return data

def vcf2npy(vcf_file):
    vcf_data = allel.read_vcf(vcf_file)
    chm_len, nout, _ = vcf_data["calldata/GT"].shape
    mat_vcf_2d = vcf_data["calldata/GT"].reshape(chm_len,nout*2).T
    return mat_vcf_2d.astype('int16')

def map2npy(map_file, shape, pop_order):
    sample_map = pd.read_csv(map_file, sep="\t", header=None)
    sample_map.columns = ["sample", "ancestry"]
    y = np.zeros(shape, dtype='int16')
    for i, a in enumerate(sample_map["ancestry"]):
        a_numeric = np.where(a==pop_order)[0][0]
        y[2*i:2*i+2] = a_numeric
    return y

def window_labels(labels, window_size):
    """
    Takes in labels of shape (N, C), aggregates labels and 
    returns window shaped labels of shape (N, W) where
    W = ceil(C/window_size)
    """

    # # Split in windows and make the last one contain the remainder
    N, C = labels.shape
    W = int(np.ceil(C / window_size))
    windowed_labels = np.zeros((N, W))
    for w in range(W):
        windowed_labels[:, w] = stats.mode(labels[:, w*window_size:(w+1)*window_size], axis=1)[0].squeeze()
    return windowed_labels

def dropout_row(data, missing_percent):
    num_drops = int(len(data)*missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)),size=num_drops,replace=False)
    data[drop_indices] = 2
    return data

def simulate_missing_values(data, missing_percent=0.0):
    if missing_percent == 0:
        return data
    return np.apply_along_axis(dropout_row, axis=1, arr=data, missing_percent=missing_percent)