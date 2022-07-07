import allel
import numpy as np
import pandas as pd
from scipy import stats

"""
Pre-processing: Move all these to BaseModels TODO: @arvind0422
"""
def load_np_data(files, verb=False):

    data = []
    for f in files:
        if verb:
            print("Reading " + f + " ...")
        data.append(np.load(f).astype(np.int16))

    data = np.concatenate(data,axis=0)
    return data

def window_reshape(data, win_size):
    """
    Takes in data of shape (N, chm_len), aggregates labels and 
    returns window shaped data of shape (N, chm_len//window_size)
    """

    # Split in windows and make the last one contain the remainder
    chm_len = data.shape[1]
    drop_last_idx = chm_len//win_size*win_size - win_size
    window_data = data[:,0:drop_last_idx]
    rem = data[:,drop_last_idx:]

    # reshape accordingly
    N, C = window_data.shape
    num_winds = C//win_size
    window_data =  window_data.reshape(N,num_winds,win_size)

    # attach thet remainder
    window_data = stats.mode(window_data, axis=2)[0].squeeze() 
    rem_label = stats.mode(rem, axis=1)[0].squeeze()
    window_data = np.concatenate((window_data,rem_label[:,np.newaxis]),axis=1)

    return window_data

def data_process(X, labels, window_size):
    """ 
    Takes in 2 numpy arrays:
        - X is of shape (N, chm_len)
        - labels is of shape (N, chm_len)

    And returns 2 processed numpy arrays:
        - X is of shape (N, chm_len)
        - labels is of shape (N, chm_len//window_size)
    """

    # Reshape labels into windows 
    y = window_reshape(labels, window_size)

    # dtypes
    X = np.array(X, dtype="int8")
    y = np.array(y, dtype="int16")

    return X, y

### Deprecated for Version 1.0

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

def dropout_row(data, missing_percent):
    num_drops = int(len(data)*missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)),size=num_drops,replace=False)
    data[drop_indices] = 2
    return data

def simulate_missing_values(data, missing_percent=0.0):
    if missing_percent == 0:
        return data
    return np.apply_along_axis(dropout_row, axis=1, arr=data, missing_percent=missing_percent)