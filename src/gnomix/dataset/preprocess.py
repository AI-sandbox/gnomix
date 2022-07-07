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
