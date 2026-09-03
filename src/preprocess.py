import allel
import gzip
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

class WindowedNpyData:
    """Read SNP features and labels for one chromosome window at a time."""

    def __init__(self, X_files, labels_files, chromosome_length, window_size):
        self.X_files = X_files
        self.labels_files = labels_files
        self.C = chromosome_length
        self.M = window_size
        self.W = self.C // self.M

        if len(X_files) != len(labels_files):
            raise ValueError("Feature and label files must have matching generations")

        self.n_samples = 0
        for X_path, labels_path in zip(X_files, labels_files):
            X = np.load(X_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
            try:
                if X.shape != labels.shape:
                    raise ValueError("Feature and label arrays must have matching shapes")
                if X.shape[1] != self.C:
                    raise ValueError("Input array chromosome length does not match metadata")
                self.n_samples += X.shape[0]
            finally:
                X._mmap.close()
                labels._mmap.close()

    def _bounds(self, window_index):
        if not 0 <= window_index < self.W:
            raise IndexError(f"Window index {window_index} is outside [0, {self.W})")
        start = window_index * self.M
        end = (window_index + 1) * self.M if window_index < self.W - 1 else self.C
        return start, end

    def load_feature_window(self, window_index, context):
        """Return the base-model feature slice, including reflected edge context."""
        start, end = self._bounds(window_index)
        left = max(0, start - context)
        right = min(self.C, end + context)
        left_pad = max(0, context - start)
        right_pad = max(0, end + context - self.C)
        windows = []

        for X_path in self.X_files:
            X = np.load(X_path, mmap_mode="r")
            pieces = []
            try:
                if left_pad:
                    pieces.append(np.flip(X[:, :left_pad], axis=1))
                pieces.append(X[:, left:right])
                if right_pad:
                    pieces.append(np.flip(X[:, self.C - right_pad:self.C], axis=1))
                window = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
                windows.append(np.array(window, dtype=np.int8, copy=True))
            finally:
                X._mmap.close()

        return np.concatenate(windows, axis=0)

    def load_label_window(self, window_index):
        """Aggregate SNP-level labels into the label for one base-model window."""
        start, end = self._bounds(window_index)
        labels = []
        for labels_path in self.labels_files:
            label_array = np.load(labels_path, mmap_mode="r")
            try:
                values, _ = stats.mode(label_array[:, start:end], axis=1)
                labels.append(np.asarray(values).reshape(-1).astype(np.int16, copy=False))
            finally:
                label_array._mmap.close()
        return np.concatenate(labels, axis=0)

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

def data_process(X, labels, window_size, missing=0.0):
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

    # simulates lacking of input
    if missing != 0:
        print("Simulating missing values...")
        X = simulate_missing_values(X, missing)

    X = np.array(X, dtype="int8")
    y = np.array(y, dtype="int16")

    return X, y

def dropout_row(data, missing_percent):
    num_drops = int(len(data)*missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)),size=num_drops,replace=False)
    data[drop_indices] = 2
    return data

def simulate_missing_values(data, missing_percent=0.0):
    if missing_percent == 0:
        return data
    return np.apply_along_axis(dropout_row, axis=1, arr=data, missing_percent=missing_percent)
