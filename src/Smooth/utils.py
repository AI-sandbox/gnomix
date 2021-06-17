import numpy as np
from scipy import stats

def slide_window(B, S, y=None):
    """
    inputs
        - B (np.array: base probabilities of shape (N,W,A)
        - S (int): smoother window size 

    """
    N, W, A = B.shape

    # pad it.
    pad = (S+1)//2
    pad_left  = np.flip(B[:,0:pad,:],axis=1)
    pad_right = np.flip(B[:,-pad:,:],axis=1)
    B_padded = np.concatenate([pad_left,B,pad_right],axis=1)

    # window it.
    X_slide = np.zeros((N,W,A*S),dtype="float32")
    for ppl, dat in enumerate(B_padded):
        for w in range(W):
            X_slide[ppl,w,:] = dat[w:w+S].ravel()

    # reshape
    X_slide = X_slide.reshape(N*W,A*S)
    y_slide = None if y is None else y.reshape(N*W)

    return X_slide, y_slide

def mode(arr):
    mode = stats.mode(arr)[0][0]
    if mode == -stats.mode(-arr)[0][0]:
        return mode # if mode is unambiguous
    else:
        return arr[len(arr)//2] # else return the center (default value)

def mode_filter(pred, size):
    if not size:
        return pred # if size is 0 or None
    pred_out = np.copy(pred)
    ends = size//2
    for i in range(len(pred))[ends:-ends]:
        pred_out[i] = mode(pred[i-ends:i+ends+1])
    
    return pred_out