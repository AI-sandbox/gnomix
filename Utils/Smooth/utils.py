import numpy as np
from scipy import stats

def slide_window(B, S, y=None, padding=None):
    """
    inputs
        - B (np.array: base probabilities of shape (N,W,A)
        - S (int): smoother window size 

    """

    N, W, A = B.shape

    padding = padding if padding is not None else S

    # pad it.
    pad_left = np.flip(B[:,0:padding,:],axis=1)
    pad_right = np.flip(B[:,-padding:,:],axis=1)
    B_padded = np.concatenate([pad_left,B,pad_right],axis=1)

    # window it.
    X_slide = np.zeros((N,W,A*S),dtype="float32")
    for ppl,dat in enumerate(B_padded):
        for win in range(X_slide.shape[1]):
            X_slide[ppl,win,:] = dat[win:win+S].ravel()

    # reshape
    X_slide = X_slide.reshape(-1,X_slide.shape[2])
    y_slide = None if y is None else y.reshape(-1)

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