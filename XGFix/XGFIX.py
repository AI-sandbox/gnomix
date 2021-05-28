import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pickle
import seaborn as sns
import sys
from time import time

try:
    # if calling from XGMix
    from src.utils import read_vcf, vcf_to_npy, fb2proba, npy_to_vcf
    from XGFix.phasing import *
    from XGFix.simple_switch import simple_switch
except ModuleNotFoundError:
    # if calling from XGFix
    from utils import read_vcf, vcf_to_npy, fb2proba, npy_to_vcf
    from phasing import *
    from simple_switch import simple_switch

def mask_base_prob(base_prob, d=0):
    """
    given base out with shape [H, W, A] where 
        - H is number of Haplotypes
        - W is number of Windows
        - A is number of Ancestry
    filter out all windows that are not more than d windows away from center
    """
    base_prob = np.array(base_prob)
    H, W, A = base_prob.shape
    c = int((W-1)/2)
    masked = np.copy(base_prob)
    masked[:,np.arange(c-d),:] = 0
    masked[:,np.arange(c+d+1,W),:] = 0
    return masked

def check(Y_m, Y_p, w, base, check_criterion):

    check = False

    if check_criterion == "all":
        check = True
    elif check_criterion == "disc_smooth":
        check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
    elif check_criterion == "disc_base":
        base_Y_ms, base_Y_ps = np.argmax(base[:,w-1:w +1,:],axis=2)
        check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
    elif check_criterion == "disc_either":
        base_Y_ms, base_Y_ps = np.argmax(base[:,w-1:w+1,:],axis=2)
        base_check = base_Y_ms[0] != base_Y_ms[1] or base_Y_ps[0] != base_Y_ps[1]
        smooth_check = Y_m[w] != Y_m[w-1] or Y_p[w] != Y_p[w-1]
        check = base_check or smooth_check
    else:
        print("Warning: check criteration not recognized. Checking all windows")
        check = True

    return check

def base_to_smooth_data(base_prob, sws, pad_size=None, labels=None):

    base_prob = np.copy(base_prob)
    N, W, A = base_prob.shape

    if pad_size is None:
        pad_size = (1+sws)//2

    # pad it.
    pad_left = np.flip(base_prob[:,0:pad_size,:],axis=1)
    pad_right = np.flip(base_prob[:,-pad_size:,:],axis=1)
    base_prob_padded = np.concatenate([pad_left,base_prob,pad_right],axis=1)

    # window it.
    windowed_data = np.zeros((N,W,A*sws),dtype="float32")
    for ppl,dat in enumerate(base_prob_padded):
        for win in range(windowed_data.shape[1]):
            windowed_data[ppl,win,:] = dat[win:win+sws].ravel()

    # reshape
    windowed_data = windowed_data.reshape(-1,windowed_data.shape[2])
    windowed_labels = None if labels is None else labels.reshape(-1)

    return windowed_data, windowed_labels

def load_smoother(path_to_smoother, verbose=True):
    if verbose:
        print("Loading smoother...")
    if path_to_smoother[-3:]==".gz":
        with gzip.open(path_to_smoother, 'rb') as unzipped:
            smoother = pickle.load(unzipped)
    else:
        smoother = pickle.load(open(path_to_smoother,"rb"))

    return smoother

def XGFix(M, P, base_prob, smoother, max_it=50, non_lin_s=0, check_criterion="disc_smooth", max_center_offset=0, prob_comp="max", d=None, prior_switch_prob = 0.5,
            naive_switch=None, end_naive_switch=None, padding=True, verbose=False):

    if verbose:
        # print configs
        print("max center offset:", max_center_offset)
        print("non_lin_s:", non_lin_s)
        print("Mask:", d)
        print("including naive switch:", naive_switch)
        print("including end naive switch:", end_naive_switch)
        print("prior switch prob:", prior_switch_prob)
        print("check criterion:", check_criterion)
        print("probability comparison:", prob_comp)
        print("padding:", padding)

    N, W, A = base_prob.shape 
    sws = len(smoother.feature_importances_)//A # smoother size
    window_size = len(M)//W # window size

    # initial position
    X_m, X_p = np.copy([M,P]).astype(int)

    # inferred labels of initial position
    smooth_data, _ = base_to_smooth_data(base_prob, sws=sws) 
    Y_m, Y_p = smoother.predict(smooth_data).reshape(2,W)

    # define windows to iterate through
    centers = (np.arange(W-sws+1)+(sws-1)/2).astype(int)
    iter_windows = np.arange(1,W) if padding else centers

    # Track convergence and progression
    X_m_its = [] # monitor convergence
    XGFix_tracker = (np.zeros_like(Y_m), np.ones_like(Y_p)) 
    history = np.array([Y_m, Y_p])

    # Fix
    st = time()
    for it in range(max_it):

        if verbose:
            sys.stdout.write("\riteration %i/%i" % (it+1, max_it))

        if naive_switch:
            # Naive switch: heuristic to catch obvious errors and save computations
            _, _, M_track, _, _ = simple_switch(Y_m,Y_p,slack=naive_switch,cont=False,verbose=False,animate=False)
            X_m, X_p = correct_phase_error(X_m, X_p, M_track, window_size)
            base_prob = np.array(correct_phase_error(base_prob[0], base_prob[1], M_track, window_size))
            smooth_data, _ = base_to_smooth_data(base_prob, sws=sws) 
            Y_m, Y_p = smoother.predict(smooth_data).reshape(2,W)
            history = np.dstack([history, [Y_m, Y_p]])

        # Stop if converged
        if np.any([np.all(X_m == X_m_it) for X_m_it in X_m_its]):
            if verbose:
                print(); print("converged, stopping..", end="")
            break
        else:
            X_m_its.append(X_m)

        # Iterate through windows
        for w in iter_windows:

            # Heuristic to save computation, only check if there's a nuance
            if check(Y_m, Y_p, w, base_prob, check_criterion):

                # Different permutations depending on window position
                if w in centers:
                    center = w
                    max_center_offset_w, non_lin_s_w = max_center_offset, non_lin_s
                else:
                    center = centers[0] if w < centers[0] else centers[-1]
                    max_center_offset_w, non_lin_s_w = 0, 0
                
                # defining scope
                scope_idxs = center + np.arange(sws) - int((sws-1)/2)

                # indices of pair-wise permutations
                switch_idxs = []
                switch_idxs += [np.array([j]) for j in range(w-max_center_offset_w, w+max_center_offset_w+1)]  # single switches: xxxxxxoooooo
                switch_idxs += [np.array([w-j,w]) for j in range(1,non_lin_s_w)] # double switches left of center: xxxoocxxx
                switch_idxs += [np.array([w,w+j+1]) for j in range(non_lin_s_w)] # double switches right of center: xxxcooxxx

                # init collection of permutations and add the original
                mps = [] 
                m_orig, p_orig = np.copy(base_prob[:,scope_idxs,:])
                mps.append(m_orig); mps.append(p_orig) 
                
                # adding more permutations
                for switch_idx in switch_idxs:
                    switch_idx = np.concatenate([[scope_idxs[0]], switch_idx.reshape(-1), [scope_idxs[-1]+1]])
                    m, p = [], []
                    for s in range(len(switch_idx)-1):
                        m_s, p_s = base_prob[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    mps.append(m); mps.append(p);

                # get 2D probabilities for permutations
                mps = np.array(mps) if d is None else mask_base_prob(mps, d=d)
                outs = smoother.predict_proba( mps.reshape(len(mps),-1) ).reshape(-1,2,A)

                # map permutation probabilities to a scalar (R^2 -> R) for comparison
                if prob_comp=="prod":
                    probs = np.prod(np.max(outs,axis=2),axis=1)
                if prob_comp=="max":
                    probs = np.max(np.max(outs,axis=2),axis=1)

                # select the most probable one
                original_prob, switch_probs = probs[0], probs[1:]
                best_switch_prob = np.max(switch_probs)
                best_switch = switch_idxs[np.argmax(switch_probs)].reshape(-1)

                # if more likely than the original, replace the output of the base
                if best_switch_prob*prior_switch_prob > original_prob*(1-prior_switch_prob):
                    switched = True
                    m, p = [], []
                    switch_idx = np.concatenate([[0], best_switch, [W]])
                    for s in range(len(switch_idx)-1):
                        m_s, p_s = base_prob[:,np.arange(switch_idx[s],switch_idx[s+1]),:]
                        if s%2:
                            m_s, p_s = p_s, m_s
                        m.append(m_s); p.append(p_s)
                    m, p = np.copy(np.concatenate(m,axis=0)), np.copy(np.concatenate(p,axis=0))
                    base_prob = np.copy(np.array([m,p])) 

                    # track the change
                    for switch in best_switch:
                        M_track, P_track = track_switch(np.zeros_like(Y_m), np.ones_like(Y_p), switch)
                        XGFix_tracker = track_switch(XGFix_tracker[0], XGFix_tracker[1], switch)

                    # correct inferred error on SNP level and re-label
                    X_m, X_p = correct_phase_error(X_m, X_p, M_track, window_size)
                    smooth_data, _ = base_to_smooth_data(base_prob, sws=sws)  
                    Y_m, Y_p = smoother.predict(smooth_data).reshape(2,W)
                    history = np.dstack([history, [Y_m, Y_p]])
    
    if naive_switch:
        end_naive_switch = naive_switch
    if end_naive_switch:
        _, _, M_track, _, _ = simple_switch(Y_m,Y_p,slack=end_naive_switch,cont=False,verbose=False,animate=False)
        X_m, X_p = correct_phase_error(X_m, X_p, M_track, window_size)
        base_prob = np.array(correct_phase_error(base_prob[0], base_prob[1], M_track, window_size))
        smooth_data, _ = base_to_smooth_data(base_prob, sws=sws) 
        Y_m, Y_p = smoother.predict(smooth_data).reshape(2,W)
        history = np.dstack([history, [Y_m, Y_p]])

    history = np.dstack([history, [Y_m, Y_p]])
    
    if verbose:
        print(); print("runtime:", np.round(time()-st))
    
    return X_m, X_p, Y_m, Y_p, history, XGFix_tracker

def main(query_file, fb_file, smoother_file, output_basename, chm, n_windows=None, verbose=False):

    # Read X from query file
    # TODO: does the npy need some modification? 
    query_vcf_data = read_vcf(query_file, chm=chm, fields="*")
    X = vcf_to_npy(query_vcf_data)
    H, C = X.shape
    N = H//2

    # Load a smoother
    S = load_smoother(smoother_file)

    # Read base_prob from fb
    base_prob = fb2proba(fb_file, n_wind=n_windows)
    print(base_prob.shape)
    H_, W, A = base_prob.shape
    base_prob = base_prob.reshape(H//2,2,W,A)
    assert H == H_, "Number of haplotypes from base probabilities must match number of query haplotypes"

    # Phase
    X_phased = np.zeros((N,2,C), dtype=int)
    Y_phased = np.zeros((N,2,W), dtype=int)
    for i, X_i in enumerate(X.reshape(N,2,C)):
        sys.stdout.write("\rPhasing individual %i/%i" % (i+1, N))
        X_m, X_p = np.copy(X_i)
        X_m, X_p, Y_m, Y_p, history, XGFix_tracker = XGFix(X_m, X_p, base_prob=base_prob[i], smoother=S,
                                                           check_criterion="disc_base", verbose=True)
        X_phased[i] = np.copy(np.array((X_m,X_p)))
        Y_phased[i] = np.copy(np.array((Y_m,Y_p)))
    X_phased = X_phased.reshape(H,C)
    print()

    # Write results
    if verbose:
        print("Writing phased SNPs to disc...")
    npy_to_vcf(query_vcf_data, X_phased, output_basename)

    return

if __name__ == "__main__":

    # TODO: Ask for citation

    # Infer mode from number of arguments (could be extended to connect with XGMix)
    mode = None
    if len(sys.argv) == 6:
        mode = "" 

    # Usage message
    if mode is None:
        if len(sys.argv) > 1:
            print("Error: Incorrect number of arguments.")
        print("Usage:")
        print("   $ python3 XGFIX.py <query_file> <fb_file> <smoother_file> <output_basename> <chm>")
        sys.exit(0)

    _, query_file, fb_file, smoother_file, output_basename, chm = sys.argv

    main(query_file, fb_file, smoother_file, output_basename, chm, verbose=True)
    # main(*sys.argv)
    