import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import sys
import seaborn as sns


FIGSIZE = (20,1) # None
MARKERSIZE = 100
MAXCOLORS = 8

# Define a LAI Colormap
# CMAP = "tab20b"
LAI_PALETTE  = ["#A60303", "#3457BF", "#75BFAA", "#613673",  "#8DA6F2", "#AAAAAA", "#254F6B", "#D9414E" ]
CMAP = ListedColormap(LAI_PALETTE)

def sim_phase_error(n, swap=0.1, noise=0.02, blurr=0.2):
  # defining the true ancestry
  P = np.concatenate([[1]*(n//4), [0]*(n//4),[2]*(n//2)])
  M = np.concatenate([[2]*(n//8),[0]*(n//4),[1]*(n//4), [0]*(n//8), [0]*(n//4)])

  # Take copies before messing it up
  P_original = np.copy(P)
  M_original = np.copy(M)

  # swap
  for i in range(1,n):
    if np.random.rand() <= swap:
      P_copy = np.copy(P[i:])
      P[i:] = M[i:]
      M[i:] = P_copy

  # Phased, but includes errors
  P_mixed = np.copy(P)
  M_mixed = np.copy(M)

  # noise
  lam = 1
  col = 3
  for i in range(n):
    if np.random.rand() <= noise:
      M[i:(i+np.random.poisson(1))] = col
    if np.random.rand() <= noise:
      P[i:(i+np.random.poisson(1))] = col

  # blurr the swap (due to LAI smoother mistakes)
  for i in range(1,n):
    if P[i] != P[i-1]: # ancestry switch
      if np.random.rand() < blurr:
        P[i] = P[i-1]
    if M[i] != M[i-1]: # ancestry switch
      if np.random.rand() < blurr:
        M[i] = M[i-1]

  # M,P is the inferred ancestry
  print("original:")
  plot_haplo(M_original, P_original) 
  plt.show()
  print("Phase error:")
  plot_haplo(M_mixed,P_mixed)
  plt.show()
  print("post LAI")
  plot_haplo(M,P)
  plt.show()

  return M, P

def update_plot(i, data, scat):
  M_i = data[0,:,i]; P_i = data[1,:,i]
  color = np.concatenate([M_i,P_i])
  scat.set_array(color)
  return scat,

def animate_history(history):
  n = history.shape[1]
  numframes = history.shape[2]
  x = np.concatenate([range(n), range(n)])
  yim = ["M"]*n
  yip = ["P"]*n
  y = np.concatenate([yim, yip])
  M_0 = history[0,:,0]; P_0 = history[1,:,0]
  c = np.concatenate([M_0,P_0])

  fig = plt.figure(figsize=FIGSIZE)
  normalize = matplotlib.colors.Normalize(vmin=0, vmax=MAXCOLORS)
  scat = plt.scatter(x, y, c=c, marker="s", cmap=CMAP, norm=normalize, s=MARKERSIZE)
  plt.xticks([])
  ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes-1), fargs=(history, scat))
  plt.close()
  return ani

def idx2pos(idx, vcf):
    return vcf['variants/POS'][idx]

def pos2cM(pos, gen_map_df):
    end_pts = tuple(np.array(gen_map_df.pos_cm)[[0,-1]])
    f = interp1d(gen_map_df.pos, gen_map_df.pos_cm, fill_value=end_pts, bounds_error=False) 
    return f(pos)

def find_ref(haplo, ref1, ref2, start_idx):

    for i in range(start_idx, len(haplo)):
        if haplo[i] == ref1[i] and haplo[i] != ref2[i]:
            return ref1, 1
        if haplo[i] != ref1[i] and haplo[i] == ref2[i]:
            return ref2, 2
    return [], 0

def plot_haplo(M,P,pop_order=None, figsize=None):

    if figsize is None:
        figsize = FIGSIZE

    M,P = np.array((M,P), dtype=int)

    # data
    n = len(M)
    x = np.concatenate([range(n), range(n)])
    yim = ["M"]*n; yip = ["P"]*n;
    y = np.concatenate([yim, yip])
    c = np.concatenate([M, P])

    # plot it
    fig = plt.figure(figsize=figsize)
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=MAXCOLORS)
    scat = plt.scatter(x, y, c=c, marker="s", cmap=CMAP, norm=normalize, s=MARKERSIZE)
    if pop_order is not None:
        handles, labels = scat.legend_elements()
        plt.legend(handles, pop_order[np.unique(c)], loc="upper right", bbox_to_anchor=[1.1,1.3], title="Ancestry")

def get_ref_map(h, ref1, ref2):
    
    ref_map = []
    
    # find starting reference
    ref, tracker = find_ref(h, ref1, ref2, start_idx=0)

    # count the switches between
    n_switches = 0
    for i in range(len(h)):
        
        # if not matching somewhere on the way, switch reference
        if h[i] != ref[i]:
            if tracker == 1:
                ref = ref2
                tracker = 2
            elif tracker == 2:
                ref = ref1
                tracker = 1
            else:
                print("something went wrong")
                return 1
            
        # log the reference
        ref_map.append(tracker)
    
    return ref_map

def find_switches(M, P, M_true, P_true, verbose=True):
    M, P = np.copy(M), np.copy(P)
    switch_idxs = []
    for i in range(len(M)):
        if M[i] != M_true[i]:
            M_temp = np.copy(M)
            M[i:] = P[i:]
            P[i:] = M_temp[i:]
            switch_idxs.append(i)

    assert np.mean(M == M_true) and np.mean(P == P_true), "Phasing error count was not successful"
    
    if verbose:
        print("number of phasing errors:", len(switch_idxs))
        print("phasing errors/SNP length:", round(len(switch_idxs)/len(M),8))
        
    return np.array(switch_idxs)

def track_switch(M_track, P_track, i):
    M_track_temp = np.concatenate([np.copy(M_track[:i]), np.copy(P_track[i:])])
    P_track = np.concatenate([np.copy(P_track[:i]), np.copy(M_track[i:])])
    M_track = M_track_temp
    return M_track, P_track

def correct_phase_error(M_scrambled, P_scrambled, M_track, window_size):
    M_corrected, P_corrected = np.copy(M_scrambled), np.copy(P_scrambled)

    # correct the errors
    correction_idxs = (np.where(M_track[:-1] != M_track[1:])[0]+1)*window_size
    for i in correction_idxs:
        M_corrected_temp = np.copy(M_corrected)
        M_corrected[i:] = P_corrected[i:]
        P_corrected[i:] = M_corrected_temp[i:]

    return M_corrected, P_corrected

        
def measure(h, ref1, ref2, true_vcf, gen_map_df, offset=0, max_dist_idx=None, dist_freq_idx=50, n_pairs=10, verbose=True):
    """
    Function for measure phasing matching between two pairs of SNPs in a given haplotype
    
    Inputs:
    - h: haplotype or a part of a haplotype to be measured, possibly containing phasing errors
    - ref1, ref2: the true paternal or maternal haplotypes or parts of it
    - offset: index at which inputed parts start at
    - max_dist_idx, dist_freq_idx and n_pairs determine the samples to be taken randomly along the haplotype
    
    Outputs:
    - dist: np array containing the distances of the measured pairs
    - correct: np array containing boolean variables indicating if the pair is correctly phased
    such that correct[i] indicates if a pair with distance dist[i] is correctly phased relative to each other
    """
    
    dist, correct = [], []
    ref_map = get_ref_map(h, ref1, ref2)

    # max distance to measure (in idx space)
    n_snps = len(h)
    if max_dist_idx is None:
        max_dist_idx = (n_snps+offset)//5
    else:
        max_dist_idx = min(max_dist_idx, len(h))
    
    # sub-sample distances in idx space to save computation
    distances = np.arange(1,max_dist_idx,dist_freq_idx)
    
    # for each distance of the chosen distances
    for d_idx, d in enumerate(distances):

        # log process
        if verbose:
            if d_idx%10==0 or d_idx==len(distances)-1:
                sys.stdout.write("\r - distance %i/%i" % (d_idx,len(distances)-1))
            
        # sample pairs for evaluation (onnly sample first idx)
        pairs = np.random.choice(range(n_snps-d),n_pairs)
        
        # for each pair
        for i in pairs:
            
            # store matching
            correct.append(ref_map[i] == ref_map[i+d])
            
            # store distance
            pos1, pos2 = idx2pos(offset+i, true_vcf), idx2pos(offset+i+d, true_vcf)
            cM1, cM2 = pos2cM(pos1, gen_map_df), pos2cM(pos2, gen_map_df)
            dist.append(cM2 - cM1)

    if verbose:
        print()

    return dist, correct

def find_hetero_regions(M_original, P_original, plotshow=True, figsize=(20,1)):
    
    """ finds heterozygous ancestry regions """
    hetero = M_original != P_original
    regions_bounds = np.concatenate([[0], np.where(hetero[:-1] != hetero[1:])[0]+1, [len(hetero)]])
    regions = []
    for b, begin in enumerate(regions_bounds[:-1]):
        if hetero[begin]:
            regions.append(np.arange(regions_bounds[b], regions_bounds[b+1]))

    if plotshow:

        print("Haplotypes (true labels):")
        plot_haplo(M_original,P_original,figsize=figsize)
        plt.show()

        plt.figure(figsize=figsize)
        cols = np.concatenate([[r]*len(regions[r]) for r in range(len(regions))])
        plt.scatter(np.concatenate(regions), [1]*len(np.concatenate(regions)), c=cols, cmap="Accent")
        plt.scatter(regions_bounds, [1]*len(regions_bounds), color="red")
        plt.yticks([], [])
        plt.title("Heterozygous Regions")
        plt.show()
    
    return regions

def evaluate_corretion(haplo, M_true, P_true, regions, true_vcf, gen_map_df, window_size, max_len_cM=15, dist_freq_idx=50, verbose=True):
    
    np.random.seed(94305)

    haplo = np.copy(haplo)
    
    start_point_pos = idx2pos(0, true_vcf)
    start_point_cM  = pos2cM(start_point_pos, gen_map_df)
    end_point_pos = idx2pos(len(M_true)-1, true_vcf)
    end_point_cM  = pos2cM(end_point_pos, gen_map_df)

    # converting max length to idx space
    max_len_idx = int(len(M_true) * max_len_cM/(end_point_cM-start_point_cM))

    dist, correct = [], []
    for r in range(len(regions)):
        if verbose:
            print(" - region %i/%i" % (r+1,len(regions)))
        snp_idxs = np.arange(regions[r][0]*window_size, regions[r][-1]*window_size)

        # references for the region
        ref1  = np.copy(M_true[snp_idxs])
        ref2  = np.copy(P_true[snp_idxs])

        # haplotype of interest
        sub_haplo = haplo[snp_idxs]

        # measure correction
        dist_region, correct_region = measure(sub_haplo, ref1, ref2, true_vcf, gen_map_df,
                                              offset=snp_idxs[0], max_dist_idx=max_len_idx,
                                               dist_freq_idx=dist_freq_idx, verbose=verbose)

        dist.append(dist_region)
        correct.append(correct_region)

        print("region ", regions[r][0], ":", np.mean(correct_region))

    dist, correct = np.concatenate(dist), np.concatenate(correct)

    return dist, correct

def plot_matching_vs_dist(dist_scram, corr_scram, dist_corr_xgmix, corr_corr_xgmix, dist_corr_rfmix=None, corr_corr_rfmix=None,
                            n_bins=500, max_len_cM=15, figsize=(12,8), title="", lab1="XGMix", lab2="RFMix"):

    dist = np.concatenate([dist_scram, dist_corr_xgmix, dist_corr_rfmix])
    correct = np.concatenate([corr_scram, corr_corr_xgmix, corr_corr_rfmix])
    haplo = np.concatenate([ ["scrambled"]*len(dist_scram), [lab1]*len(dist_corr_xgmix), [lab2]*len(dist_corr_rfmix) ])
    bins = np.arange(0, max_len_cM, max_len_cM/n_bins)
    phasing_df = pd.DataFrame({"dist_cM": dist, "correct": correct, "haplo": haplo})
    phasing_df["dist_group_cM"] = pd.cut(x=phasing_df['dist_cM'], bins=bins)
    phasing_df_grouped = phasing_df.groupby(['dist_group_cM', 'haplo'])
    plot_df = phasing_df_grouped.mean()
    
    fig, ax = plt.subplots(1,2,figsize=figsize)

    ax[0].hist([dist_scram, dist_corr_xgmix, dist_corr_rfmix], label=["scrambled", lab1, lab2])
    ax[0].legend()
    ax[0].set_title("Sample distribution of distances")
    ax[0].set_xlabel("Distance between pairs (cM)")

    ax[1].plot(plot_df['dist_cM'].unstack()["scrambled"], plot_df['correct'].unstack()["scrambled"],label="scrambled")
    ax[1].plot(plot_df['dist_cM'].unstack()[lab1], plot_df['correct'].unstack()[lab1],label=lab1)
    ax[1].plot(plot_df['dist_cM'].unstack()[lab2], plot_df['correct'].unstack()[lab2],label=lab2)
    ax[1].plot([0,max_len_cM],[0.5, 0.5], color="black", alpha=0.5)
    ax[1].legend()
    ax[1].set_xlim(0,max_len_cM)
    ax[1].set_title("Performance")
    ax[1].set_xlabel("Distance between pairs (cM)")
    ax[1].set_ylabel("Fraction of SNPs correctly phased")
    
    fig.suptitle(title, fontsize=16)

    plt.show()


def plot_matching_vs_dist(dist_scram, corr_scram, dists_corr, corrs_corr, labs, n_bins=500, max_len_cM=15, figsize=(14,8), title="", alpha=1):

    haplo_len = len(dist_scram)

    dist = [dist_scram]
    [dist.append(d) for d in dists_corr]
    dist = np.array(dist).reshape(-1)

    correct = [corr_scram]
    [correct.append(c) for c in corrs_corr]
    correct = np.array(correct).reshape(-1)

    plot_labs = [["scrambled"]*haplo_len]
    [plot_labs.append([l]*haplo_len) for l in labs]
    plot_labs = np.array(plot_labs).reshape(-1)

    bins = np.arange(0, max_len_cM, max_len_cM/n_bins)
    phasing_df = pd.DataFrame({"dist_cM": dist, "correct": correct, "label": plot_labs})
    phasing_df["dist_group_cM"] = pd.cut(x=phasing_df['dist_cM'], bins=bins)
    phasing_df_grouped = phasing_df.groupby(['dist_group_cM', 'label'])
    plot_df = phasing_df_grouped.mean()
    
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(plot_df['dist_cM'].unstack()["scrambled"], plot_df['correct'].unstack()["scrambled"],label="scrambled", alpha=alpha)
    for lab in labs:
        ax.plot(plot_df['dist_cM'].unstack()[lab], plot_df['correct'].unstack()[lab],label=lab,  alpha=alpha)
    ax.plot([0,max_len_cM],[0.5, 0.5], color="black", alpha=0.5)
    ax.legend()
    ax.set_xlim(0,max_len_cM)
    ax.set_ylim(0,1)
    ax.set_title("Performance")
    ax.set_xlabel("Distance between pairs (cM)")
    ax.set_ylabel("Fraction of SNPs correctly phased")
    
    fig.suptitle(title, fontsize=16)

    plt.show()

def plot_matching_vs_dist3(dists, corrs, labs, colors=None, n_bins=500, max_len_cM=15, figsize=(14,8), title="", alpha=1, fz=None, figname=None):

    sns.set_style("white")
    mpl.rcParams['xtick.labelsize'] = fz 
    mpl.rcParams['ytick.labelsize'] = fz 
    mpl.rcParams['legend.fontsize'] = fz

    haplo_len = len(dists[0])
    if colors is None:
        colors = [None]*haplo_len

    dist = np.array(dists).reshape(-1)
    correct = np.array(corrs).reshape(-1)

    plot_labs = np.repeat(np.array(labs), haplo_len)

    bins = np.arange(0, max_len_cM, max_len_cM/n_bins)
    phasing_df = pd.DataFrame({"dist_cM": dist, "correct": correct, "label": plot_labs})
    phasing_df["dist_group_cM"] = pd.cut(x=phasing_df['dist_cM'], bins=bins)
    phasing_df_grouped = phasing_df.groupby(['dist_group_cM', 'label'])
    plot_df = phasing_df_grouped.mean()
    
    fig, ax = plt.subplots(figsize=figsize)

    for i, lab in enumerate(labs):
        ax.plot(plot_df['dist_cM'].unstack()[lab], plot_df['correct'].unstack()[lab],label=lab, color=colors[i], alpha=alpha)
    ax.plot([0,max_len_cM],[0.5, 0.5], color="black", alpha=0.25)
    ax.legend()
    ax.set_xlim(0,max_len_cM)
    ax.set_ylim(0,1)
    ax.set_title(title)
    ax.set_xlabel("Distance between pairs [cM]", fontsize=fz)
    ax.set_ylabel("Fraction of SNP pairs correctly phased", fontsize=fz)
    
    fig.suptitle(title, fontsize=16)

    if figname is not None:
        plt.savefig(figname+'.svg', format='svg')

    plt.show()