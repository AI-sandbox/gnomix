import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def CM(y, y_pred, labels, save_path=None, verbose=True):
    cm = confusion_matrix(y, y_pred)
    if verbose:
        print("Confusion matrix for validation data:")
        print(cm)
    if save_path is not None:
        n_digits = int(np.ceil(np.log10(np.max(cm))))
        str_fmt = '%-'+str(n_digits)+'.0f'
        np.savetxt(save_path+"/confusion_matrix.txt", cm, fmt=str_fmt)
        cm_figure = plot_cm(cm, normalize=True, labels=labels)
        cm_figure.figure.savefig(save_path+"/confusion_matrix_normalized.png")
        if verbose:
            print("Confusion matrix saved in", save_path)
    return cm

FIGSIZE = None
MARKERSIZE = 100
MAXCOLORS = 8

# Define a LAI Colormap
# CMAP = "tab20b"
# LAI_PALETTE  = ["#A60303", "#8DA6F2", "#254F6B", "#613673",  "#75BFAA", "#AAAAAA", "#3457BF", "#D9414E" ]
LAI_PALETTE  = ["#A60303", "#3457BF", "#75BFAA", "#613673",  "#8DA6F2", "#AAAAAA", "#254F6B", "#D9414E" ]
CMAP = ListedColormap(LAI_PALETTE)

def visualize_palette(palette=None):
    if palette is None:
        palette = LAI_PALETTE
    nn = 100
    for i, col in enumerate(palette):
        plt.plot(np.arange(nn), np.repeat(i,nn), color=col, linewidth=20)
    plt.show()

def haplo_tile_plot(haplos, pop_order=None, bbox_to_anchor=[1.2,1.0]):
    """
    Tile plot for visualizing haplotypes.
        - haplos: array of haplotypes
        - pop_order: order of ancestry for figure legend
    """
    ANI_FIGSIZE = (10,2)

    haplos = np.array(haplos, dtype=int)
    n_anc = len(np.unique(haplos))

    n_haplo, n_wind = haplos.shape
    XX = np.array([range(n_wind) for _ in range(n_haplo)]).reshape(-1)
    YY = np.array([np.repeat(i, n_wind) for i in range(n_haplo)]).reshape(-1)
    CC = haplos.reshape(-1)

    fig, ax = plt.subplots(figsize=ANI_FIGSIZE, constrained_layout=True)
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=n_anc)
    scat = ax.scatter(XX, YY, c = CC, marker="s", cmap=CMAP, norm=normalize, s=MARKERSIZE)
    y_ticks_new = ["P" if i%2 else "M" for i in range(n_haplo)]
    y_ticks_new = [tick+"'" if t%4>1 else tick for t, tick in enumerate(y_ticks_new)]
    plt.setp(ax, yticks=range(n_haplo), yticklabels=y_ticks_new)

    if pop_order is not None:
        handles, labels = scat.legend_elements()
        plt.legend(handles, pop_order[np.unique(CC)], loc="upper right", bbox_to_anchor=bbox_to_anchor, title="Ancestry")

    return fig, ax

def plot_cm(cm, normalize=True, labels=None, figsize=(12,10)):
    plt.figure(figsize=figsize)
    
    # normalize w.r.t. number of samples from class
    if normalize:
        cm = cm/np.sum(cm, axis=1)
        cm = np.nan_to_num(cm, copy=False, nan=0.0)
        
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    sns.set(font_scale=1.4) # for label size
    if labels is None:
        fig = sns.heatmap(df_cm, annot=False, annot_kws={"size": 16}) # font size
    else:
        fig = sns.heatmap(df_cm, xticklabels=labels, yticklabels=labels,
                   annot=False, annot_kws={"size": 16}) # font size
    
    plt.show()
    return fig

def plot_chm(sample_id, msp_df, rm_img=False, img_name="chm_img"):

    """
    Wrapper function for plotting with Tagore. 
    Requires an msp dataframe and a sample_id of which to plot the chromosome.
    """
    
    # defining a color palette
    palette = sns.color_palette("colorblind").as_hex()
    
    # get the base of the tagore style dataframe
    nrows = msp_df.shape[0]
    default_params = pd.DataFrame({"feature": [0]*nrows, "size": [1]*nrows})
    tagore_base = msp_df[["#chm", "spos", "epos"]].join(default_params)
    tagore_base.columns = ["chm", "start", "stop", "feature", "size"]
    
    # adding data from the individual with that sample_id
    colors0 = [palette[i] for i in np.array(msp_df[sample_id+".0"])]
    colors1 = [palette[i] for i in np.array(msp_df[sample_id+".1"])]
    tagore0 = tagore_base.join(pd.DataFrame({"color": colors0, "chrCopy": 1}))
    tagore1 = tagore_base.join(pd.DataFrame({"color": colors1, "chrCopy": 2}))
    tagore_df = pd.concat([tagore0, tagore1])

    # plot the results
    tagore_df_fname = "./tagore.tsv"
    tagore_df.to_csv(tagore_df_fname, sep="\t", index=False, 
                     header = ['#chr','spos','epos','feature','size','color','ChrCopy'])
    os.system("tagore --i " + tagore_df_fname + " -p "+ img_name +  " --build hg37 -f")
    if rm_img:
        os.system("rm " + tagore_df_fname)
