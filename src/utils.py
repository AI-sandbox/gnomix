import allel
import gzip
import numpy as np
import os
import pandas as pd
import random
from scipy.interpolate import interp1d
import string
import sys
from time import time
import pickle

def save_dict(D, path):
    with open(path, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def get_num_outs(sample_map_paths, r_admixed=1.0):
    # r_admixed: generated r_admixed * num-founders for each set
    # TODO: cap train2 lengths to a pre-defined value.
    num_outs = []
    for path in sample_map_paths:
        with open(path,"r") as f:
            length = len(f.readlines()) # how many founders.
            num_outs.append(int(length *r_admixed))
    return num_outs

def run_shell_cmd(cmd, verb=True):
    if verb:
        print("Running:", cmd)
    rval = os.system(cmd)
    if rval != 0:
        signal = rval & 0xFF
        exit_code = rval >> 8
        if signal != 0:
            sys.stderr.write("\nCommand %s exits with signal %d\n\n" % (cmd, signal))
            sys.exit(signal)
        sys.stderr.write("\nCommand %s failed with return code %d\n\n" % (cmd, exit_code))
        sys.exit(exit_code)

def join_paths(p1,p2="",verb=True):
    path = os.path.join(p1,p2)
    if not os.path.exists(path):
        os.makedirs(path)
        if verb:
            print("path created:", path)
    return path

def read_vcf(vcf_file, chm=None, fields=None, verbose=False):
    """
    Wrapper function for reading vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unecessary
    """
    if fields is None:
        fields = ['variants/CHROM', 'variants/POS', 'calldata/GT', 'variants/REF', 'samples']

    if vcf_file[-3:]==".gz":
        with gzip.open(vcf_file, 'rb') as vcf:
            data = allel.read_vcf(vcf,  region=chm, fields=fields)
    else: 
        data = allel.read_vcf(vcf_file, region=chm, fields=fields)

    if verbose:    
        chmlen, n, _ = data["calldata/GT"].shape
        print("File read:", chmlen, "SNPs for", n, "individuals")

    return data

def snp_intersection(pos1, pos2, verbose=False):
    """
    Finds interception of snps given two arrays of snp position 
    in O(max[size(pos1), size(pos2)])
    """

    assert len(pos2) != 0, "No SNPs of specified chromosome found in query file."
    
    ind_dict_1 = dict((p,i) for i,p in enumerate(pos1)) # O(n1)
    ind_dict_2 = dict((p,i) for i,p in enumerate(pos2)) # O(n2)
    intersection = set(pos1) & set(pos2)                # O(min[n1, n2])
    assert len(intersection) != 0, "No matching SNPs between model and query file."
    idx12 = [ (ind_dict_1[p], ind_dict_2[p]) for p in intersection ] # O(min[n1, n2])
    idx1, idx2 = np.array(idx12).T

    if verbose:
        print("- Number of SNPs from model:", len(pos1))
        print("- Number of SNPs from file:",  len(pos2))
        print("- Number of intersecting SNPs:", len(intersection))
        intersect_percentage = round(len(intersection)/len(pos1),4)*100
        print("- Percentage of model SNPs covered by query file: ",
              intersect_percentage, "%", sep="")

    return idx1, idx2


def vcf_to_npy(vcf_data, snp_pos_fmt=None, snp_ref_fmt=None, miss_fill=2, return_idx=False, verbose=True):
    """
    Converts vcf file to numpy matrix. 
    If SNP position format is specified, then comply with that format by filling in values 
    of missing positions and ignoring additional positions.
    If SNP reference variant format is specified, then comply with that format by swapping where 
    inconsistent reference variants.
    Inputs
        - vcf_data: already loaded data from a vcf file
        - snp_pos_fmt: desired SNP position format
        - snp_ref_fmt: desired reference variant format
        - miss_fill: value to fill in where there are missing snps
    Outputs
        - npy matrix on standard format
    """

    # reshape binary represntation into 2D np array 
    data = vcf_data["calldata/GT"]
    chm_len, n_ind, _ = data.shape
    data = data.reshape(chm_len,n_ind*2).T
    mat_vcf_2d = data
    vcf_idx, fmt_idx = np.arange(n_ind*2), np.arange(n_ind*2)

    if snp_pos_fmt is not None:
        # matching SNP positions with standard format (finding intersection)
        vcf_pos = vcf_data['variants/POS']
        fmt_idx, vcf_idx = snp_intersection(snp_pos_fmt, vcf_pos, verbose=verbose)
        # only use intersection of variants (fill in missing values)
        fill = np.full((n_ind*2, len(snp_pos_fmt)), miss_fill)
        fill[:,fmt_idx] = data[:,vcf_idx]
        mat_vcf_2d = fill

    if snp_ref_fmt is not None:
        # adjust binary matrix to match model format
        # - find inconsistent references
        vcf_ref = vcf_data['variants/REF']
        swap = vcf_ref[vcf_idx] != snp_ref_fmt[fmt_idx] # where to swap w.r.t. intersection
        if swap.any() and verbose:
            swap_n = sum(swap)
            swap_p = round(np.mean(swap)*100,4)
            print("- Found ", swap_n, " (", swap_p, "%) different reference variants. Adjusting...", sep="")
        # - swapping 0s and 1s where inconsistent
        fmt_swap_idx = np.array(fmt_idx)[swap]  # swap-index at model format
        mat_vcf_2d[:,fmt_swap_idx] = (mat_vcf_2d[:,fmt_swap_idx]-1)*(-1)

    # make sure all missing values are encoded as required
    missing_mask = np.logical_and(mat_vcf_2d != 0, mat_vcf_2d != 1)
    mat_vcf_2d[missing_mask] = miss_fill

    # return npy matrix
    if return_idx:
        return mat_vcf_2d, vcf_idx, fmt_idx

    return mat_vcf_2d

def read_genetic_map(genetic_map_path, chm=None):

    gen_map_df = pd.read_csv(genetic_map_path, sep="\t", comment="#", header=None, dtype="str")
    gen_map_df.columns = ["chm", "pos", "pos_cm"]
    gen_map_df = gen_map_df.astype({'chm': str, 'pos': np.int64, 'pos_cm': np.float64})

    if chm is not None:
        if len(gen_map_df[gen_map_df.chm == chm]) == 0:
            gen_map_df = gen_map_df[gen_map_df.chm == "chr"+chm]
        else:
            gen_map_df = gen_map_df[gen_map_df.chm == chm]

    return gen_map_df

def cM2nsnp(cM, chm_len_pos, genetic_map, chm=None):

    if type(genetic_map) == str:
        if chm is not None:
            gen_map_df = read_genetic_map(genetic_map, chm)
        else:
            print("Need chromosome number to read genetic map")
    else:
        gen_map_df = genetic_map

    chm_len_cM = np.array(gen_map_df["pos_cm"])[-1]
    snp_len = int(round(cM*(chm_len_pos/chm_len_cM)))

    return snp_len

def fb2proba(path_to_fb, n_wind=None):
    
    with open(path_to_fb) as f:
        header = f.readline().split("\n")[0]
        ancestry = np.array(header.split("\t")[1:])
    A = len(ancestry)
    
    fb_df = pd.read_csv(path_to_fb, sep="\t", skiprows=[0])

    samples = [s.split(":::")[0] for s in fb_df.columns[4::A*2]]
    
    # Probabilities in snp space
    fb = np.array(fb_df)[:,4:]
    C, AN = fb.shape
    N = AN//A
    fb_reshaped = fb.reshape(C, N, A)      # (C, N, A)
    proba = np.swapaxes(fb_reshaped, 0, 1) # (N, C, A)
    
    # Probabilities in window space
    if n_wind is not None:
        gen_pos = np.array(fb_df['genetic_position'])
        w_cM = np.arange(gen_pos[0], gen_pos[-1], step = gen_pos[-1]/n_wind)
        f = interp1d(gen_pos, np.arange(C), fill_value=(0, C), bounds_error=False) 
        w_idx = f(w_cM).astype(int)
        proba = proba[:,w_idx,:]
    
    return proba

def update_vcf(vcf_data, mask=None, Updates=None):

    out = vcf_data.copy()
    
    if mask is not None:
        for key in vcf_data.keys():
            if key != "samples":
                out[key] = vcf_data[key][mask]

    if Updates is not None:
        for key in Updates.keys():
            if key != "samples":
                out[key] = Updates[key]

    return out

def get_name(name_len=8):
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(name_len)) 

def npy_to_vcf(reference, npy, results_file, verbose=False):
    """
    - reference: str path to reference file which provides metadata for the results
                 or alternatively, a allel.read_vcf output
    - npy: npy matrix - shape: (num_samples, num_snps)
           make sure npy file has same snp positions
    - results_file: str output vcf path
    
    this is a very light version of npy_to_vcf for LAI applications
    
    Function behavior
    a vcf file called <results_file> with data in npy file and metadata from reference
    - metadata includes all fields except for genotype data
    - npy file must follow convention where maternal and paternal sequences appear one after the other
      for each sample

    NOTE: New to production. Has not been bullet-tested.
    """
    
    if results_file.split(".")[-1] not in [".vcf", ".bcf"]:
        results_file += ".vcf"

    # read in the input vcf data
    if type(reference) == str:
        data = allel.read_vcf(reference)
    else:
        data = reference.copy()
    
    # infer chromosome length and number of samples
    npy = npy.astype(int)
    chmlen, _, _ = data["calldata/GT"].shape
    h, c = npy.shape
    n = h//2
    assert chmlen == c, "reference (" + str(chmlen) + ") and numpy matrix (" + str(c) + ") not compatible"

    # Keep sample names if appropriate
    if "samples" in list(data.keys()) and len(data["samples"]) == n:
        if verbose:
            print("Using same sample names")
        data_samples = data["samples"]
    else:
        data_samples = [get_name() for _ in range(n)]

    # metadata 
    df = pd.DataFrame()
    df["CHROM"]  = data["variants/CHROM"]
    df['POS']    = data["variants/POS"]
    df["ID"]     = data["variants/ID"]
    df["REF"]    = data["variants/REF"]
    df["VAR"]    = data["variants/ALT"][:,0] # ONLY THE FIRST SINCE WE ONLY CARE ABOUT BI-ALLELIC SNPS HERE FOR NOW
    df["QUAL"]   = data["variants/QUAL"]
    df["FILTER"] = ["PASS"]*chmlen
    df["INFO"]   = ["."]*chmlen
    df["FORMAT"] = ["GT"]*chmlen
    
    # genotype data for each sample
    for i in range(n):
    
        # get that particular individual's maternal and paternal snps
        maternal = npy[i*2,:].astype(str) # maternal is the first
        paternal = npy[i*2+1,:].astype(str) # paternal is the second

        # create "maternal|paternal"
        lst = [maternal, ["|"]*chmlen, paternal]
        genotype_person = list(map(''.join, zip(*lst)))
        df[data_samples[i]] = genotype_person

    if verbose:
        print("writing vcf data in "+results_file)

    # write header
    with open(results_file,"w") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write("##source=pyadmix (XGMix)\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        f.write("#"+"\t".join(df.columns)+"\n") # mandatory header
    
    # genotype data
    df.to_csv(results_file,sep="\t",index=False,mode="a",header=False)
    
    return
