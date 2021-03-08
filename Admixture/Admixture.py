import allel
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import random
import sys

from Admixture.utils import *
from Admixture.simulation import simulate

def read_sample_map(sample_map_file, population_path):

    samples = pd.read_csv(sample_map_file, sep="\t")
    samples.columns = ['Sample', 'Population']
    
    # Register and writing population mapping to labels: [Population, Label]
    pops = [dat["Population"] for idx, dat in samples.iterrows()]
    pop_ids = dict(zip(sorted(set(pops)),range(len(pops))))
    sorted_pop = sorted(pop_ids.items(), key=itemgetter(1)) # sort alphabetically to avoid confusion
    pop_order = [p[0] for p in sorted_pop]
    with open(population_path+"/populations.txt", 'w') as f:
        f.write(" ".join(pop_order))

    return samples, pop_ids

def split_sample_map(sample_ids, populations, ratios, pop_ids, sample_map_paths):
    """
    Given sample_ids, populations and the amount of data to be put into each set,
    Split it such that all sets get even distribution of sample_ids for each population.
    """

    assert sum(ratios) == 1, "ratios must sum to 1"
    
    set_ids = [[] for _ in ratios]
    
    for p in np.unique(populations):

        # subselect population
        pop_idx = populations == p
        pop_sample_ids = list(np.copy(sample_ids[pop_idx]))
        n_pop = len(pop_sample_ids)

        # find numbr of samples in each set
        n_sets = [round(r*n_pop) for r in ratios]
        while sum(n_sets) > n_pop:
            n_sets[0] -= 1    
        while sum(n_sets) < n_pop:
            n_sets[-1] += 1

        # divide the samples accordingly
        for s, r in enumerate(ratios):
            n_set = n_sets[s]
            set_ids_idx = np.random.choice(len(pop_sample_ids),n_set,replace=False)
            set_ids[s] += [[pop_sample_ids.pop(idx), p] for idx in sorted(set_ids_idx,reverse=True)]

    # write to disk
    for i, sample_fname in enumerate(sample_map_paths):
        write_sample_map(set_ids[i], sample_map_paths[i])

    sample_map_file_idxs = [get_sample_map_file_idxs(f, pop_ids) for f in sample_map_paths]
        
    return sample_map_file_idxs
    
def main_admixture(chm, root, sub_instance_names, sample_map_files, sample_map_files_idxs, reference_file, genetic_map_file,
    num_outs, generations = [2,4,6], use_phase_shift = False, verbose=True):

    output_path = join_paths(root, 'chm{}'.format(chm), verb=verbose)
    
    # path for simulation output
    simulation_output_path = join_paths(output_path, 'simulation_output')

    # Register and writing SNP physical positions
    ref = read_vcf(reference_file)
    np.savetxt(output_path +  "/positions.txt", ref['variants/POS'], delimiter='\n')
    np.savetxt(output_path + "/references.txt", ref['variants/REF'], delimiter='\n', fmt="%s")

    # Convert to .bcf file format if not already there (format required by rfmix-simulate)
    reference_file_bcf = convert_to_bcf(reference_file, output_path=output_path)

    # simulate for each sub-instance
    for i, instance_name in enumerate(sub_instance_names):

        if num_outs[i] > 0:
            # paths for each set
            instance_path = join_paths(simulation_output_path, instance_name, verb=verbose)
            
            simulate(reference_file_bcf, sample_map_files[i], sample_map_files_idxs[i],
                    genetic_map_file, generations, num_outs[i], instance_path, chm,
                    use_phase_shift)

if __name__ == "__main__":

    # ARGS
    instance_name, sample_map_file, genetic_map_file, reference_file = sys.argv[1:5]

    # Set output path
    root = './generated_data'
    if instance_name is not None:
        root += '/' + instance_name

    # Splitting the sample in train/val/test
    sub_instance_names = ["train", "val", "test"]
    sample_map_files, sample_map_files_idxs = split_sample_map(root, sample_map_file)
    num_outs = [700, 200, 0] # how many individuals in each sub-instance

    # Simulate for all chromosomes
    chms = np.array(range(22))+1
    for chm in chms:
        print("-"*80+"\n"+"-"*80+"\n"+"-"*80+"\n")
        print("Simulating chromosome " + chm)
        main_admixture(chm, root, sub_instance_names, sample_map_files, sample_map_files_idxs, reference_file, genetic_map_file, num_outs)


