import allel
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from operator import itemgetter
import os
import sys
import re

from Admixture.utils import *

def write_founders_bcf(reference_file_bcf, sample_file, output_path, verb=True):
    """
    Copy contents of sample map into founders map, create the bcg.gz files and index them.
    """

     # founders map - same as input map except renamed
    founder_map_cmd = "cp %s %s/founders.map" % (sample_file, output_path)
    run_shell_cmd(founder_map_cmd, verb=verb)

    founders_bcf_cmd = "cat {0}/founders.map | cut -f 1 | bcftools view --output-type b --output-file {0}/founders.bcf.gz --samples-file - --threads 12 --force-samples {1}".format(output_path, reference_file_bcf)
    run_shell_cmd(founders_bcf_cmd, verb=verb)

    index_founders_cmd = "bcftools index -f %s/founders.bcf.gz" % (output_path)
    run_shell_cmd(index_founders_cmd, verb=verb)
    
    # create vcf files for the founders for downstream conversion into npy files.
    founders_vcf_cmd = "bcftools view %s/founders.bcf.gz -o %s/founders.vcf -O v"% (output_path,output_path)
    run_shell_cmd(founders_vcf_cmd, verb=verb)
    
def simulate(reference_file_bcf, sample_file, idx_to_pop_map, genetic_map_file,
             generations, num_out, sim_output_path, chm, use_phase_shift):

    # path to RFMix/simulate binary 
    rfmix_sim_path = "./Admixture/simulate" 
    # NOTE: If running from a different directory than XGMIX/, this needs to
    # be updated.

    # assume everyone in the sample map is founder
    print("Creating founders.bcf.gz for {}".format(sample_file))
    write_founders_bcf(reference_file_bcf, sample_file, sim_output_path)

    for gen in generations:
        
        print("-"*80)
        print("Simulation for generation {} from {}".format(gen,sample_file))

        # generation simulation output path
        gen_path = join_paths(sim_output_path, "gen_"+str(gen))
        out_basename = gen_path+'/admix'
        
        # Simulating the individuals via rfmix simulation
        print("simulating ...")
        rfmix_sim_cmd = rfmix_sim_path + " -f %s/founders.bcf.gz -m %s/founders.map -g %s -o %s --growth-rate=1.5 --maximum-size=2000 --n-output=%d -c %s -G %d -p %f --random-seed=%d %s"
        rfmix_sim = rfmix_sim_cmd % (sim_output_path, sim_output_path, genetic_map_file, out_basename, num_out, chm, gen, 0.0, 123, "--dephase" if use_phase_shift else "")
        try:
            run_shell_cmd(rfmix_sim, verb=True)
        except:
            print("something went wrong using rfmix/simulate ...", end=" ")
            print("trying -c chr"+chm+" insted of -c "+chm+" ...")
            rfmix_sim = rfmix_sim_cmd % (sim_output_path, sim_output_path, genetic_map_file, out_basename, num_out, "chr"+chm, gen, 0.0, 123, "--dephase" if use_phase_shift else "")
            run_shell_cmd(rfmix_sim, verb=True)

        # reading .vcf output of simulation and converting to npy matricies
        print('reading .vcf output of simulation and converting to npy matricies ...')
        vcf_data = allel.read_vcf(out_basename+".query.vcf")
        chm_len, nout, _ = vcf_data["calldata/GT"].shape
        mat_vcf_2d = vcf_data["calldata/GT"].reshape(chm_len,nout*2).T
        np.save(gen_path+"/mat_vcf_2d.npy", mat_vcf_2d)
        
        # reading .map output of simulation and converting to npy matricies
        print('reading .map output of simulation and converting to npy matricies ...')
        map_path = out_basename + ".result"
        matrix = sample_map_to_matrix(map_path)

        # Finally map them to original labels (which can be further mapped to coordinates) and saving
        matrix = np.vectorize(idx_to_pop_map.get)(matrix)
        np.save(gen_path+"/mat_map.npy",matrix)
    
    print("-"*80)
    print("Finishing up ...")
