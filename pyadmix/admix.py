"""
Admix++
- Get a faster pipe for getting admixture simulation.
- Follow an approximate method to get this done.
- Allow weights for different ancestries - 
    seen this use case in our xgmix, lainet work.
- Get numpy files directly and be mindful of storage requirements
- sparse for vcf, rle for ancestries could be future steps

"""

import allel
import sys

from .utils import build_founders, create_dataset, write_output, create_non_rec_dataset
from .utils import get_chm_info, get_sample_map_data
RECURSIVE = False

def simulate(vcf_data, sample_map_data, genetic_map_data, out_root,
             num_samples_per_gen, gens_to_ret,random_seed=42,verbose=True):

    """

    out_root is modified to infer chromosome number, gen number. 
    vcf_data: allel.read_vcf output
    sample_map: Samples in the current split
    genetic_map: Genetic map
    out_root: Where output is stored. Needs to have chm name and split name.
    num_samples_per_gen: Number of samples per generation to simulate.
    gens_to_ret: Generations to return. Will output gen0 automatically.

    """
    if verbose:
        print("Building founders")
    founders, founders_weight= build_founders(vcf_data, 
                                              genetic_map_data, 
                                              sample_map_data)

    if verbose:
        print("Simulating...")
    if RECURSIVE:
        print("Warning!!! Using recursive simulation. May lead to biases")
        dataset = create_dataset(founders,
                                founders_weight,
                                num_samples_per_gen,gens_to_ret,
                                breakpoint_probability=genetic_map_data["breakpoint_probability"],
                                random_seed=random_seed,
                                verbose=verbose)
    else:
        dataset = create_non_rec_dataset(founders,
                                founders_weight,
                                num_samples_per_gen,gens_to_ret,
                                breakpoint_probability=genetic_map_data["breakpoint_probability"],
                                random_seed=random_seed,
                                verbose=verbose)
     
    if out_root == None:
        return dataset # useful when we want to create dataset iterators.
    if verbose:
        print("Writing output")
    write_output(out_root,dataset,verbose=verbose)

if __name__ == "__main__":

    """
    Sample command:
    python admix.py /home/arvindsk/datagen/world_wide_references/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22.vcf /home/wknd37/Admixture/generated_data/6_even_anc_t2/chm22/sample_maps/train.map /home/database/maps/rfmix/allchrs.b37.gmap /home/arvindsk/datagen/fast_admix/chm22/train/ 8 400

    """

    reference = sys.argv[1]
    sample_map = sys.argv[2]
    genetic_map = sys.argv[3]
    out_root = sys.argv[4]

    max_gen = int(sys.argv[5])
    # returns 2 to max_gen.
    # by default we write gen 0.
    gens_to_ret = range(2,max_gen+1)

    num_samples_per_gen = int(sys.argv[6])

    random_seed=42
    sample_weights=None

    if len(sys.argv) >= 8:
        random_seed = int(sys.argv[7])

    if len(sys.argv) >= 9:
        sample_weights = sys.argv[8]

    print("Reading vcf data")
    vcf_data = allel.read_vcf(reference)

    genetic_map_data = get_chm_info(genetic_map, vcf_data)
    sample_map_data = get_sample_map_data(sample_map, vcf_data, sample_weights)

    simulate(vcf_data, sample_map_data, genetic_map_data, out_root,
             num_samples_per_gen, gens_to_ret,
             random_seed)
