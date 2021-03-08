import numpy as np
from Admixture.utils import *
from pyadmix import simulate
from pyadmix import get_chm_info, get_sample_map_data, write_output

def main_admixture_fast(chm, root, sub_instance_names, sample_map_files, sample_map_files_idxs, reference_file, genetic_map_file,
    num_outs, generations, verbose=True):

    """
    chm: chm number
    root: data path with generated_data folder
    sub_instance_names: (a list) subsets like train1, train2, val
    sample_map_files: (a list) the files of the above
    sample_map_file_idxs: (a list) a way to make sure the individual sets map to the original
    population names
    reference_file: vcf file
    genetic_map_file: gmap file
    num_outs: (a list) number of outputs for each generation for each set
    generations: generations to simulate for each set (a single list)

    """

    output_path = join_paths(root, 'chm{}'.format(chm), verb=verbose)
    
    # path for simulation output
    simulation_output_path = join_paths(output_path, 'simulation_output')

    # Register and writing SNP physical positions
    ref = read_vcf(reference_file)
    np.savetxt(output_path +  "/positions.txt", ref['variants/POS'], delimiter='\n')
    np.savetxt(output_path + "/references.txt", ref['variants/REF'], delimiter='\n', fmt="%s")

    # Process genetic map data
    genetic_map_data = get_chm_info(genetic_map_file, ref)

    # simulate for each sub-instance
    for i, instance_name in enumerate(sub_instance_names):

        if num_outs[i] > 0:
            # paths for each set
            instance_path = join_paths(simulation_output_path, instance_name, verb=verbose)
            # get sample map data
            sample_map_data = get_sample_map_data(sample_map_files[i], ref, sample_weights=None)
            # get the dataset
            dataset = simulate(ref, sample_map_data, genetic_map_data, out_root=None,
                num_samples_per_gen=num_outs[i], gens_to_ret=generations,
                random_seed=94305,verbose=verbose)
            
            # apply sample_map_files_idxs trasnform
            idx_to_pop_map = sample_map_files_idxs[i]
            if not is_same(idx_to_pop_map):
                for key in dataset.keys():
                    for i in range(len(dataset[key])):
                        dataset[key][i].maternal["anc"] = np.vectorize(idx_to_pop_map.get)(dataset[key][i].maternal["anc"])
                        dataset[key][i].paternal["anc"] = np.vectorize(idx_to_pop_map.get)(dataset[key][i].paternal["anc"])

            # save the data
            write_output(instance_path,dataset)

def is_same(mapper):
    for key in mapper.keys():
        if key != mapper[key]:
            return False
    return True
