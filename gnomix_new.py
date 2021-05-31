import yaml


import gzip
import numpy as np
import os
import pickle
import sys

from Admixture.Admixture import read_sample_map, split_sample_map, main_admixture
from Admixture.fast_admix import main_admixture_fast

from src.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, npy_to_vcf, update_vcf 
from src.utils import cM2nsnp, get_num_outs, read_genetic_map
from src.preprocess import load_np_data, data_process, get_gen_0
from src.postprocess import get_meta_data, write_msp_tsv, write_fb_tsv
from src.visualization import plot_cm

from src.gnomix import Gnomix

from XGFix.XGFIX import XGFix

from config import verbose, run_simulation, founders_ratios, generations, rm_simulated_data
from config import model_name, inference, window_size_cM, smooth_size, missing, n_cores, r_admixed
from config import retrain_base, calibrate, context_ratio, instance_name

CLAIMER = 'When using this software, please cite: \n' + \
          'Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A. \n' + \
          '"XGMix: Local-Ancestry Inference With Stacked XGBoost" \n' + \
          'International Conference on Learning Representations Workshops \n' + \
          'ICLR, 2020, Workshop AI4AH \n' + \
          'https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1'

np.random.seed(94305)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_model(path_to_model, verbose=True):
    if verbose:
        print("Loading model...")
    if path_to_model[-3:]==".gz":
        with gzip.open(path_to_model, 'rb') as unzipped:
            model = pickle.load(unzipped)
    else:
        model = pickle.load(open(path_to_model,"rb"))

    return model

def get_data(data_path, chm, window_size_cM, generations, gen_0, genetic_map_df, validate=True, verbose=False):

    if verbose:
        print("Preprocessing data...")
    
    # ------------------ Meta ------------------

    position_map_file   = data_path + "/chm"+ chm + "/positions.txt"
    reference_map_file  = data_path + "/chm"+ chm + "/references.txt"
    population_map_file = data_path + "/populations.txt"

    snp_pos = np.loadtxt(position_map_file,  delimiter='\n').astype("int")
    snp_ref = np.loadtxt(reference_map_file, delimiter='\n', dtype=str)
    pop_order = np.genfromtxt(population_map_file, dtype="str")
    A = len(pop_order)
    C = len(snp_pos)
    M = cM2nsnp(cM=window_size_cM, chm=chm, chm_len_pos=C, genetic_map=genetic_map_df)

    meta = {
        "A": A, # number of ancestry
        "C": C, # chm length
        "M": M, # window size in SNPs
        "snp_pos": snp_pos,
        "snp_ref": snp_ref,
        "pop_order": pop_order
    }

    # ------------------ Process data ------------------

    def read(split, gen_0):
        paths = [data_path + "/chm" + chm + "/simulation_output/"+split+"/gen_" + str(gen) + "/" for gen in generations]
        X_files = [p + "mat_vcf_2d.npy" for p in paths]
        labels_files = [p + "mat_map.npy" for p in paths]
        X_raw, labels_raw = [load_np_data(f) for f in [X_files, labels_files]]
        if gen_0:
            X_raw_gen_0, y_raw_gen_0 = get_gen_0(data_path + "/chm" + chm, population_map_file, split)
            X_raw = np.concatenate([X_raw, X_raw_gen_0])
            labels_raw = np.concatenate([labels_raw, y_raw_gen_0])
        X, y = data_process(X_raw, labels_raw, M)
        return X, y

    X_t1, y_t1 = read("train1", gen_0)
    X_t2, y_t2 = read("train2", gen_0)
    X_v, y_v   = read("val"   , False) if validate else (None, None)

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v))

    return data, meta

def main(args, verbose=True, **kwargs):

    run_simulation=kwargs.get("run_simulation")
    founders_ratios=kwargs.get("founders_ratios")
    generations=kwargs.get("generations")
    rm_simulated_data=kwargs.get("rm_simulated_data")
    model_name=kwargs.get("model_name")
    inference=kwargs.get("inference")
    window_size_cM=kwargs.get("window_size_cM")
    n_cores=kwargs.get("n_cores")
    retrain_base=kwargs.get("retrain_base")
    calibrate=kwargs.get("calibrate")
    context_ratio=kwargs.get("context_ratio")
    instance_name=kwargs.get("instance_name")
    r_admixed = kwargs.get("r_admixed")
    simulated_data_path=kwargs.get("simulated_data_path")
    # the above variable has to be a path that ends with /generated_data/
    # gotta be careful if using rm_simulated_data. NOTE

    # option to bypass validation
    validate = founders_ratios[-1] != 0

    output_path = args.output_basename if instance_name == "" else join_paths(args.output_basename,instance_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gen_map_df = read_genetic_map(args.genetic_map_file, args.chm)

    mode = args.mode # this needs to be done. master change 1.

    # The simulation can't handle generation 0, add it separetly
    gen_0 = 0 in generations
    generations = list(filter(lambda x: x != 0, generations))
    gen_0 = False
    generations = [0]+generations

    np.random.seed(94305) # TODO: move into config file, LAIData/simulation and model should take as argument

    # Either load pre-trained model or simulate data from reference file, init model and train it
    if mode == "pre-trained":
        model = load_model(args.path_to_model, verbose=verbose)
    elif args.mode == "train":

        # Set output path: master change 2
        data_path = join_paths(output_path, 'generated_data', verb=False)

        # added functionality: users can now specify where pre-imulated data is
        if run_simulation == False and simulated_data_path is not None:
            data_path = simulated_data_path + "/"

        # Running simulation. If data is already simulated, skipping can save a lot of time
        if run_simulation:

            # Splitting the data into train1 (base), train2 (smoother), val, test 
            if verbose:
                print("Reading sample maps and splitting in train/val...")
            samples, pop_ids = read_sample_map(args.sample_map_file, population_path = data_path)
            set_names = ["train1", "train2", "val"]
            sample_map_path = join_paths(data_path, "sample_maps", verb=verbose)
            sample_map_paths = [sample_map_path+"/"+s+".map" for s in set_names]
            sample_map_idxs = split_sample_map(sample_ids = np.array(samples["Sample"]),
                                                populations = np.array(samples["Population"]),
                                                ratios = founders_ratios,
                                                pop_ids = pop_ids,
                                                sample_map_paths=sample_map_paths)

            # Simulating data
            if verbose:
                print("Running simulation...")
            num_outs = get_num_outs(sample_map_paths, r_admixed)
            num_outs_per_gen = [n//len(generations) for n in num_outs]
            print("Running admixture...")
            main_admixture_fast(args.chm, data_path, set_names, sample_map_paths, sample_map_idxs,
                        args.reference_file, args.genetic_map_file, num_outs_per_gen, generations)

            if verbose:
                print("Simulation done.")
                print("-"*80)
        else:
            print("Using simulated data from " + data_path + " ...")

        # Processing data
        data, meta = get_data(data_path, args.chm, window_size_cM, generations, gen_0, gen_map_df, validate=validate, verbose=verbose)

        # init model
        model = Gnomix(C=meta["C"], M=meta["M"], A=meta["A"],
                        snp_pos=meta["snp_pos"], snp_ref=meta["snp_ref"],
                        population_order=meta["pop_order"],
                        mode=inference, calibrate=calibrate,
                        n_jobs=n_cores, context_ratio=context_ratio)

        # train it
        model.train(data=data, retrain_base=retrain_base, evaluate=True, verbose=verbose)

        # store it
        model_repo = join_paths(output_path, "models", verb=False)
        model_repo = join_paths(model_repo, model_name + "_chm_" + args.chm, verb=False)
        model_path = model_repo + "/" + model_name + "_chm_" + args.chm + ".pkl"
        pickle.dump(model, open(model_path,"wb"))

        # brief analysis
        if verbose:
            print("Analyzing model performance...")
        analysis_path = join_paths(model_repo, "analysis", verb=False)
        cm_path = analysis_path+"/confusion_matrix_{}.txt"
        cm_plot_path = analysis_path+"/confusion_matrix_{}_normalized.png"
        analysis_sets = ["train", "val"] if validate else ["train"]
        for d in analysis_sets:
            cm, idx = model.Confusion_Matricies[d]
            n_digits = int(np.ceil(np.log10(np.max(cm))))
            np.savetxt(cm_path.format(d), cm, fmt='%-'+str(n_digits)+'.0f')
            plot_cm(cm, labels=model.population_order[idx], path=cm_plot_path.format(d))
            if verbose:
                print("Estimated "+d+" accuracy: {}%".format(model.accuracies["smooth_"+d+"_acc"]))

        # write the model parameters of type int, float, str into a file config TODO: test
        model_config_path = os.path.join(model_repo, "config.txt")
        model.write_config(model_config_path)

        if verbose:
            print("Model, info and analysis saved at {}".format(model_repo))
            print("-"*80)

        if rm_simulated_data:
            if verbose:
                print("Removing simulated data...")
            chm_path = join_paths(data_path, "chm" + args.chm, verb=False)
            remove_data_cmd = "rm -r " + chm_path
            run_shell_cmd(remove_data_cmd, verbose=False)

    # Predict the query data
    if args.query_file is not None:
        
        if verbose:
            print("Loading and processing query file...")

        # Load and process user query vcf file
        query_vcf_data = read_vcf(args.query_file, chm=args.chm, fields="*")
        X_query, vcf_idx, fmt_idx = vcf_to_npy(query_vcf_data, model.snp_pos, model.snp_ref, return_idx=True, verbose=verbose)

        # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
        if verbose:
            print("Inferring ancestry on query data...")

        if args.phase:
            X_query_phased, label_pred_query_window = model.phase(X_query)
            if verbose:
                print("Writing phased SNPs to disc...")
            U = {
                "variants/REF": model.snp_ref[fmt_idx],
                "variants/ALT": np.expand_dims(np.repeat("NA", len(fmt_idx)),axis=1)
            }
            query_vcf_data = update_vcf(query_vcf_data, mask=vcf_idx, Updates=U)
            query_phased_prefix = output_path + "/" + "query_file_phased"
            npy_to_vcf(query_vcf_data, X_query_phased[:,fmt_idx], query_phased_prefix)
            proba_query_window = model.predict_proba(X_query_phased)
        else: 
            label_pred_query_window = model.predict(X_query)
            proba_query_window = model.predict_proba(X_query)

        # writing the result to disc
        if verbose:
            print("Writing inference to disc...")
        meta_data = get_meta_data(args.chm, model.snp_pos, query_vcf_data['variants/POS'], model.W, model.M, gen_map_df)
        out_prefix = output_path + "/" + output_path.split("/")[-1]
        write_msp_tsv(out_prefix, meta_data, label_pred_query_window, model.population_order, query_vcf_data['samples'])
        write_fb_tsv(out_prefix, meta_data, proba_query_window, model.population_order, query_vcf_data['samples'])


if __name__ == "__main__":

    # Citation
    print("-"*80+"\n"+"-"*35+"  Gnomix  "+"-"*35 +"\n"+"-"*80)
    print(CLAIMER)
    print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

    # Infer mode from number of arguments
    mode = None
    if len(sys.argv) == 7:
        mode = "pre-trained" 
    if len(sys.argv) == 8:
        mode = "train"

    # Usage message
    if mode is None:
        if len(sys.argv) > 1:
            print("Error: Incorrect number of arguments.")
        print("Usage when training a model from scratch:")
        print("   $ python3 gnomix.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <reference_file> <sample_map_file>")
        print("Usage when using a pre-trained model:")
        print("   $ python3 gnomix.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <path_to_model>")
        sys.exit(0)

    # Deconstruct CL arguments
    base_args = {
        'mode': mode,
        'query_file': sys.argv[1] if sys.argv[1].strip() != "None" else None,
        'genetic_map_file': sys.argv[2],
        'output_basename': sys.argv[3],
        'chm': sys.argv[4],
        'phase': True if sys.argv[5].lower() == "true" else False
    }
    if mode == "train":
        base_args["reference_file"]  = sys.argv[6]
        base_args["sample_map_file"] = sys.argv[7]
        base_args["config_file"] = "./config.yaml"
        if sys.argv[8] is not None:
            base_args["config_file"] = sys.argv[8]
    elif mode == "pre-trained":
        base_args["path_to_model"] = sys.argv[6]

    # base_args is a dict
    # update it with the config yaml
    with open(base_args["config_file"],"r") as file:
        conf = yaml.load(file)
    
    verbose = conf["verbose"]
    # process args here...

    # Run it
    if verbose:
        print("Launching Gnomix in", mode, "mode...")
    
    if mode == "pre-trained":
        model = load_model(base_args["path_to_model"], verbose=verbose)
    else:
        model = train_model(base_args, conf, verbose=verbose)
        
    if base_args["query_data"]:
        run_inference(base_args, model, verbose=verbose)

    # main(args, verbose=verbose, run_simulation=run_simulation, founders_ratios=founders_ratios,
    #     generations=generations, rm_simulated_data=rm_simulated_data,
    #     model_name=model_name, inference=inference, window_size_cM=window_size_cM, smooth_size=smooth_size, 
    #     missing=missing, n_cores=n_cores, r_admixed=r_admixed,
    #     retrain_base=retrain_base, calibrate=calibrate, context_ratio=context_ratio, 
    #     instance_name=instance_name)


def train_model(base_args, conf, verbose):
    model = None
    # simulation and its outputs

    # read data and convert to npy

    # init and train model

    # analysis and writing output

    # remove simulated data

    return model

def run_inference(base_args, model, verbose):
    return None
