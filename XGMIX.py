import gzip
import numpy as np
import os
import pickle
import sys
from time import time

from Utils.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, npy_to_vcf, update_vcf 
from Utils.utils import cM2nsnp, get_num_outs, read_genetic_map
from Utils.preprocess import load_np_data, data_process, get_gen_0
from Utils.postprocess import get_meta_data, write_msp_tsv, write_fb_tsv
from Utils.visualization import plot_cm, CM
from Utils.gnomix import Gnomix
from Admixture.Admixture import read_sample_map, split_sample_map, main_admixture
from Admixture.fast_admix import main_admixture_fast

from XGFix.XGFIX import XGFix

from config import verbose, run_simulation, founders_ratios, generations, rm_simulated_data
# from config import num_outs
from config import model_name, window_size_cM, smooth_size, missing, n_cores, r_admixed
from config import retrain_base, calibrate, context_ratio, instance_name, mode_filter_size, smooth_depth

CLAIMER = 'When using this software, please cite: \n' + \
          'Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A. \n' + \
          '"XGMix: Local-Ancestry Inference With Stacked XGBoost" \n' + \
          'International Conference on Learning Representations Workshops \n' + \
          'ICLR, 2020, Workshop AI4AH \n' + \
          'https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1'

FAST_ADMIX = True
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

    # This is temorary while there are still pre-trained models missing those members
    try:
        model.calibrate
    except AttributeError:
        model.calibrate = None

    # Same for mode filter
    try:
        model.mode_filter_size
    except AttributeError:
        model.mode_filter_size = 5

    # Same for context_ratio
    try:
        model.context
    except AttributeError:
        model.context = 0

    return model

def train(chm, model_name, genetic_map_df, data_path, generations, window_size_cM, 
          smooth_size, missing, n_cores, verbose, instance_name, 
          retrain_base, calibrate, context_ratio, mode_filter_size, smooth_depth, gen_0,
          output_path):

    if verbose:
        print("Preprocessing data...")
    
    # ------------------ Config ------------------
    model_name += "_chm_" + chm
    model_repo = join_paths(output_path, "models", verb=False)
    model_repo = join_paths(model_repo, model_name, verb=False)
    model_path = model_repo + "/" + model_name + ".pkl"

    train1_paths = [data_path + "/chm" + chm + "/simulation_output/train1/gen_" + str(gen) + "/" for gen in generations]
    train2_paths = [data_path + "/chm" + chm + "/simulation_output/train2/gen_" + str(gen) + "/" for gen in generations]
    val_paths    = [data_path + "/chm" + chm + "/simulation_output/val/gen_"    + str(gen) + "/" for gen in generations]

    position_map_file   = data_path + "/chm"+ chm + "/positions.txt"
    reference_map_file  = data_path + "/chm"+ chm + "/references.txt"
    population_map_file = data_path + "/populations.txt"

    snp_pos = np.loadtxt(position_map_file,  delimiter='\n').astype("int")
    snp_ref = np.loadtxt(reference_map_file, delimiter='\n', dtype=str)
    pop_order = np.genfromtxt(population_map_file, dtype="str")
    chm_len = len(snp_pos)
    num_anc = len(pop_order)

    window_size_pos = cM2nsnp(cM=window_size_cM, chm=chm, chm_len_pos=chm_len, genetic_map=genetic_map_df)
    
    # ------------------ Process data ------------------
    # gather feature data files (binary representation of variants)
    X_fname = "mat_vcf_2d.npy"
    X_train1_files = [p + X_fname for p in train1_paths]
    X_train2_files = [p + X_fname for p in train2_paths]
    X_val_files    = [p + X_fname for p in val_paths]

    # gather label data files (population)
    labels_fname = "mat_map.npy"
    labels_train1_files = [p + labels_fname for p in train1_paths]
    labels_train2_files = [p + labels_fname for p in train2_paths]
    labels_val_files    = [p + labels_fname for p in val_paths]

    # load the data
    train_val_files = [X_train1_files, labels_train1_files, X_train2_files, labels_train2_files, X_val_files, labels_val_files]
    X_train1_raw, labels_train1_raw, X_train2_raw, labels_train2_raw, X_val_raw, labels_val_raw = [load_np_data(f) for f in train_val_files]

    # adding generation 0
    if gen_0:
        if verbose:
            print("Including generation 0...")
        
        # get it
        gen_0_sets = ["train1", "train2"]
        X_train1_raw_gen_0, y_train1_raw_gen_0, X_train2_raw_gen_0, y_train2_raw_gen_0 = get_gen_0(data_path + "/chm" + chm, population_map_file, gen_0_sets)

        # add it
        X_train1_raw = np.concatenate([X_train1_raw, X_train1_raw_gen_0])
        labels_train1_raw = np.concatenate([labels_train1_raw, y_train1_raw_gen_0])
        X_train2_raw = np.concatenate([X_train2_raw, X_train2_raw_gen_0])
        labels_train2_raw = np.concatenate([labels_train2_raw, y_train2_raw_gen_0])

        # delete it
        del X_train1_raw_gen_0, y_train1_raw_gen_0, X_train2_raw_gen_0, y_train2_raw_gen_0 

    # reshape according to window size 
    X_t1, y_t1 = data_process(X_train1_raw, labels_train1_raw, window_size_pos, missing)
    X_t2, y_t2 = data_process(X_train2_raw, labels_train2_raw, window_size_pos, missing)
    X_v, y_v       = data_process(X_val_raw, labels_val_raw, window_size_pos, missing)

    del X_train1_raw, X_train2_raw, X_val_raw, labels_train1_raw, labels_train2_raw, labels_val_raw

    # ------------------ Train model ------------------    
    # init, train, evaluate and save model
    if verbose:
        print("Initializing XGMix model and training...")
    # model = XGMIX(chm_len, window_size_pos, smooth_size, num_anc, 
    #               snp_pos, snp_ref, pop_order, calibrate=calibrate, 
    #               cores=n_cores, context_ratio=context_ratio,
    #               mode_filter_size=mode_filter_size, 
    #               base_params = [20,4], smooth_params=[100,smooth_depth])
    model = Gnomix(C=chm_len, M=window_size_pos, A=num_anc, 
                  snp_pos=snp_pos, snp_ref=snp_ref, population_order=pop_order,
                  calibrate=calibrate, n_jobs=n_cores, context_ratio=context_ratio)
    # other params: mode_filter_size

    model.train( data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v)), retrain_base=retrain_base, verbose=verbose)

    # evaluate model
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    CM(y_v.ravel(), model.predict(X_v).ravel(), pop_order, analysis_path, verbose)
    print("Saving model at {}".format(model_path))
    pickle.dump(model, open(model_path,"wb"))

    # write the model parameters of type int, float, str into a file config.
    # so there is more clarity on what the model parameters were.
    # NOTE: Not tested fully yet. # TODO
    model_config_path = os.path.join(model_repo,"config.txt")
    print("Saving model info at {}".format(model_config_path))
    model.write_config(model_config_path)

    return model

def main(args, verbose=True, **kwargs):

    run_simulation=kwargs.get("run_simulation")
    founders_ratios=kwargs.get("founders_ratios")
    #num_outs=kwargs.get("num_outs")
    generations=kwargs.get("generations")
    rm_simulated_data=kwargs.get("rm_simulated_data")
    model_name=kwargs.get("model_name")
    window_size_cM=kwargs.get("window_size_cM")
    smooth_size=kwargs.get("smooth_size")
    missing=kwargs.get("missing")
    n_cores=kwargs.get("n_cores")
    retrain_base=kwargs.get("retrain_base")
    calibrate=kwargs.get("calibrate")
    context_ratio=kwargs.get("context_ratio")
    instance_name=kwargs.get("instance_name")
    mode_filter_size=kwargs.get("mode_filter_size")
    smooth_depth=kwargs.get("smooth_depth")
    r_admixed = kwargs.get("r_admixed")
    simulated_data_path=kwargs.get("simulated_data_path")
    # the above variable has to be a path that ends with /generated_data/
    # gotta be careful if using rm_simulated_data. NOTE

    output_path = args.output_basename if instance_name == "" else join_paths(args.output_basename,instance_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gen_map_df = read_genetic_map(args.genetic_map_file, args.chm)

    mode = args.mode # this needs to be done. master change 1.
    # The simulation can't handle generation 0, add it separetly
    gen_0 = 0 in generations
    generations = list(filter(lambda x: x != 0, generations))
    if FAST_ADMIX:
        gen_0 = False
        generations = [0]+generations

    np.random.seed(94305)

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
            if FAST_ADMIX:
                print("Fast admix...")
                main_admixture_fast(args.chm, data_path, set_names, sample_map_paths, sample_map_idxs,
                           args.reference_file, args.genetic_map_file, num_outs_per_gen, generations)
            else:
                main_admixture(args.chm, data_path, set_names, sample_map_paths, sample_map_idxs,
                           args.reference_file, args.genetic_map_file, num_outs_per_gen, generations)

            if verbose:
                print("Simulation done.")
                print("-"*80+"\n"+"-"*80+"\n"+"-"*80)
        else:
            print("Using simulated data from " + data_path + " ...")

        # Processing data, init and training model
        model = train(args.chm, model_name, gen_map_df, data_path, generations,
                        window_size_cM, smooth_size, missing, n_cores, verbose,
                        instance_name, retrain_base, calibrate, context_ratio,
                        mode_filter_size, smooth_depth, gen_0, output_path)
        if verbose:
            print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

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

    if mode=="train" and rm_simulated_data:
        if verbose:
            print("Removing simulated data...")
        chm_path = join_paths(data_path, "chm" + args.chm, verb=False)
        remove_data_cmd = "rm -r " + chm_path
        run_shell_cmd(remove_data_cmd, verbose=False)

    if verbose:
        print("Finishing up...")

if __name__ == "__main__":

    # Citation
    print("-"*80+"\n"+"-"*35+"  XGMix  "+"-"*36 +"\n"+"-"*80)
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
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <reference_file> <sample_map_file>")
        print("Usage when using a pre-trained model:")
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <path_to_model>")
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
    args = Struct(**base_args)
    if mode == "train":
        args.reference_file  = sys.argv[6]
        args.sample_map_file = sys.argv[7]
    elif mode == "pre-trained":
        args.path_to_model = sys.argv[6]

    # Run it
    if verbose:
        print("Launching XGMix in", mode, "mode...")
    main(args, verbose=verbose, run_simulation=run_simulation, founders_ratios=founders_ratios,
        generations=generations, rm_simulated_data=rm_simulated_data,
        model_name=model_name, window_size_cM=window_size_cM, smooth_size=smooth_size, 
        missing=missing, n_cores=n_cores, r_admixed=r_admixed,
        retrain_base=retrain_base, calibrate=calibrate, context_ratio=context_ratio, 
        instance_name=instance_name, mode_filter_size=mode_filter_size, smooth_depth=smooth_depth)
