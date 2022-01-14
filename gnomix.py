import gzip
import numpy as np
import os
import pandas as pd
import pickle
import sys
import yaml

from src.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, npy_to_vcf, update_vcf 
from src.utils import read_genetic_map, save_dict, load_dict, read_headers
from src.preprocess import load_np_data, data_process
from src.postprocess import get_meta_data, write_msp, write_fb, msp_to_lai, msp_to_bed
from src.visualization import plot_cm, plot_chm
from src.laidataset import LAIDataset

from src.model import Gnomix

CLAIMER = 'When using this software, please cite: \n' + \
          'Helgi Hilmarsson, Arvind S Kumar, Richa Rastogi, Carlos D Bustamante, \n' + \
          'Daniel Mas Montserrat, Alexander G Ioannidis: \n' + \
          '"High Resolution Ancestry Deconvolution for Next Generation Genomic Data" \n' + \
          "https://www.biorxiv.org/content/10.1101/2021.09.19.460980v1"

def load_model(path_to_model, verbose=True):
    if verbose:
        print("Loading model...")
    if path_to_model[-3:]==".gz":
        with gzip.open(path_to_model, 'rb') as unzipped:
            model = pickle.load(unzipped)
    else:
        model = pickle.load(open(path_to_model,"rb"))

    return model

def run_inference(base_args, model, visualize, snp_level=False, bed_file_output=False, verbose=False):

    if verbose:
        print("Loading and processing query file...")

    query_file = base_args["query_file"]
    chm = base_args["chm"]
    output_path = base_args["output_basename"]
    gen_map_df = model.gen_map_df

    # Load and process user query vcf file
    query_vcf_data = read_vcf(query_file, chm=chm, fields="*")
    X_query, vcf_idx, fmt_idx = vcf_to_npy(query_vcf_data, model.snp_pos, model.snp_ref, return_idx=True, verbose=verbose)

    # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
    if verbose:
        print("Inferring ancestry on query data...")

    B_query = model.base.predict_proba(X_query)
    if not base_args["phase"]:
        y_proba_query = model.smooth.predict_proba(B_query)
        y_pred_query = np.argmax(y_proba_query, axis=-1)
    else:
        X_query_phased, y_pred_query = model.phase(X_query, B=B_query)
        if verbose:
            print("Writing phased SNPs to disk...")
        U = {
            "variants/REF": model.snp_ref[fmt_idx],
            "variants/ALT": model.snp_alt[fmt_idx].reshape(len(fmt_idx),1)
        }
        query_vcf_data_phase = update_vcf(query_vcf_data, mask=vcf_idx, Updates=U)
        query_phased_prefix = output_path + "/" + "query_file_phased"
        inf_headers = read_headers(query_file)
        npy_to_vcf(query_vcf_data_phase, X_query_phased[:,fmt_idx], query_phased_prefix, headers=inf_headers)
        # copy header to preserve it
        y_proba_query = model.predict_proba(X_query_phased)

    # writing the result to disk
    if verbose:
        print("Saving results...")
    meta_data = get_meta_data(chm, model.snp_pos, query_vcf_data['variants/POS'], model.W, model.M, gen_map_df)
    out_prefix = output_path + "/" + "query_results"
    write_msp(out_prefix, meta_data, y_pred_query, model.population_order, query_vcf_data['samples'])
    write_fb(out_prefix, meta_data, y_proba_query, model.population_order, query_vcf_data['samples'])

    # write the snp level results (BETA)
    if snp_level:
        msp_to_lai(msp_file=out_prefix+".msp", positions=query_vcf_data['variants/POS'], lai_file=out_prefix+".lai")

    if bed_file_output:
        bed_root = output_path + "/" + "query_results_bed"
        if not os.path.exists(bed_root):
            os.makedirs(bed_root)
        msp_to_bed(msp_file=out_prefix+".msp", root=bed_root, pop_order=model.population_order)

    # visualize results
    if visualize:
        vis_path = join_paths(output_path, "visual", verb=False)
        msp_df = pd.read_csv(out_prefix+".msp", sep="\t", skiprows=[0])
        for sample_id in query_vcf_data['samples']:
            sample_path = join_paths(vis_path, sample_id, verb=False)
            plot_chm(sample_id, msp_df, img_name=sample_path+"/chromosome_painting")

    return

def get_data(data_path, generations, window_size_cM):

    # ------------------ Meta ------------------
    assert(type(generations)==dict), "Generations must be a dict with list of generations to read in for each split"

    laidataset_meta_path = os.path.join(data_path,"metadata.pkl")
    laidataset_meta = load_dict(laidataset_meta_path)

    snp_pos = laidataset_meta["pos_snps"]
    snp_ref = laidataset_meta["ref_snps"]
    snp_alt = laidataset_meta["alt_snps"]
    pop_order = laidataset_meta["num_to_pop"]
    pop_list = []
    for i in range(len(pop_order.keys())):
        pop_list.append(pop_order[i])
    pop_order = np.array(pop_list)

    A = len(pop_order)
    C = len(snp_pos)
    M = int(round(window_size_cM*(C/(100*laidataset_meta["morgans"]))))

    meta = {
        "A": A, # number of ancestry
        "C": C, # chm length
        "M": M, # window size in SNPs
        "snp_pos": snp_pos,
        "snp_ref": snp_ref,
        "snp_alt":snp_alt,
        "pop_order": pop_order
    }

    # ------------------ Process data ------------------

    def read(split):

        paths = [os.path.join(data_path,split,"gen_"+str(gen)) for gen in generations[split]]
        X_files = [p + "/mat_vcf_2d.npy" for p in paths]
        labels_files = [p + "/mat_map.npy" for p in paths]
        X_raw, labels_raw = [load_np_data(f) for f in [X_files, labels_files]]
        X, y = data_process(X_raw, labels_raw, M)
        return X, y

    X_t1, y_t1 = read("train1")
    X_t2, y_t2 = read("train2")
    X_v, y_v = (None, None)
    if generations.get("val") is not None:
        X_v, y_v = read("val")

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v))

    return data, meta

def train_model(config, data_path, verbose):

    # data_path contains - train1/, train2/, val/, metadata, sample_maps/

    rm_simulated_data=config["simulation"]["rm_data"]
    model_name=config["model"].get("name", "model")
    inference=config["model"].get("inference", "default")
    window_size_cM=config["model"].get("window_size_cM")
    smooth_window_size=config["model"].get("smooth_size")
    n_cores=config["model"].get("n_cores", None)
    retrain_base=config["model"].get("retrain_base")
    calibrate=config["model"].get("calibrate")
    context_ratio=config["model"].get("context_ratio")
    chm = base_args["chm"]

    # option to bypass validation
    ratios = config["simulation"]["splits"]["ratios"]
    validate = True if ratios.get("val") else False

    generations = config["simulation"]["splits"]["gens"]
    if validate == False:
        del generations["val"]

    output_path = base_args["output_basename"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Either load pre-trained model or simulate data from reference file, init model and train it
    # Processing data
    if verbose:
        print("Reading data...")
    data, meta = get_data(data_path, generations, window_size_cM)

    # init model
    model = Gnomix(C=meta["C"], M=meta["M"], A=meta["A"], S=smooth_window_size,
                    snp_pos=meta["snp_pos"], snp_ref=meta["snp_ref"], snp_alt=meta["snp_alt"],
                    population_order=meta["pop_order"],
                    mode=inference, calibrate=calibrate,
                    n_jobs=n_cores, context_ratio=context_ratio, seed=config["seed"])

    # train it
    if verbose:
        print("Building model...")
    model.train(data=data, retrain_base=retrain_base, evaluate=True, verbose=verbose)
    # write gentic map df
    model.write_gen_map_df(load_dict(os.path.join(data_path,"gen_map_df.pkl")))

    # store it
    model_repo = join_paths(output_path, "models", verb=False)
    model_repo = join_paths(model_repo, model_name + "_chm_" + str(chm), verb=False)
    model_path = model_repo + "/" + model_name + "_chm_" + str(chm) + ".pkl"
    pickle.dump(model, open(model_path,"wb"))

    # brief analysis
    if verbose:
        print("Analyzing model performance...")
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    cm_path = analysis_path+"/confusion_matrix_{}.txt"
    cm_plot_path = analysis_path+"/confusion_matrix_{}_normalized.png"
    analysis_sets = ["train", "val"] if validate else ["train"]
    for d in analysis_sets:
        cm, idx = model.Confusion_Matrices[d]
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
        splits_to_rem = ["train1","train2","val"] if validate else ["train1","train2"]
        for split in splits_to_rem: # train1, train2, val (if val is there)
            chm_path = join_paths(data_path, split, verb=False)
            remove_data_cmd = "rm -r " + chm_path
            run_shell_cmd(remove_data_cmd, verbose=False)

    return model
    
def simulate_splits(base_args,config,data_path):

    # build LAIDataset object
    chm = base_args["chm"]
    reference = base_args["reference_file"]
    genetic_map = base_args["genetic_map_file"]
    sample_map = base_args["sample_map_file"]
    outdir = base_args["output_basename"]

    laidataset = LAIDataset(chm, reference, genetic_map, seed=config["seed"])
    laidataset.buildDataset(sample_map)

    # create output directories
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    sample_map_path = os.path.join(data_path,"sample_maps")
    if not os.path.exists(sample_map_path):
        os.makedirs(sample_map_path)

    # split sample map and write it.
    splits = config["simulation"]["splits"]["ratios"]
    if len(laidataset) <= 25:
        if splits.get("val"):
            print("WARNING: Too few samples to run validation.")
            del config["simulation"]["splits"]["ratios"]["val"]
    laidataset.create_splits(splits,sample_map_path)

    # TODO
    # write metadata into data_path/metadata.yaml
    # check lines 94 to 96 where this is read...
    save_dict(laidataset.metadata(), os.path.join(data_path,"metadata.pkl"))
    # Save genetic map df and store it inside model later after training
    gen_map_df = read_genetic_map(genetic_map, chm)
    save_dict(gen_map_df, os.path.join(data_path,"gen_map_df.pkl"))

    # get num_outs
    split_generations = config["simulation"]["splits"]["gens"]
    r_admixed = config["simulation"]["r_admixed"]
    num_outs = {}
    min_splits = {"train1":800,"train2":150,"val":50}
    for split in splits:
        total_sim = max(len(laidataset.return_split(split))*r_admixed, min_splits[split])
        num_outs[split] = int(total_sim/len(split_generations[split]))

    if verbose:
        print("Running Simulation...")
    for split in splits:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for gen in split_generations[split]:
            laidataset.simulate(num_outs[split],
                                split=split,
                                gen=gen,
                                outdir=os.path.join(split_path,"gen_"+str(gen)),
                                return_out=False)

    return


if __name__ == "__main__":
    
    print("...")

    # Citation
    print("-"*80+"\n"+"-"*35+"  Gnomix  "+"-"*35 +"\n"+"-"*80)
    print(CLAIMER)
    print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

    # Infer mode from number of arguments
    mode = None
    if len(sys.argv) == 6:
        mode = "pre-trained" 
    if len(sys.argv) == 8 or len(sys.argv) == 9:
        mode = "train"

    # Usage message
    if mode is None:
        if len(sys.argv) > 1:
            print("Error: Incorrect number of arguments.")
        print("Usage when training a model from scratch:")
        print("   $ python3 gnomix.py <query_file> <output_basename> <chr_nr> <phase> <genetic_map_file> <reference_file> <sample_map_file>")
        print("Usage when using a pre-trained model:")
        print("   $ python3 gnomix.py <query_file> <output_basename> <chr_nr> <phase> <path_to_model>")
        sys.exit(0)

    # Deconstruct CL arguments
    base_args = {
        'mode': mode,
        'query_file': sys.argv[1] if sys.argv[1].strip() != "None" else None,
        'output_basename': sys.argv[2],
        'chm': sys.argv[3],
        'phase': True if sys.argv[4].lower() == "true" else False
    }

    if not os.path.exists(base_args["output_basename"]):
        os.makedirs(base_args["output_basename"])

    base_args["config_file"] = "./config.yaml"
    if mode == "train":
        base_args["genetic_map_file"] = sys.argv[5]
        base_args["reference_file"]  = sys.argv[6]
        base_args["sample_map_file"] = sys.argv[7]
        if len(sys.argv) == 9:
            base_args["config_file"] = sys.argv[8]
    elif mode == "pre-trained":
        base_args["path_to_model"] = sys.argv[5]

    with open(base_args["config_file"],"r") as file:
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    if mode == "pre-trained":
        print("Launching in pre-trained mode...")
        model = load_model(base_args["path_to_model"], verbose=True)

        # Update changable model parameters for this particular execution
        model.n_cores = config["model"].get("n_cores", None)
        model.calibrate = config["model"].get("calibrate")
        model.smooth.calibrate = config["model"].get("calibrate")
        
        # TEMPORARY FOR BACKWARDS COMPATIBILITY FOR MODELS TRAINED BEFORE 10/2021
        model.base.vectorize = True

    else:
        print("Launching in training mode...")

        # process args here...
        verbose = config["verbose"]

        if config["simulation"]["splits"]["ratios"].get("val") == 0:
            del config["simulation"]["splits"]["ratios"]["val"]

        generations = config["simulation"]["gens"]
        gens_with_zero = list(set(generations + [0]))
        gens_without_zero = [generation for generation in generations if generation != 0]
        config["simulation"]["splits"]["gens"] = {
            "train1": gens_with_zero,
            "train2": generations,
            "val": gens_without_zero
        }

        # make sure data is ready...
        if config["simulation"]["run"]==False and config["simulation"]["path"] is not None:
            print("Using pre-simulated data from: ",config["simulation"]["path"])
            config["simulation"]["rm_data"] = False # this must be false if using pre-generated data regardless of what input is given for safety reasons!
            data_path = config["simulation"]["path"] # path with train1/ train2/ val/ metadata.yaml
        else:
            data_path = os.path.join(base_args["output_basename"],"generated_data")
            simulate_splits(base_args, config, data_path) # will create the simulation_output folder
        
        # train the model
        if verbose:
            print("Training...")
        model = train_model(config, data_path, verbose=verbose)
        
    # run inference if applicable.
    if base_args["query_file"]:

        print("Launching inference...")
        run_inference(base_args, model, 
                        visualize=config["inference"]["visualize_inference"],
                        snp_level=config["inference"]["snp_level_inference"],
                        bed_file_output=config["inference"]["bed_file_output"],
                        verbose=True)