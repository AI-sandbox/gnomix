
import click
import os
import numpy as np
from gnomix.core.utils import join_paths, read_vcf, vcf_to_npy, npy_to_vcf, update_vcf 
from gnomix.core.utils import read_genetic_map, save_dict, load_dict, read_headers, load_model
from gnomix.core.postprocess import get_meta_data, write_msp, write_fb, msp_to_lai, msp_to_bed
from gnomix.core.visualization import plot_cm, plot_chm

def run_inference(query_file, output_path, model_path, verbose, inference_config):

    if verbose:
        print("Loading and processing query file...")

    model = load_model(model_path, verbose)

    snp_level = inference_config.get("snp_level_inference",False)
    bed_file_output = inference_config.get("bed_file_output",False)
    visualize = inference_config.get("visualize_inference",False)

    phase = inference_config.get("phase",False)

    # TODO: Make sure model and query file have same chromosome
    chm = model.chm
    gen_map_df = model.gen_map_df

    # Load and process user query vcf file
    query_vcf_data = read_vcf(query_file, chm=chm, fields="*")
    X_query, vcf_idx, fmt_idx = vcf_to_npy(query_vcf_data, model.snp_pos, model.snp_ref, return_idx=True, verbose=verbose)

    # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
    if verbose:
        print("Inferring ancestry on query data...")

    B_query = model.base.predict_proba(X_query)
    if not phase:
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

def infer_gnomix(query_file, trained_model_path, verbose, inference_config):

    snp_level = inference_config.get("snp_level_inference",False)
    bed_file_output = inference_config.get("bed_file_output",False)
    visualize = inference_config.get("visualize_inference",False)

    

    return

def download_models():

    return
    