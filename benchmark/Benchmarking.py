import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, accuracy_score
import sys

# local
sys.path.append('../../gnomix')
from src.gnomix import Gnomix
from src.preprocess import load_np_data, data_process

# --------------------------------- Models ---------------------------------

from src.Base.models import LogisticRegressionBase, LGBMBase, RandomStringKernelBase, StringKernelBase
from src.Smooth.models import XGB_Smoother, CRF_Smoother, CNN_Smoother

Bases = {
    "lgb": LGBMBase,
    "logreg": LogisticRegressionBase,
    "randomstringkernel": RandomStringKernelBase,
    "stringkernel": StringKernelBase
}

Smoothers = {
    "xgb": XGB_Smoother,
    "crf": CRF_Smoother,
    "cnn": CNN_Smoother
}

# --------------------------------- Utils ---------------------------------
def save_dict(D, path):
    with open(path, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def get_data(data_path, M, gens, chm, verbose=False, only_founders=False):
    """
    input
    - M is window size in SNPs (int)
    - gens is a list of generations

    returns 
    - data: tuple of train1, train2 and independent test set on the form
            ( (X_t1,y_t1),(X_t2,y_t2),(X_v,y_v) )
    - meta: dict with meta information including:
            A: number of ancestries
            M: window size (in SNPs)
            C: chm size (in SNPs)
    """
    test_gens = [_ for _ in gens if _!= 0] # without 0

    if only_founders:
        gens, test_gens = [0], [0]

    # get paths - gen0 only for train1
    train1_paths = [data_path + "/chm{}/simulation_output/train1/gen_".format(chm) + str(gen) + "/" for gen in gens]
    train2_paths = [data_path + "/chm{}/simulation_output/train2/gen_".format(chm) + str(gen) + "/" for gen in test_gens]
    val_paths    = [data_path + "/chm{}/simulation_output/val/gen_".format(chm)    + str(gen) + "/" for gen in test_gens] 
    test_paths   = [data_path + "/chm{}/simulation_output/test/gen_".format(chm)   + str(gen) + "/" for gen in test_gens] 

    # gather feature data files
    X_fname = "mat_vcf_2d.npy"
    X_train1_files = [p + X_fname for p in train1_paths]
    X_train2_files = [p + X_fname for p in train2_paths]
    X_val_files    = [p + X_fname for p in val_paths]
    X_test_files   = [p + X_fname for p in test_paths]

    # gather label data files
    labels_fname = "mat_map.npy"
    labels_train1_files = [p + labels_fname for p in train1_paths]
    labels_train2_files = [p + labels_fname for p in train2_paths]
    labels_val_files    = [p + labels_fname for p in val_paths]
    labels_test_files   = [p + labels_fname for p in test_paths]

    # load the data
    train_val_files = [X_train1_files, labels_train1_files, X_train2_files, labels_train2_files, X_val_files, labels_val_files]
    X_train1_raw, labels_train1_raw, X_train2_raw, labels_train2_raw, X_val_raw, labels_val_raw = [load_np_data(f, verbose) for f in train_val_files]

    # reshape according to window size
    X_train1, labels_window_train1 = data_process(X_train1_raw, labels_train1_raw, M, 0)
    X_train2, labels_window_train2 = data_process(X_train2_raw, labels_train2_raw, M, 0)
    X_val, labels_window_val       = data_process(X_val_raw, labels_val_raw, M, 0)

    # for training and storing a pre-trained model
    population_map_file = data_path+"/populations.txt"
    position_map_file   = data_path+"/chm{}/positions.txt".format(chm)
    reference_map_file  = data_path+"/chm{}/references.txt".format(chm)

    snp_pos = np.loadtxt(position_map_file, delimiter='\n').astype("int")
    pop_order = np.genfromtxt(population_map_file, dtype="str")
    snp_ref = np.loadtxt(reference_map_file, delimiter='\n', dtype=str)
    
    train1, train2, val = [np.array(data).astype("int8") for data in [X_train1, X_train2, X_val]]
    train1_lab, train2_lab, val_lab = [np.array(data).astype("int16") for data in [labels_window_train1, labels_window_train2, labels_window_val]]

    data = ((train1, train1_lab), (train2, train2_lab), (val, val_lab))
    meta = {
        "A": len(pop_order),   # number of ancestry
        "C": train1.shape[-1], # chm length
        "M": M                 # window size in SNPs
    }

    return data, meta, labels_val_raw


accr = lambda y, y_hat : round(accuracy_score(y.reshape(-1), y_hat.reshape(-1))*100,2)

def acc_per_gen(y, y_pred, gens):
    set_size = (y.shape[0])//len(gens)
    accs = []
    for g, gen in enumerate(gens):
        y_g = y[g*set_size:(g+1)*set_size]
        y_pred_g = y_pred[g*set_size:(g+1)*set_size]
        acc = accr(y_g, y_pred_g)
        accs.append(acc)

    return accs

def get_snp_lvl_acc(y_snp, y_pred, M):
    """
    Evaluate snp level accuracy by comparing each snp with it's associated label
    M is window size
    """
    N, C = y_snp.shape
    y_pred_snp = np.zeros_like(y_snp)
    rem_len = C - (C//M)*M
    for i in range(N):
        y_pred_snp[i] = np.concatenate(( np.repeat(y_pred[i,:-1],M), np.repeat(y_pred[i,-1],M+rem_len) ))

    snp_lvl_acc = accr(y_snp, y_pred_snp)
    return snp_lvl_acc

def acc_per_gen_snp_lvl(y_snp, y_pred, gens, M):
    set_size = (y_snp.shape[0])//len(gens)
    accs = []
    for g, gen in enumerate(gens):
        y_g = y_snp[g*set_size:(g+1)*set_size]
        y_pred_g = y_pred[g*set_size:(g+1)*set_size]
        acc = get_snp_lvl_acc(y_g, y_pred_g,M)
        accs.append(acc)

    return accs

# --------------------------------- Train&Eval ---------------------------------

def bm_train(bases, smooth, model_path, data_path, gens, chm, M=1000, verbose=True):
    """
    data is a string referring to some dataset

    returns metrics for that model

    base: list
    smooth: list
    model_path: tip: keep it same structure as data_path 
    data_path: must end with generated_data/
    gens = [0,2,4, etc...]

    """
    if verbose:
        print("Reading data...")
    data, meta, _ = get_data(data_path, M=M, gens=gens, chm=chm, verbose=False, only_founders=False)
    (X_t1, y_t1), (X_t2, y_t2), _ = data
    
    data_founders, _, _ = get_data(data_path, M=M, gens=gens, chm=chm, verbose=False, only_founders=True)
    (X_t1_f, y_t1_f), (X_t2_f, y_t2_f), _ = data_founders
        
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    base_dir = os.path.join(model_path, "base_models")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    smooth_dir = os.path.join(model_path, "smoothers")
    if not os.path.exists(smooth_dir):
        os.makedirs(smooth_dir)

    for b in bases:
        if verbose:
            print("BASE:", b)

        base = Gnomix(C=meta["C"], M=meta["M"], A=meta["A"], base=Bases[b]).base

        # train
        if verbose:
            print("Training base")
        if base.train_admix:
            base.train(X_t1, y_t1)
        else:
            base.train(X_t1_f, y_t1_f)
        pickle.dump(base, open( os.path.join(base_dir, b + ".pkl") , "wb" ))

        # retrain
        if verbose:
            print("Re-training base")
        if base.train_admix:
            base.train( np.concatenate([X_t1, X_t2]), np.concatenate([y_t1, y_t2]) )
        else:
            base.train( np.concatenate([X_t1_f, X_t2_f]), np.concatenate([y_t1_f, y_t2_f]) )
        pickle.dump(base, open( os.path.join(base_dir, b + "_retrained.pkl") , "wb" ))


    if len(smooth) == 0:
        return

    if verbose:
        print("Now training smoothers")

    for b in bases:

        print("Fetchinig base probabilites from", b, "base")

        base = pickle.load( open(os.path.join(base_dir,b + ".pkl"),"rb") )
        B_t2 = base.predict_proba(X_t2)

        for s in smooth:

            smoother = Gnomix(C=meta["C"], M=meta["M"], A=meta["A"], base=Bases[b], smooth=Smoothers[s]).smooth
            # train
            if verbose:
                print("Training", s, "smoother")
            smoother.train(B_t2, y_t2)
            
            pickle.dump(smoother, open( os.path.join(smooth_dir, b + "_" + s + ".pkl") ,"wb"))

round_accr = lambda accr : round(np.mean(accr)*100,2)

def bm_eval(model, val_data, gens=None, y_snp=None, base_smooth_paths=None, B_v=None, verbose=False):

    metrics = {}

    X_v, y_v = val_data

    if verbose:
        print("retrieving validation estimates..")
    if B_v is None:
        B_v = model.base.predict_proba(X_v)
    y_v_pred_base = np.argmax(B_v, axis=-1)
    P_v = model.smooth.predict_proba(B_v)
    y_v_pred_smooth = np.argmax(P_v, axis=-1)

    # Accuracy
    if verbose:
        print("Evaluating Accuracy..")
    
    # Base
    metrics["base_val_acc"]       = round_accr( accuracy_score(y_v.reshape(-1), y_v_pred_base.reshape(-1)) )
    metrics["base_val_acc_bal"]   = round_accr( balanced_accuracy_score(y_v.reshape(-1), y_v_pred_base.reshape(-1)) )
    metrics["base_val_log_loss"]  = np.round( log_loss(y_v.reshape(-1), B_v.reshape(-1, model.A)) ,2)

    # Smooth
    metrics["smooth_val_acc"]      = round_accr( accuracy_score(y_v.reshape(-1), y_v_pred_smooth.reshape(-1)) )
    metrics["smooth_val_acc_bal"]  = round_accr( balanced_accuracy_score(y_v.reshape(-1), y_v_pred_smooth.reshape(-1)) )
    metrics["smooth_val_log_loss"] = np.round( log_loss(y_v.reshape(-1), P_v.reshape(-1, model.A)) ,2)

    if verbose:
        print("Logging times..")

    # time
    metrics["base_training_time"]  = model.base.time["train"] 
    metrics["base_inference_time"] = model.base.time["inference"] 
    metrics["smooth_training_time"]  = model.smooth.time["train"] 
    metrics["smooth_inference_time"] = model.smooth.time["inference"]
    metrics["training_time"]  = model.base.time["train"]     + model.smooth.time["train"] 
    metrics["inference_time"] = model.base.time["inference"] + model.smooth.time["inference"] 

    # model size
    if base_smooth_paths is not None:
        if verbose:
            print("Estimating model size..")
        metrics["model_total_size_mb"] = np.sum([round(os.stat(p).st_size/(10**6),2) for p in base_smooth_paths])

    # generation vise performance
    if gens is not None:
        if verbose:
            print("estimating accuracy for each generation..")
        assert 0 not in gens, "Shouldn't be evaluating generation 0!"
        gen_performance = {}
        gen_performance["gens"] = gens
        gen_performance["accs"] = acc_per_gen(y_v, y_v_pred_smooth, gens)
        if y_snp is not None:
            gen_performance["accs_snp_lvl"] = acc_per_gen_snp_lvl(y_snp, y_v_pred_smooth, gens, M=model.M)
        metrics["gen_performance"] = gen_performance

    # snp level accuracy
    if y_snp is not None:
        metrics["val_acc_snp_lvl"] = get_snp_lvl_acc(y_snp, y_v_pred_smooth, M=model.M)
        
    return metrics


           