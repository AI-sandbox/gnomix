import allel
from copy import deepcopy
from functools import partial
import numpy as np
import os
import pandas as pd
import pickle
import scipy
import sys
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, log_loss, accuracy_score, balanced_accuracy_score

# vis
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# local
sys.path.append('/home/arvindsk/XGMix_benchmark/') # TODO
from XGMIX import *
from Utils.preprocess import load_np_data, data_process
from Utils.visualization import plot_cm, plot_chm
from Utils.postprocess import read_vcf, write_msp_tsv
from Utils.crf import CRF
from Utils.conv_smoother import CONV

# models
from sklearn import ensemble
import lightgbm as lgb
import catboost as cb
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# --------------------------------- Utils ---------------------------------
def save_dict(D, path):
    with open(path, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as handle:
        return pickle.load(handle)

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

def get_snp_lvl_acc(y_snp, y_pred, W):
    """
    Evaluate snp level accuracy by comparing each snp with it's associated label
    W is window size
    """
    N, C = y_snp.shape
    y_pred_snp = np.zeros_like(y_snp)
    rem_len = C - (C//W)*W
    for i in range(N):
        y_pred_snp[i] = np.concatenate(( np.repeat(y_pred[i,:-1],W), np.repeat(y_pred[i,-1],W+rem_len) )) # fixed a bug here

    snp_lvl_acc = accr(y_snp, y_pred_snp)
    return snp_lvl_acc

def acc_per_gen_snp_lvl(y_snp, y_pred, gens, W):
    set_size = (y_snp.shape[0])//len(gens)
    accs = []
    for g, gen in enumerate(gens):
        y_g = y_snp[g*set_size:(g+1)*set_size]
        y_pred_g = y_pred[g*set_size:(g+1)*set_size]
        acc = get_snp_lvl_acc(y_g, y_pred_g,W)
        accs.append(acc)

    return accs


def eval_cal_acc(model, data):
    
    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data

    # calibrate on t1 (biased input but larga set and unseen founders)
    model.calibrate = True

    # train and apply calibrator
    zs = model.predict_proba(X_t1,rtn_calibrated=False).reshape(-1,model.num_anc)
    model.calibrator = calibrator_module(zs, y_t1.reshape(-1), model.num_anc, method ='Isotonic')  
    model._evaluate_smooth(X_t2,y_t2,X_v,y_v)

    # evaluate accuracy
    val_acc_cal = model.smooth_acc_val

    # evaluate log loss
    val_probs_cal = model.predict_proba(X_v)
    val_ll_cal = log_loss(y_v.reshape(-1), val_probs_cal.reshape(-1, model.num_anc))

    model.calibrate = False
    
    return val_acc_cal, val_ll_cal, val_probs_cal

def get_data(data_path, W, gens, chm, verbose=False):
    """
    input
    - W is window size in SNPs (int)
    - gens is a list of generations

    returns 
    - data: tuple of train1, train2 and independent test set on the form
            ( (X_t1,y_t1),(X_t2,y_t2),(X_v,y_v) )
    - meta: dict with meta information including:
            A: number of ancestries
            W: window size (in SNPs)
            C: chm size (in SNPs)
    """
    test_gens = [_ for _ in gens if _!= 0] # without 0

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
    X_train1, labels_window_train1 = data_process(X_train1_raw, labels_train1_raw, W, 0)
    X_train2, labels_window_train2 = data_process(X_train2_raw, labels_train2_raw, W, 0)
    X_val, labels_window_val       = data_process(X_val_raw, labels_val_raw, W, 0)

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
        "W": W                 # window size in SNPs
    }

    return data, meta, labels_val_raw

# --------------------------------- Models ---------------------------------

def get_base_model(model_name="xgb"):
    # tree params
    trees = 20; max_depth = 4; reg_lambda = 1
    reg_alpha = 0; lr = 0.1; random_state = 94305
    missing = 2; cores = 30

    # model
    if model_name == "xgb":
        model = xgb.XGBClassifier(n_estimators=trees,max_depth=max_depth,
                learning_rate=lr, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                nthread=cores, missing=missing, random_state=random_state)
    elif model_name == "rf":
        model = ensemble.RandomForestClassifier(n_estimators=trees,max_depth=max_depth,n_jobs=cores) 
    elif model_name == "lgb":
        model = lgb.LGBMClassifier(n_estimators=trees, max_depth=max_depth,
                    learning_rate=lr, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                    nthread=cores, random_state=random_state) 
    elif model_name == "cb":
        model = cb.CatBoostClassifier(n_estimators=trees, max_depth=max_depth,
                    learning_rate=lr, reg_lambda=reg_lambda,
                    thread_count=cores, verbose=0)
    elif model_name == "svm":
        model = svm.SVC(C=100., gamma=0.001, probability=True)
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=1)
    elif model_name == "logreg":
        model = LogisticRegression(penalty="l1", C = 1., solver="liblinear", max_iter=1000)
    elif model_name == "lda":
        model = LDA()
    else:
        print("Don't recognize the model")

    return model

def get_smoother(smoother_name, num_anc, default_sws=75):

    sws = default_sws
    
    if smoother_name == "xgb":
        smoother = xgb.XGBClassifier(n_estimators=100,max_depth=4, learning_rate=0.1,
                                            reg_lambda=1, reg_alpha=0, nthread=16, 
                                            random_state=1)
    elif smoother_name == "logreg":
        smoother = LogisticRegression(penalty="l2", C = 0.01, solver="lbfgs", max_iter=1000)
    elif smoother_name == "crf":
        smoother = CRF()
        sws = None
    elif smoother_name == "cnv":
        smoother = CONV(num_anc=num_anc, sws=sws)
        sws = None
    else:
        print("smoother not recognized")
    
    return smoother, sws

# --------------------------------- Train&Eval ---------------------------------


def bm_train(base, smooth, root, data_path, gens, chm, W=1000, load_base=True, load_smooth=True, eval=True, models_exist=None, verbose=False):
    """
    data is a string referring to some dataset

    returns metrics for that model

    base: list
    smooth: list
    root: tip: keep it same as data_path so the expts fall under the data
    data_path: must end with generated_data/
    gens = [0,2,4, etc...]

    """
    metrics = {}

    if verbose:
        print("Reading data...")
    data, meta, y_val_snp = get_data(data_path, W=W, gens=gens, chm=chm, verbose=False)
    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data
    test_gens = [_ for _ in gens if _!= 0] # without 0

    if not os.path.exists(root):
        os.makedirs(root)

    for b in base:
        
        print("BASE:", b)

        if models_exist:
            print("Skipping base...")
        else:
            # load base model
            base_dir = os.path.join(root, "base_models")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            base_model_path = os.path.join(base_dir, b + ".pkl")
            
            if load_base:
                if os.path.exists(base_model_path):
                    print(base_model_path)
                    model = pickle.load(open(base_model_path,"rb"))
                else:
                    print("Trained base not found, performing training...")
                    load_base=False

            if not load_base:
                # train and evaluate base, save 
                bmg = partial(get_base_model, b)
                # adding context_ratio.
                model = XGMIX(chmlen=meta["C"], win=meta["W"], num_anc=meta["A"], sws=None, base_model_generator=bmg,
                              context_ratio=0.5)
                model._train_base(X_t1, y_t1)
                pickle.dump(model, open(base_model_path, "wb" ))
            
        for s in smooth:
            print("SMOOTH:", s)
            
            models_dir = os.path.join(root, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            model_name = b + "_" + s
            model_path = os.path.join(models_dir, model_name + ".pkl")
            print(model_path)
            
            if load_smooth:
                if os.path.exists(model_path):
                    model = pickle.load(open(model_path,"rb"))
                else:
                    print("Trained model not found, performing smoother training...")
                    load_smooth = False

            if not load_smooth:
                # train and evaluate smoother, save
                smoother, sws = get_smoother(s, num_anc=meta["A"])
                model.smooth = smoother
                model.sws = sws
                model._train_smooth(X_t2, y_t2)
                # retrain base...
                # merge X_t1 and X_t2, y_t1 and y_t2
                train, train_lab = np.concatenate([X_t1, X_t2]), np.concatenate([y_t1, y_t2])
                model._train_base(train, train_lab)
                pickle.dump(model, open(model_path, "wb" ))

            if eval:
                metrics[model_name] = bm_eval(model_path, data, gens=test_gens, verbose=verbose)

    return metrics

def bm_eval(model_path, data, gens=None, eval_calibration=True, y_snp=None, verbose=False):

    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data

    model = pickle.load(open(model_path,"rb"))
    model._evaluate_base(X_t1, y_t1, X_v, y_v)
    model._evaluate_smooth(X_t2, y_t2, X_v, y_v)

    metrics = {}
        
    # Accuracy
    if verbose:
        print("retrieving accuracies..")
    
    # Base
    metrics["base_train_acc"]     = model.base_acc_train
    metrics["base_val_acc"]       = model.base_acc_val
    metrics["base_train_acc_bal"] = model.base_acc_train_balanced
    metrics["base_val_acc_bal"]   = model.base_acc_val_balanced

    # Smooth
    metrics["train_acc"]     = model.smooth_acc_train
    metrics["val_acc"]       = model.smooth_acc_val
    metrics["train_acc_bal"] = model.smooth_acc_train_balanced
    metrics["val_acc_bal"]   = model.smooth_acc_val_balanced

    # log loss
    if verbose:
        print("calculating log loss..")
    proba = model.predict_proba(X_v)
    ll = log_loss(y_v.reshape(-1), proba.reshape(-1, model.num_anc))
    metrics["log_loss"] = np.round(ll,2)

    # train time
    if verbose:
        print("retrieving training and inference time..")   
    metrics["smooth_training_time"]  = model.smooth_training_time
    metrics["smooth_inference_time"] = model.smooth_inference_time
    if model.base_training_time is not None:
        metrics["training_time"]  = model.base_training_time  + model.smooth_training_time
    if model.base_inference_time is not None:
        metrics["inference_time"] = model.base_inference_time + model.smooth_inference_time

    # model size
    if verbose:
        print("estimating model size..")    
    metrics["model_total_size_mb"] = round(os.stat(model_path).st_size/(10**6),2)
    
    y_val_pred = model.predict(X_v)

    # generation vise performance
    if gens is not None:
        if verbose:
            print("estimating accuracy for each generation..")
        assert 0 not in gens, "Shouldn't be evaluating generation 0!"
        gen_performance = {}
        gen_performance["gens"] = gens
        gen_performance["accs"] = acc_per_gen(y_v, y_val_pred, gens)
        gen_performance["accs_snp_lvl"] = acc_per_gen_snp_lvl(y_snp, y_val_pred, gens, W=model.win)
        metrics["gen_performance"] = gen_performance

    # snp level accuracy
    if y_snp is not None:
        metrics["val_acc_snp_lvl"] = get_snp_lvl_acc(y_snp, y_val_pred, W=model.win)

    # evaluate the performance with calibrated data
    if eval_calibration:
        val_acc_cal, val_ll_cal, _ = eval_cal_acc(model,data)
        metrics["val_acc_cal"] = val_acc_cal
        metrics["val_ll_cal"]  = val_ll_cal
        
    return metrics


           