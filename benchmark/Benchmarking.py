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
sys.path.append('/home/arvindsk/XGMix/') # TODO
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

## the below 2 fucntion are called by get_gen_0

# def vcf2npy(vcf_file):
#     vcf_data = allel.read_vcf(vcf_file)
#     chm_len, nout, _ = vcf_data["calldata/GT"].shape
#     mat_vcf_2d = vcf_data["calldata/GT"].reshape(chm_len,nout*2).T
#     return mat_vcf_2d.astype('int16')

# def map2npy(map_file, shape, pop_order):
#     sample_map = pd.read_csv(map_file, sep="\t", header=None)
#     sample_map.columns = ["sample", "ancestry"]
#     y = np.zeros(shape, dtype='int16')
#     for i, a in enumerate(sample_map["ancestry"]):
#         a_numeric = np.where(a==pop_order)[0][0]
#         y[2*i:2*i+2] = a_numeric
#     return y

# def get_gen_0(data_path, sets):
#     gen_0_path = data_path + "/simulation_output"

#     population_map_file = data_path+"/populations.txt"
#     pop_order = np.genfromtxt(population_map_file, dtype="str")
    
#     out = []
#     for s in sets:
#         X_vcf = gen_0_path + "/"+s+"/founders.vcf"
#         y_map = gen_0_path + "/"+s+"/founders.map"
#         X_raw_gen_0 = vcf2npy(X_vcf)
#         y_raw_gen_0 = map2npy(y_map, X_raw_gen_0.shape, pop_order)
#         out.append(X_raw_gen_0)
#         out.append(y_raw_gen_0)
    
#     return out

# def get_gens(data_name):
#     if data_name == "6_even_anc":
#         gens = [2,4,6,8,12,16,24,32,48,64]
#     return gens

accr = lambda y, y_hat : round(accuracy_score(y.reshape(-1), y_hat.reshape(-1))*100,2)
def acc_per_gen(model, X_val, y_val, gens):
    set_size = (X_val.shape[0])//len(gens)
    accs = []
    for g, gen in enumerate(gens):
        X = X_val[g*set_size:(g+1)*set_size]
        y = y_val[g*set_size:(g+1)*set_size] 
        prediction = model.predict(X)
        acc = accr(y, prediction)
        accs.append(acc)

    return accs

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

#     if data == "6_even_anc":
#         data_path = "../Admixture/generated_data/6_even_anc_t2/chm22"
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

    X_train_raw = np.concatenate([X_train1_raw, X_train2_raw])
    labels_train_raw = np.concatenate([labels_train1_raw, labels_train2_raw])
    del X_train1_raw, X_train2_raw, X_val_raw, labels_train1_raw, labels_train2_raw, labels_val_raw

    X_train = np.concatenate([X_train1,X_train2])
    labels_window_train = np.concatenate([labels_window_train1,labels_window_train2])

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

    return data, meta

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
        # use np.nan for missing values
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


def bm_train(base, smooth, root, data_path, gens, chm, W=1000, load_base=True, load_smooth=True, eval=True, verbose=False):
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
    # gens = get_gens(data_name)
    data, meta = get_data(data_path, W=W, gens=gens, chm=chm, verbose=False)
    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data

    if not os.path.exists(root):
        os.makedirs(root)

    for b in base:
        
        print("BASE:", b)
                
        # load base model
        base_dir = os.path.join(root, "base_models")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        # base_data_dir = os.path.join(base_dir, data_name)
        # if not os.path.exists(base_data_dir):
        #     os.makedirs(base_data_dir)
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
            model = XGMIX(chmlen=meta["C"], win=meta["W"], num_anc=meta["A"], sws=None, base_model_generator=bmg)
            model._train_base(X_t1, y_t1)
            pickle.dump(model, open(base_model_path, "wb" ))
            
        for s in smooth:
            print("SMOOTH:", s)
            
            models_dir = os.path.join(root, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            # models_data_dir = os.path.join(models_dir, data_name)
            # if not os.path.exists(models_data_dir):
            #     os.makedirs(models_data_dir)
            model_name = b + "_" + s
            model_path = os.path.join(models_dir, model_name + ".pkl")
            
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
                pickle.dump(model, open(model_path, "wb" ))

            if eval:
                metrics[model_name] = bm_eval(model_path, data, gens=gens, verbose=verbose)

    return metrics

def bm_eval(model_path, data, gens=None, Xy_cal=None, verbose=False):

    # if metrics_path is None:
    #     metrics_path = model_path.split(".")[0] + "_metrics.pk"

    (X_t1, y_t1), (X_t2, y_t2), (X_v, y_v) = data

    model = pickle.load(open(model_path,"rb"))
    model._evaluate_base(X_t1, y_t1, X_v, y_v)
    model._evaluate_smooth(X_t2, y_t2, X_v, y_v)

    metrics = {}
        
    # Accuracy
    if verbose:
        print("retrieving accuracies..")
    
    metrics["train_acc"] = model.smooth_acc_train
    metrics["val_acc"]   = model.smooth_acc_val
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
    
    # generation vise performance
    if gens is not None:
        if verbose:
            print("estimating accuracy and log loss for each generation..")
            print("NOTE: this assumes that the validation set contains equal number from each generation in that order")
        gen_performance = {}
        gen_performance["gens"] = gens
        gen_performance["accs"] = acc_per_gen(model, X_v, y_v, gens)
        metrics["gen_performance"] = gen_performance

    if Xy_cal is not None:
        print("Calibration evaluation not implemented yet!")

    return metrics


           