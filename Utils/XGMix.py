import argparse
import gzip
import logging
import numpy as np
import os
import pandas as pd
import pickle
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import sys
from time import time
import xgboost as xgb

from Utils.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, cM2nsnp
from Utils.preprocess import load_np_data, data_process, get_gen_0
from Utils.visualization import plot_cm, CM
from Utils.Calibration import calibrator_module, apply_calibration
from Admixture.Admixture import read_sample_map, split_sample_map, main_admixture

from XGFix.XGFIX import XGFix

class XGMIX():

    def __init__(self,chmlen,win,sws,num_anc,snp_pos=None,snp_ref=None,population_order=None,save=None,
                base_model_generator=None, smoother=None,
                mode_filter_size=False,calibrate=False,context_ratio=0.0,
                
                # deprecated
                reg_lambda=None,reg_alpha=None, lr=None, cores=None, # 1, 0, 0.1, 16
                base_params=None, smooth_params=None): # [20,4], [100,4]

        if base_params is not None:
            print("'base_params' argument is deprecated, please use 'base_model' to specify the full base model")
        if smooth_params is not None:
            print("'smooth_params' argument is deprecated, please use 'smoother' to specify the full smoother model")

        self.chmlen = chmlen
        self.win = win
        self.save = save
        if sws is not None:
            self.sws = sws if sws %2 else sws-1
        else:
            self.sws = None
        self.context = int(self.win*context_ratio)
        
        self.num_anc = num_anc
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.population_order = population_order
        self.missing = 2
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.mode_filter_size = mode_filter_size
        self.calibrate = calibrate
        self.flat_base = sws is None

        self.cores = cores

        self.num_windows = self.chmlen//self.win

        self.base = {}
        self.smooth = None
        self.calibrator = None

        # model stats
        self.training_time = None
        self.base_training_time = None
        self.base_inference_time = None
        self.smooth_training_time = None
        self.smooth_inference_time = None
        self.base_acc_train = None
        self.base_acc_val = None
        self.smooth_acc_train = None
        self.smooth_acc_val = None
        self.base_acc_train_balanced = None
        self.base_acc_val_balanced = None
        self.smooth_acc_train_balanced= None
        self.smooth_acc_val_balanced = None

        # Base model
        def xgb_generator():
            return xgb.XGBClassifier(n_estimators=20,max_depth=4, learning_rate=0.1,
                                    reg_lambda=1, reg_alpha=0, nthread=16,
                                    missing=self.missing, random_state=1,
                                    num_class=self.num_anc)

        if base_model_generator is not None:
            self.base_model_generator = base_model_generator
        else:
            self.base_model_generator = xgb_generator

        # Smoother
        if smoother is not None:
            self.smooth = smoother
        else:
            self.smooth = xgb.XGBClassifier(n_estimators=100,max_depth=4, learning_rate=0.1,
                                            reg_lambda=1, reg_alpha=0, nthread=16, 
                                            random_state=1, num_class=self.num_anc)
            


    def _train_base(self,train,train_lab,evaluate=True,base_model_generator=None):

        time_begin = time()

        self.base = {}
        if self.context != 0.0:
            pad_left = np.flip(train[:,0:self.context],axis=1)
            pad_right = np.flip(train[:,-self.context:],axis=1)
            train = np.concatenate([pad_left,train,pad_right],axis=1)

        start = self.context

        for idx in range(self.num_windows):

            # a particular window across all examples
            
            tt = train[:,start-self.context:start+self.context+self.win]
            
            ll_t = train_lab[:,idx]

            if idx == self.num_windows-1:
                tt = train[:,start-self.context:]
                
            start += self.win
            # fit model
            if base_model_generator is None:
                model = self.base_model_generator()
            else:
                model = base_model_generator()
                if idx == 0:
                    print("model:", model)
            model.fit(tt,ll_t)
            self.base["model"+str(idx*self.win)] = model

            sys.stdout.write("\rWindows done: %i/%i" % (idx+1, self.num_windows))
        
        print("")
        self.base_training_time = round(time()-time_begin,2)

    def _get_smooth_data(self, data=None, labels=None, base_out=None, flat_base=False, return_base_out=False):

        if return_base_out:
            flat_base = True
            print("return_base_out argument is depricated and will soon be unavailable, use flat_base instead")
        
        if self.sws is None:
            flat_base = True

        time_begin = time()

        if base_out is None:
            n_ind = data.shape[0]

            # get base output
            base_out = np.zeros((data.shape[0],len(self.base),self.num_anc),dtype="float32")
            start = self.context
            
            if self.context != 0.0:
                pad_left = np.flip(data[:,0:self.context],axis=1)
                pad_right = np.flip(data[:,-self.context:],axis=1)
                data = np.concatenate([pad_left,data,pad_right],axis=1)
            
            for i in range(len(self.base)):
                inp = data[:,start-self.context:start+self.context+self.win]
                if i == len(self.base)-1:
                    inp = data[:,start-self.context:]
                start += self.win
                base_model = self.base["model"+str(i*self.win)]
                base_out[:,i,base_model.classes_] = base_model.predict_proba(inp)
    
            if flat_base:
                self.base_inference_time = round(time()-time_begin,2)
                return base_out, labels
                
            del data, inp, base_model
            
        else:
            n_ind = base_out.shape[0]

        # pad it.
        pad_size = (1+self.sws)//2
        pad_left = np.flip(base_out[:,0:pad_size,:],axis=1)
        pad_right = np.flip(base_out[:,-pad_size:,:],axis=1)
        base_out_padded = np.concatenate([pad_left,base_out,pad_right],axis=1)
        del base_out

        # window it.
        windowed_data = np.zeros((n_ind,len(self.base),self.num_anc*self.sws),dtype="float32")
        for ppl,dat in enumerate(base_out_padded):
            for win in range(windowed_data.shape[1]):
                windowed_data[ppl,win,:] = dat[win:win+self.sws].ravel()

        # reshape
        windowed_data = windowed_data.reshape(-1,windowed_data.shape[2])
        windowed_labels = None if labels is None else labels.reshape(-1)

        self.base_inference_time = round(time()-time_begin,2)

        return windowed_data, windowed_labels


    def _train_smooth(self,train,train_lab,smoother=None,flat_base=False,verbose=True):

        time_begin = time()

        self.flat_base = flat_base

        if smoother is not None:
            self.smooth = smoother
        
        tt, ttl = self._get_smooth_data(train,train_lab, flat_base=self.flat_base)
        self.smooth.fit(tt,ttl)

        self.smooth_training_time = round(time()-time_begin,2)


    def _evaluate_base(self,train,train_lab,val,val_lab,verbose=True):
        
        train_accr, val_accr, train_accr_bal, val_accr_bal = [], [], [], []
        
        start = self.context

        if self.context != 0.0:
            pad_left = np.flip(train[:,0:self.context],axis=1)
            pad_right = np.flip(train[:,-self.context:],axis=1)
            train = np.concatenate([pad_left,train,pad_right],axis=1)
            
            pad_left = np.flip(val[:,0:self.context],axis=1)
            pad_right = np.flip(val[:,-self.context:],axis=1)
            val = np.concatenate([pad_left,val,pad_right],axis=1)
        
        for idx in range(self.num_windows):

            model = self.base["model"+str(idx*self.win)]

            # a particular window across all examples
            tt = train[:,start-self.context:start+self.context+self.win]
            vt = val[:,start-self.context:start+self.context+self.win]
            ll_t = train_lab[:,idx]
            ll_v = val_lab[:,idx]

            if idx == self.num_windows-1:
                tt = train[:,start-self.context:]
                vt = val[:,start-self.context:]
                
            start += self.win
            
            y_pred = model.predict(tt)
            train_accr.append( accuracy_score(ll_t, y_pred) )
            train_accr_bal.append( balanced_accuracy_score(ll_t, y_pred) )

            y_pred = model.predict(vt)
            val_accr.append( accuracy_score(ll_v, y_pred) )
            val_accr_bal.append( balanced_accuracy_score(ll_v, y_pred) )

        round_accr = lambda accr : round(np.mean(accr)*100,2)
        self.base_acc_train = round_accr(train_accr)
        self.base_acc_train_balanced = round_accr(train_accr_bal)
        self.base_acc_val  = round_accr(val_accr)
        self.base_acc_val_balanced = round_accr(val_accr_bal)

        if verbose:
            print("Base Training Accuracy:   {}%".format(self.base_acc_train))
            print("Base Validation Accuracy: {}%".format(self.base_acc_val))

    def _evaluate_smooth(self,train,train_lab,val,val_lab,verbose=True):

        t_pred = self.predict(train)
        v_pred = self.predict(val)

        accr = lambda y, y_hat : round(accuracy_score(y.reshape(-1), y_hat.reshape(-1))*100,2)
        baccr = lambda y, y_hat : round(balanced_accuracy_score(y.reshape(-1), y_hat.reshape(-1))*100,2)

        self.smooth_acc_train = accr(train_lab, t_pred)
        self.smooth_acc_val   = accr(val_lab, v_pred)
        self.smooth_acc_train_balanced = baccr(train_lab, t_pred)
        self.smooth_acc_val_balanced   = baccr(val_lab, v_pred)

        if verbose:
            print("Smooth Training Accuracy: {}%".format(self.smooth_acc_train))
            print("Smooth Validation Accuracy: {}%".format(self.smooth_acc_val))

    def train(self,train1,train1_lab,train2,train2_lab,val,val_lab,
             retrain_base=True,evaluate=True,verbose=True):

        train_time_begin = time()

        train1, train2, val = [np.array(data).astype("int8") for data in [train1, train2, val]]
        train1_lab, train2_lab, val_lab = [np.array(data).astype("int16") for data in [train1_lab, train2_lab, val_lab]]

        # Store both training data in one np.array for memory efficency
        train_split_idx = int(len(train1))
        train, train_lab = np.concatenate([train1, train2]), np.concatenate([train1_lab, train2_lab])
        del train1, train2, train1_lab, train2_lab
        
        if verbose:
            print("Training base models...")
        self._train_base(train[:train_split_idx], train_lab[:train_split_idx])

        if verbose:
            print("Training smoother...")
        self._train_smooth(train[train_split_idx:], train_lab[train_split_idx:])

        if retrain_base:
            # Re-using the smoother-training-data to re-train the base models
            if verbose:
                print("Re-training base models...")
            self._train_base(train, train_lab)

        if self.calibrate:
            # calibrates the predictions to be balanced w.r.t. the train1 class distribution
            if verbose:
                print("Calibrating...")
            calibrate_light = int(0.05*train_split_idx)
            calibrate_idxs = np.random.choice(train_split_idx,calibrate_light,replace=False)
            zs = self.predict_proba(train[calibrate_idxs],rtn_calibrated=False).reshape(-1,self.num_anc)
            self.calibrator = calibrator_module(zs, train_lab[calibrate_idxs].reshape(-1), self.num_anc, method ='Isotonic')        
            del zs 

        # Evaluate model
        if evaluate:
            if verbose:
                print("Evaluating model...")
            self._evaluate_base(train[:train_split_idx], train_lab[:train_split_idx],   val,val_lab)
            self._evaluate_smooth(train[train_split_idx:], train_lab[train_split_idx:], val,val_lab)

        # Save model
        if self.save is not None:
            pickle.dump(self, open(self.save+"model.pkl", "wb" ))

        self.training_time = round(time() - train_time_begin,2)

    def _mode(self, arr):
        mode = stats.mode(arr)[0][0]
        if mode == -stats.mode(-arr)[0][0]:
            return mode # if mode is unambiguous
        else:
            return arr[len(arr)//2] # else return the center (default value)

    def _mode_filter(self, pred, size):
        if not size:
            return pred # if size is 0 or None
        pred_out = np.copy(pred)
        ends = size//2
        for i in range(len(pred))[ends:-ends]:
            pred_out[i] = self._mode(pred[i-ends:i+ends+1])
        
        return pred_out

    def predict(self,tt,rtn_calibrated=None,phase=False):
        if phase:
            X_phased, y_phased = self.phase(tt, calibrate=rtn_calibrated)
            return y_phased
        if rtn_calibrated is None:
            rtn_calibrated = self.calibrate
        if rtn_calibrated:
            y_cal_probs = self.predict_proba(tt,rtn_calibrated=True)
            y_preds = np.argmax(y_cal_probs, axis = 2)
        else:    
            n,_ = tt.shape
            tt,_ = self._get_smooth_data(tt, flat_base=self.flat_base)
            time_begin = time()
            y_preds = self.smooth.predict(tt).reshape(n,len(self.base))
            self.smooth_inference_time = round(time()-time_begin,2)
        
        if self.mode_filter_size:
            y_preds = np.apply_along_axis(func1d=self._mode_filter, axis=1, arr=y_preds, size=self.mode_filter_size)

        return y_preds

    def predict_proba(self,tt,rtn_calibrated=None):

        if rtn_calibrated is None:
            rtn_calibrated = self.calibrate

        n,_ = tt.shape
        tt,_ = self._get_smooth_data(tt, flat_base=self.flat_base)
        proba = self.smooth.predict_proba(tt).reshape(n,-1,self.num_anc)

        if rtn_calibrated:
            if self.calibrator is not None:
                proba = apply_calibration(proba, self.calibrator)
            else:
                print("No calibrator found, returning uncalibrated probabilities")

        return proba

    def write_config(self,fname):
        with open(fname,"w") as f:
            for attr in dir(self):
                val = getattr(self,attr)
                if type(val) in [int,float,str,bool,np.float64,np.float32,np.int]:
                    f.write("{}\t{}\n".format(attr,val))

    def phase(self,X,base=None,verbose=False):
        """
        Wrapper for XGFix
        """

        if self.smooth is None:
            print("Smoother is not trained, returning original haplotypes")
            return X, None

        n_haplo, n_snp = X.shape
        n_ind = n_haplo//2
        X_phased = np.zeros((n_ind,2,n_snp), dtype=int)
        Y_phased = np.zeros((n_ind,2,self.num_windows), dtype=int)

        for i, X_i in enumerate(X.reshape(n_ind,2,n_snp)):
            sys.stdout.write("\rPhasing individual %i/%i" % (i+1, n_ind))
            X_m, X_p = np.copy(X_i)
            base_prob, _ = self._get_smooth_data(X_i, flat_base=True)
            X_m, X_p, Y_m, Y_p, history, XGFix_tracker = XGFix(X_m, X_p, base_prob=base_prob, smoother=self.smooth, verbose=verbose)
            X_phased[i] = np.copy(np.array((X_m,X_p)))
            Y_phased[i] = np.copy(np.array((Y_m,Y_p)))

        print()
        
        return X_phased.reshape(n_haplo, n_snp), Y_phased.reshape(n_haplo, self.num_windows)

