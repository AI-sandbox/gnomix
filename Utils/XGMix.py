import numpy as np
import pickle
import sys
from time import time

from Utils.Base.models import LogisticRegressionBase, RandomStringKernelBase

from Utils.Smooth.xgb import XGB_Smoother
from Utils.Smooth.crf import CRF_Smoother
from Utils.Smooth.cnn import CNN_Smoother

from XGFix.XGFIX import XGFix

class XGMIX():

    def __init__(self, C, M, S, A,
                base=LogisticRegressionBase, smooth=XGB_Smoother, # base and smooth models
                snp_pos=None, snp_ref=None, population_order=None, missing_encoding=2, # dataset specific, TODO: store in one object
                n_jobs=None, path=None, # configs
                calibrate=False, context_ratio=0.0, mode_filter=False, # hyperparams
                seed=94305, verbose=False
    ):
        """
        Inputs
           C: chromosome length in SNPs
           M: number of windows for chromosome segmentation
           S: Smoother window size: number of windows considered for a smoother 
                - (TODO: S is only a parameter for some smoother types, should we only have it defined there locally?)
           A: number of ancestry considered
        """

        self.C = C
        self.M = M
        self.A = A
        self.W = self.C//self.M # number of windows

        # configs
        self.path = path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # data
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.population_order = population_order

        # gnomix hyperparams
        self.context = int(self.M*context_ratio)
        self.calibrate = calibrate
  
        self.base = base(chm_len=self.C, window_size=self.M, num_ancestry=self.A,
                            missing_encoding=missing_encoding, n_jobs=self.n_jobs,
                            seed=self.seed, verbose=self.verbose)

        self.smooth = smooth(n_windows=self.W, smooth_window_size=S, num_ancestry=self.A,
                            n_jobs=self.n_jobs, calibrate=self.calibrate, mode_filter=mode_filter, 
                            seed=self.seed, verbose=self.verbose)
        
        # model stats
        self.time = {}
        self.accuracies = {}

    def save(self):
        if self.path is not None:
            pickle.dump(self, open(self.path+"model.pkl", "wb"))

    def train(self,data,retrain_base=True,evaluate=True,verbose=True):

        train_time_begin = time()

        (X_t1,y_t1), (X_t2,y_t2), (X_v,y_v) = data

        # TODO: this should be done somewhere upstream
        X_t1, X_t2, X_v = [np.array(data).astype("int8")  for data in [X_t1, X_t2, X_v]]
        y_t1, y_t2, y_v = [np.array(data).astype("int16") for data in [y_t1, y_t2, y_v]]
        
        if verbose:
            print("Training base models...")
        self.base.train(X_t1, y_t1)

        if verbose:
            print("Training smoother...")
        B_t2 = self.base.predict_proba(X_t2)
        self.smooth.train(B_t2,y_t2)

        if self.calibrate:
            # calibrates the predictions to be balanced w.r.t. the train1 class distribution
            if verbose:
                print("Calibrating...")
            B_t1 = self.base.predict_proba(X_t1)
            self.smooth.train_calibrator(B_t1, y_t1)

        # Evaluate model
        if evaluate:
            if verbose:
                print("Evaluating model...")

            Acc = {}
            Acc["base_train_acc"], Acc["base_train_acc_bal"] = self.base.evaluate(X_t1, y_t1)
            Acc["base_val_acc"], Acc["base_val_acc_bal"] = self.base.evaluate(X_v, y_v)
            
            B_v = self.base.predict_proba(X_v)
            Acc["smooth_train_acc"], Acc["smooth_train_acc_bal"] = self.smooth.evaluate(B_t2, y_t2)
            Acc["smooth_val_acc"], Acc["smooth_val_acc_bal"] = self.smooth.evaluate(B_v, y_v)
            
            self.accuracies = Acc

        if retrain_base:
            # Store both training data in one np.array for memory efficency
            X_t, y_t = np.concatenate([X_t1, X_t2, X_v]), np.concatenate([y_t1, y_t2, y_v])
            del X_t1, X_t2, y_t1, y_t2, X_v, y_v

            # Re-using all the data to re-train the base models
            if verbose:
                print("Re-training base models...")
            self.base.train(X_t, y_t)

        self.save()

        self.time["training"] = round(time() - train_time_begin,2)

    def predict(self,X,phase=False):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict(B)
        # TODO: GNOFIX
        return y_pred

    def predict_proba(self,X,phase=False):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict_proba(B)
        # TODO: GNOFIX
        return y_pred

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
        Y_phased = np.zeros((n_ind,2,self.W), dtype=int)

        for i, X_i in enumerate(X.reshape(n_ind,2,n_snp)):
            sys.stdout.write("\rPhasing individual %i/%i" % (i+1, n_ind))
            X_m, X_p = np.copy(X_i)
            base_prob, _ = self._get_smooth_data(X_i, flat_base=True)
            X_m, X_p, Y_m, Y_p, history, XGFix_tracker = XGFix(X_m, X_p, base_prob=base_prob, smoother=self.smooth, verbose=verbose)
            X_phased[i] = np.copy(np.array((X_m,X_p)))
            Y_phased[i] = np.copy(np.array((Y_m,Y_p)))

        print()
        
        return X_phased.reshape(n_haplo, n_snp), Y_phased.reshape(n_haplo, self.W)

