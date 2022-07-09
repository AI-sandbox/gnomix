import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import sys
from time import time
from copy import deepcopy

from gnomix.model.base.models import LogisticRegressionBase, CovRSKBase
from gnomix.model.smooth.models import XGB_Smoother
from gnomix.model.gnofix.gnofix import gnofix

class Gnomix:

    def __init__(self, C, M, A, S,
                base=None, smooth=None, mode="default", # base and smooth models
                snp_pos=None, snp_ref=None, snp_alt=None, population_order=None, missing_encoding=2, # dataset specific, TODO: store in one object
                n_jobs=None, path=None, # configs
                calibrate=False, context_ratio=0.5, mode_filter=False, # hyperparams
                seed=94305, verbose=False
    ):
        """
        Inputs
           C: chromosome length in SNPs
           M: number of windows for chromosome segmentation
           A: number of ancestry considered
        """

        self.C = C
        self.M = M
        self.W = int(np.ceil(self.C / self.M)) # number of windows
        self.A = A
        self.S = S

        # configs
        self.path = path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # data
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.snp_alt = snp_alt
        self.population_order = population_order

        # gnomix hyperparams
        self.context = int(self.M*context_ratio)
        self.calibrate = calibrate

        if base is None:
            if mode == "fast":
                base = LogisticRegressionBase
            elif mode == "best":
                base = CovRSKBase
            elif mode == "large":
                base = LogisticRegressionBase
            else:
                base = LogisticRegressionBase
            if verbose:
                print("Base models:", base)
        if smooth is None:
            if mode == "fast":
                from gnomix.model.smooth.models import CRF_Smoother # import here to avoid strict crf suite dependency
                smooth = CRF_Smoother 
            elif mode == "large":
                from gnomix.model.smooth.models import CNN_Smoother # import here to avoid strict torch dependency
                smooth = CNN_Smoother 
            elif mode=="best":
                smooth = XGB_Smoother
            else:
                smooth = XGB_Smoother
            if verbose:
                print("Smoother:", smooth)

        self.base = base(C=self.C, M=self.M, W=self.W, A=self.A,
                            missing_encoding=missing_encoding, context=self.context,
                            n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)

        self.smooth = smooth(n_windows=self.W, num_ancestry=self.A, smooth_window_size=self.S,
                            n_jobs=self.n_jobs, calibrate=self.calibrate, mode_filter=mode_filter, 
                            seed=self.seed, verbose=self.verbose)
        
        # model stats
        self.time = {}
        self.accuracies = {}

        # gen map df
        self.gen_map_df = {}

    def write_gen_map_df(self,gen_map_df):
        self.gen_map_df = deepcopy(gen_map_df)

    def conf_matrix(self, y, y_pred):

        cm = confusion_matrix(y.reshape(-1), y_pred.reshape(-1))
        indices = sorted(np.unique( np.concatenate((y.reshape(-1),y_pred.reshape(-1))) ))

        return cm, indices

    def save(self):
        if self.path is not None:
            pickle.dump(self, open(self.path+"model.pkl", "wb"))

    def train(self,data,retrain_base=True,evaluate=True,verbose=True):

        train_time_begin = time()

        (X_t1,y_t1), (X_t2,y_t2), (X_v,y_v) = data
        
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
                print("Fitting calibrator...")
            B_t1 = self.base.predict_proba(X_t1)
            self.smooth.train_calibrator(B_t1, y_t1)

        # Evaluate model
        if evaluate:
            if verbose:
                print("Evaluating model...")

            Acc = {}
            CM  = {}

            # training accuracy
            B_t1 = self.base.predict_proba(X_t1)
            y_t1_pred = self.smooth.predict(B_t1)
            y_t2_pred = self.smooth.predict(B_t2)
            Acc["base_train_acc"],   Acc["base_train_acc_bal"]   = self.base.evaluate(X=None,   y=y_t1, B=B_t1)
            Acc["smooth_train_acc"], Acc["smooth_train_acc_bal"] = self.smooth.evaluate(B=None, y=y_t2, y_pred=y_t2_pred)
            CM["train"] = self.conf_matrix(y=y_t1, y_pred=y_t1_pred)
            
            # val accuracy
            if X_v is not None:
                B_v = self.base.predict_proba(X_v)
                y_v_pred  = self.smooth.predict(B_v)
                Acc["base_val_acc"],     Acc["base_val_acc_bal"]     = self.base.evaluate(X=None,   y=y_v,  B=B_v )
                Acc["smooth_val_acc"],   Acc["smooth_val_acc_bal"]   = self.smooth.evaluate(B=None, y=y_v,  y_pred=y_v_pred )
                CM["val"] = self.conf_matrix(y=y_v, y_pred=y_v_pred)

            self.accuracies = Acc
            self.Confusion_Matrices = CM

        if retrain_base:
            # Store both training data in one np.array for memory efficency
            if X_v is not None:
                X_t, y_t = np.concatenate([X_t1, X_t2, X_v]), np.concatenate([y_t1, y_t2, y_v])
            else:
                X_t, y_t = np.concatenate([X_t1, X_t2]), np.concatenate([y_t1, y_t2])

            # Re-using all the data to re-train the base models
            if verbose:
                print("Re-training base models...")
            self.base.train(X_t, y_t)

        self.save()

        self.time["training"] = round(time() - train_time_begin,2)

    def predict(self,X):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict(B)
        return y_pred

    def predict_proba(self,X):

        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict_proba(B)
        return y_pred

    def write_config(self,fname):
        with open(fname,"w") as f:
            for attr in dir(self):
                val = getattr(self,attr)
                if type(val) in [int,float,str,bool,np.float64,np.float32,np.int]:
                    f.write("{}\t{}\n".format(attr,val))

    def phase(self,X,B=None,verbose=False):
        """
        Wrapper for XGFix
        """

        assert self.smooth is not None, "Smoother is not trained, returning original haplotypes"
        assert self.smooth.gnofix, "Type of Smoother ({}) does not currently support re-phasing".format(self.smooth)

        N, C = X.shape
        n = N//2
        X_phased = np.zeros((n,2,C), dtype=int)
        Y_phased = np.zeros((n,2,self.W), dtype=int)

        if B is None:
            B = self.base.predict_proba(X)
        B = B.reshape(n, 2, self.W, self.A)

        for i, X_i in enumerate(X.reshape(n,2,C)):
            sys.stdout.write("\rPhasing individual %i/%i" % (i+1, n))
            X_m, X_p = X_i
            X_m_, X_p_, Y_m_, Y_p_, history, tracker = gnofix(X_m, X_p, B=B[i], smoother=self.smooth, verbose=verbose)
            X_phased[i] = np.copy(np.array((X_m_,X_p_)))
            Y_phased[i] = np.copy(np.array((Y_m_,Y_p_)))

        print()
        
        return X_phased.reshape(N, C), Y_phased.reshape(N, self.W)

