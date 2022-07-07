import numpy as np
import pickle
import gzip
import os
from sklearn.metrics import confusion_matrix
import sys
from copy import deepcopy

from gnomix.model.base.base import Base
from gnomix.model.smooth.smooth import Smoother
from gnomix.model.gnofix.gnofix import gnofix

class Gnomix():

    def __init__(self, C, M, A, S,context_ratio=0.5, # calibrate would come here in the future
                 mode="default",n_jobs=32,seed=94305, verbose=False):
        """
        Inputs
           C: chromosome length in SNPs
           M: Window length in SNPs
           A: number of ancestry considered
           S: Smoother size
        Other:
           W: Window size
        """

        self.C = C
        self.M = M
        self.A = A
        self.S = S
        self.W = self.C//self.M # number of windows
        self.context = int(self.M*context_ratio)
        # self.calibrate = calibrate

        # configs for training and inference later
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        base_type = None
        smooth_type = None

        if mode == "fast":
            base_type = "logreg"
            smooth_type = "crf"

        elif mode == "best":
            base_type = "covrsk"
            smooth_type = "xgb"

        elif mode == "large":
            base_type = "logreg"
            smooth_type = "xgb"
        
        else:
            base_type = "logreg"
            smooth_type = "xgb"

        self.base = Base(chm_len=self.C, window_size=self.M, num_ancestry=self.A, context=self.context,
                         model_type=base_type, n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)

        self.smooth = Smoother(n_windows=self.W, num_ancestry=self.A, smooth_window_size=self.S,
                               model_type=smooth_type,n_jobs=self.n_jobs, seed=self.seed, verbose=self.verbose)
        


    ## Load model given path
    @staticmethod
    def load(path_to_model):
        if path_to_model[-3:]==".gz":
            with gzip.open(path_to_model, 'rb') as unzipped:
                model = pickle.load(unzipped)
        else:
            model = pickle.load(open(path_to_model,"rb"))

        return model
    
    
    ### Saving methods
    def save(self,path):
        with open(os.path.join(path,"model.pkl","wb")) as f:
            pickle.dump(self, f)

    # def write_config(self,fname):
    #     with open(fname,"w") as f:
    #         for attr in dir(self):
    #             val = getattr(self,attr)
    #             if type(val) in [int,float,str,bool,np.float64,np.float32,np.int]:
    #                 f.write("{}\t{}\n".format(attr,val))


    # Training methods
    def train_base(self,X_t, y_t):
        if self.verbose:
            print("Training base models")
        self.base.train(X_t, y_t)

    def train_smooth(self,X_t2,y_t2):
        if self.verbose:
            print("Training smoother")
        B_t2 = self.base.predict_proba(X_t2)
        self.smooth.train(B_t2,y_t2)

    def add_ladataset_metadata(self,laidataset):
        ### All the dataset dependent metadata are initialized at the time of seeing the data
        ## Chromosome
        self.chm = None
        ## Genotyping data
        self.snp_pos = None
        self.snp_cm = None
        self.snp_ref = None
        self.snp_alt = None
        ## Ancestry data
        self.population_order = None

    ## Evaluation methods
    def evaluate(self, X_v,y_v):
        # Evaluate model
        if self.verbose:
            print("Evaluating model...")

        Acc = {}
        CM  = {}

        # accuracy
        B_v = self.base.predict_proba(X_v)
        y_v_pred  = self.smooth.predict(B_v)
        Acc["base_val_acc"],     Acc["base_val_acc_bal"]     = self.base.evaluate(X=None,   y=y_v,  B=B_v )
        Acc["smooth_val_acc"],   Acc["smooth_val_acc_bal"]   = self.smooth.evaluate(B=None, y=y_v,  y_pred=y_v_pred )
        CM["val"] = self._conf_matrix(y=y_v, y_pred=y_v_pred)
        
        return Acc, CM

    ### Evaluate
    def _conf_matrix(self, y, y_pred):

        cm = confusion_matrix(y.reshape(-1), y_pred.reshape(-1))
        indices = sorted(np.unique( np.concatenate((y.reshape(-1),y_pred.reshape(-1))) ))

        return cm, indices

    ## Inference
    def predict(self,X):
        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict(B)
        return y_pred

    def predict_proba(self,X):
        B = self.base.predict_proba(X)
        y_pred = self.smooth.predict_proba(B)
        return y_pred

    ## Writing into files (import from inference)

    ## Visualizing (import from inference)

    ## Phasing
    def phase(self,X,B=None,verbose=False):
        """
        Wrapper for XGFix
        """

        assert self.smooth is not None, "Smoother is not trained, returning original haplotypes"

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

