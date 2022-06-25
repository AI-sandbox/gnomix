import numpy as np
import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from time import time
from multiprocessing import get_context
import tqdm

class Base():

    def __init__(self, chm_len, window_size, num_ancestry, missing_encoding=2,
                    context=0.5, train_admix=True, n_jobs=None, seed=94305, verbose=False):

        self.C = chm_len
        self.M = window_size
        self.W = self.C//self.M # Number of windows
        self.A = num_ancestry
        self.missing_encoding=missing_encoding
        self.context = context
        self.train_admix = train_admix
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.base_multithread = False
        self.log_inference = False
        self.vectorize = True

        self.time = {}

    def init_base_models(self, model_factory):
        """
        inputs:
            - model_factory: function that returns a model object that has the functions
                - fit
                - predict
                - predict_proba
            - and the attributes
                - classes_
        """
        self.models = [model_factory() for _ in range(self.W)]

    def pad(self,X):
        pad_left = np.flip(X[:,0:self.context],axis=1)
        pad_right = np.flip(X[:,-self.context:],axis=1)
        return np.concatenate([pad_left,X,pad_right],axis=1)
        
    def train(self, X, y, verbose=True):
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
            - y: np.array of shape (N, C) where N is sample size and C chm length
        """
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.train_vectorized(X, y)
            except AttributeError:
                print("Vectorized implementation requires numpy versions 1.20+.. Using loopy version..")
                self.vectorize = False
        if not self.vectorize:
            return self.train_loopy(X, y, verbose=verbose)

    def train_loopy(self, X, y, verbose=True):
        """Depricated"""

        t = time()

        if self.context != 0.0:
            X = self.pad(X)

        start = self.context

        for i in range(self.W):

            X_w = X[:,start-self.context:start+self.context+self.M]
            y_w = y[:,i]

            if i == self.W-1:
                X_w = X[:,start-self.context:]

            # train model
            self.models[i].fit(X_w,y_w)

            start += self.M

            if verbose:
                sys.stdout.write("\rWindows done: %i/%i" % (i+1, self.W))
        
        if verbose:
            print("")

        self.time["train"] = time() - t

    def train_base_model(self, b, X, y):
        return b.fit(X, y)

    def predict_proba_base_model(self, b, X):
        return b.predict_proba(X)

    def train_vectorized(self, X, y):

        slide_window = np.lib.stride_tricks.sliding_window_view

        t = time()

        # pad
        if self.context != 0.0:
            X = self.pad(X)
            
        # convolve
        M_ = self.M + 2*self.context        
        idx = np.arange(0,self.C,self.M)[:-2]
        X_b = slide_window(X, M_, axis=1)[:,idx,:]

        # stack
        train_args = tuple(zip( self.models[:-1], np.swapaxes(X_b,0,1), np.swapaxes(y,0,1)[:-1] ))
        rem = self.C - self.M*self.W
        train_args += ((self.models[-1], X[:,X.shape[1]-(M_+rem):], y[:,-1]),)

        # train
        log_iter = tqdm.tqdm(train_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)
        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                self.models = pool.starmap(self.train_base_model, log_iter) 
        else:
            self.models = [self.train_base_model(*b) for b in log_iter]

        self.time["train"] = time() - t

    def predict_proba(self, X):
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
        returns 
            - B: base probabilities of shape (N,W,A)
        """
        if self.vectorize:
            try:
                np.lib.stride_tricks.sliding_window_view
                return self.predict_proba_vectorized(X)
            except AttributeError:
                print("Vectorized implementation requires numpy versions 1.20+.. Using loopy version..")
                self.vectorize = False
        if not self.vectorize:
            return self.predict_proba_loopy(X)

    def predict_proba_vectorized(self, X):

        slide_window = np.lib.stride_tricks.sliding_window_view

        t = time()

        # pad
        if self.context != 0.0:
            X = self.pad(X)
            
        # convolve
        M_ = self.M + 2*self.context        
        idx = np.arange(0,self.C,self.M)[:-2]
        X_b = slide_window(X, M_, axis=1)[:,idx,:]

        # stack
        base_args = tuple(zip( self.models[:-1], np.swapaxes(X_b,0,1) ))
        rem = self.C - self.M*self.W
        base_args += ((self.models[-1], X[:,X.shape[1]-(M_+rem):]), )

        if self.log_inference:
            base_args = tqdm.tqdm(base_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)

        # predict proba
        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                B = np.array(pool.starmap(self.predict_proba_base_model, base_args))
        else:
            B = np.array([self.predict_proba_base_model(*b) for b in base_args])

        B = np.swapaxes(B, 0, 1)

        self.time["inference"] = time() - t

        return B


    def predict_proba_loopy(self, X):
        """Depricated"""

        t = time()

        N = len(X)
        B = np.zeros( (N, self.W, self.A), dtype="float32" )
        
        start = self.context
        
        if self.context != 0.0:
            X = self.pad(X)
        
        for i in range(self.W):
            X_w = X[:,start-self.context:start+self.context+self.M]

            if i == self.W-1:
                X_w = X[:,start-self.context:]

            B[:,i,self.models[i].classes_] = self.models[i].predict_proba(X_w)

            start += self.M

        self.time["inference"] = time() - t
            
        return B
    
    def predict(self, X):
        B = self.predict_proba(X)
        return np.argmax(B, axis=-1)
        
    def evaluate(self,X=None,y=None,B=None):

        round_accr = lambda accr : round(np.mean(accr)*100,2)

        if X is not None:
            y_pred = self.predict(X)
        elif B is not None:
            y_pred = np.argmax(B, axis=-1)
        else:
            print("Error: Need either SNP input or estimated probabilities to evaluate.")

        accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
        accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

        return accr, accr_bal