import numpy as np
import sys
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class Base():

    def __init__(self, chm_len, window_size, num_ancestry, missing_encoding=2, context=0, n_jobs=None, seed=94305, verbose=False):

        self.C = chm_len
        self.M = window_size
        self.W = self.C//self.M # Number of windows
        self.A = num_ancestry
        self.missing_encoding=missing_encoding
        self.context = context
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

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
        self.models = {}
        for w in range(self.W):
            self.models["model"+str(w*self.M)] = model_factory()

    def train(self,X,y,verbose=True):
        
        if self.context != 0.0:
            pad_left = np.flip(X[:,0:self.context],axis=1)
            pad_right = np.flip(X[:,-self.context:],axis=1)
            X = np.concatenate([pad_left,X,pad_right],axis=1)

        start = self.context

        for i in range(self.W):

            X_w = X[:,start-self.context:start+self.context+self.M]
            y_w = y[:,i]

            if i == self.W-1:
                X_w = X[:,start-self.context:]

            # train model
            self.models["model"+str(i*self.M)].fit(X_w,y_w)

            start += self.M

            if verbose:
                sys.stdout.write("\rWindows done: %i/%i" % (i+1, self.W))
        
        if verbose:
            print("")


    def predict_proba(self, X):
        """
        returns 
            - B: base probabilities of shape (N,W,A)
        """

        N = len(X)
        B = np.zeros( (N, self.W, self.A), dtype="float32" )
        
        start = self.context
        
        if self.context != 0.0:
            pad_left = np.flip(X[:,0:self.context],axis=1)
            pad_right = np.flip(X[:,-self.context:],axis=1)
            X = np.concatenate([pad_left,X,pad_right],axis=1)
        
        for i in range(self.W):
            X_w = X[:,start-self.context:start+self.context+self.M]

            if i == self.W-1:
                X_w = X[:,start-self.context:]

            base_model = self.models["model"+str(i*self.M)]
            B[:,i,base_model.classes_] = base_model.predict_proba(X_w)

            start += self.M
            
        return B
    
    def predict(self, X):
        B = self.predict_proba(X)
        return np.argmax(B, axis=-1)

    def evaluate(self,X,y,verbose=True):

        round_accr = lambda accr : round(np.mean(accr)*100,2)

        y_pred = self.predict(X)
        accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
        accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

        return accr, accr_bal