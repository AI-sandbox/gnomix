from copy import deepcopy
import numpy as np

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.metrics import accuracy_score

class CRF:

    def __init__(self, solver="lbfgs", max_it=10000, verbose=False):
        self.CRF = sklearn_crfsuite.CRF(
            algorithm=solver, 
            max_iterations=max_it,
            all_possible_transitions=True,
            all_possible_states=True,
            verbose=verbose
        )
        

    def npy2crf(self, X, Y=None):
        """format data for linear-chain CRF"""
        N, B, A = X.shape
        X_out, Y_out = [], []
        for i in range(N):
            x, y = [], []
            for b in range(B):
                datapoint={}
                for a in range(A):
                    datapoint[str(a)]=X[i,b,a]
                x.append(datapoint)
                if Y is not None:
                    y.append(str(Y[i,b]))

            X_out.append(x)
            Y_out.append(y)
            
        return X_out, Y_out

    def crf2npy(self, proba):
        """format data from CRF probabilities to np probabilities"""
        N = len(proba)
        B = len(proba[0])
        A = len(proba[0][0])
        
        # level 2 dict to list
        if A > 1:
            proba = deepcopy(proba)
            for i in range(N):
                for b in range(B):
                    proba[i][b] = [proba[i][b][str(a)] for a in range(A)]

        return np.array(proba)

    def fit(self, X, y):
        X_CRF, y_CRF = self.npy2crf(X, y)
        self.CRF.fit(X_CRF, y_CRF)

    def predict(self, X):
        X_CRF, _ = self.npy2crf(X)
        y_pred_CRF = self.CRF.predict(X_CRF)
        y_pred = np.array(y_pred_CRF, dtype=int)
        return y_pred

    def predict_proba(self, X):
        # extract probabilites
        X_CRF, _ = self.npy2crf(X)
        proba_CRF = self.CRF.predict_marginals(X_CRF)
        proba = self.crf2npy(proba_CRF)
        return proba
    

    # # One hot encoder
    # def ohe(y, n_anc):
    #     N, B = y.shape
    #     out = np.zeros((N,B,n_anc), dtype=int)
    #     for i in range(N):
    #         for b in range(B):
    #             out[i,b,y[i,b]] = 1  
    #     return out