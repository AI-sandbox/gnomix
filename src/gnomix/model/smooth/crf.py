from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import sklearn_crfsuite

from gnomix.model.smooth.smoothmodel import SmootherModel

@dataclass
class CRFSmootherConfig:
    max_it: int=10000
    verbose:bool = False
    

class CRF(SmootherModel):

    def __init__(self,config: CRFSmootherConfig):

        max_it=config.max_it

        self.CRF = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            max_iterations=max_it,
            all_possible_transitions=True,
            all_possible_states=True,
            verbose=config.verbose
        )

    def _npy2crf(self, X, Y=None):
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

    def _crf2npy(self, proba):
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
        X_CRF, y_CRF = self._npy2crf(X, y)
        self.CRF.fit(X_CRF, y_CRF)
        self.classes_ = self.CRF.classes_

    # def predict(self, X):
    #     X_CRF, _ = self._npy2crf(X)
    #     y_pred_CRF = self.CRF.predict(X_CRF)
    #     y_pred = np.array(y_pred_CRF, dtype=int)
    #     return y_pred

    def predict_proba(self, X):
        # extract probabilites
        X_CRF, _ = self._npy2crf(X)
        proba_CRF = self.CRF.predict_marginals(X_CRF)
        proba = self._crf2npy(proba_CRF)
        return proba