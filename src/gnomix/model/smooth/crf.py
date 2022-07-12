from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
import sklearn_crfsuite
from typing import Dict, Tuple, Any, Optional

from gnomix.model.smooth.smoother import Smoother

class CRFSmoother(Smoother):

    def __init__(
        self,
        num_ancestry: int,
        crf_config: Optional[Dict[str, Any]] = {
            "algorithm": "lbfgs",
            "max_iterations": 10000,
            "all_possible_transitions": True,
            "all_possible_states": True
        },
        input_dtype: Optional[str] = "float32",
        proba_dtype: Optional[str] = "float32"
    ):

        self.CRF = sklearn_crfsuite.CRF(**crf_config)
        self.input_dtype = input_dtype
        self.proba_dtype = proba_dtype

    def train(self, X, y):
        X_CRF, y_CRF = self.npy2crf(X, y)
        self.CRF.fit(X_CRF, y_CRF)

    def predict_proba(self, X):
        X_CRF, _ = self.npy2crf(X)
        proba_CRF = self.CRF.predict_marginals(X_CRF)
        proba = self.crf2npy(proba_CRF)
        return proba

    def crf2npy(self, crf_proba):
        """format data from CRF probabilities to np probabilities"""
        N = len(crf_proba)
        W = len(crf_proba[0])
        A = len(crf_proba[0][0])
        
        proba = np.zeros((N, W, A), dtype=self.proba_dtype)
        for i in range(N):
            for w in range(W):
                for a, a_crf in enumerate(self.CRF.classes_):
                    proba[i, w, a] = crf_proba[i][w][a_crf]

        return proba

    @staticmethod
    def npy2crf(X, Y=None):
        """format data for linear-chain CRF"""
        N, W, A = X.shape
        X_out = [[{str(a): X[i, w, a] for a in range(A)} for w in range (W)] for i in range(N)]
        Y_out = [[str(Y[i, w]) for w in range(W)] for i in range(N)] if Y is not None else []
        return X_out, Y_out