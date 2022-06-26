import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from multiprocessing import get_context
import tqdm

from typing import Any, Tuple, Callable
from numpy.typing import ArrayLike

class Base:

    X_dtype = np.int8
    y_dtype = np.int16
    proba_dtype = np.float32

    def __init__(
        self,
        C: int,
        M: int,
        W: int,
        A: int,
        missing_encoding: int = 2,
        context: int = 0,
        train_admix: bool = True,
        n_jobs: int = None,
        seed: int = 94305,
        verbose: bool = False,
        vectorize: bool = True,
        base_multithread: bool = False,
        log_inference: bool = False,
    ) -> None:

        self.C = C
        self.M = M
        self.W = W
        self.A = A
        self.rem = self.W * self.M - self.C
        assert self.rem >= 0
        self.missing_encoding=missing_encoding
        self.context = context 
        self.train_admix = train_admix
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose
        self.vectorize = vectorize
        self.base_multithread = base_multithread
        self.log_inference = log_inference

    def init_base_models(self, model_factory: Callable) -> None:
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

    @staticmethod
    def pad_with_reflection_along_first_axis(
        array: ArrayLike,
        pad_width: int
    ) -> ArrayLike:

        if pad_width == 0:
            return array

        pad_left = np.flip(array[:, 0:pad_width], axis=1)
        pad_right = np.flip(array[:, -pad_width:], axis=1)
        return np.concatenate([pad_left, array, pad_right], axis=1)

    @staticmethod
    def _train_base_model_(b, X: ArrayLike, y: ArrayLike) -> Any:
        return b.fit(X, y)

    @staticmethod
    def _predict_proba_base_model(b, X: ArrayLike) -> ArrayLike:
        return b.predict_proba(X)

    def preprocess(self, X: ArrayLike, y: ArrayLike = None) -> Tuple[ArrayLike, ArrayLike]:
        """
        inputs:
            - X: shape (N, C)
            - y: shape (N, W)
        outputs:
            - X: shape (W, N, M)
            - y: shape (W, N)
        """

        N_x, C = X.shape
        assert C == self.C, f"Mismatch in number of SNPs for model ({self.C}) and data ({C})"

        if y is not None:
            N_y, W = y.shape
            assert W == self.W, f"Mismatch in number of windows for model ({self.W}) and data ({W})"
            assert N_x == N_y, f"Mismatch in number of samples for sequences ({N_x}) and labels ({N_y})"

        N = N_x
        padding = np.zeros((N, self.rem))
        X = np.concatenate([X, padding], axis=1)

        if self.context != 0:
            X = self.pad_with_reflection_along_first_axis(X, pad_width=self.context)

        X_processed = np.zeros((self.W, N, self.M + 2 * self.context), dtype=self.X_dtype)
        start = self.context
        for w in range(self.W):
            X_processed[w] = X[:,start-self.context:start+self.context+self.M] # (N, M)
            start += self.M

        y_processed = np.swapaxes(y, 0, 1).astype(self.y_dtype) if y is not None else None

        return X_processed, y_processed

    def train(self, X, y):
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
            - y: np.array of shape (N, W) where N is sample size and C chm length
        """        
        X, y = self.preprocess(X, y)

        train_args = tuple(zip(self.models, X, y))

        log_iter = tqdm.tqdm(train_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)
        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                self.models = pool.starmap(self._train_base_model_, log_iter) 
        else:
            for base_model, X, y in log_iter:
                self._train_base_model_(base_model, X, y) 

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        inputs:
            - X: np.array of shape (N, C) where N is sample size and C chm length
        returns 
            - B: base probabilities of shape (N,W,A)
        """
        X, _ = self.preprocess(X)

        base_args = tuple(zip(self.models, X))

        if self.log_inference:
            base_args = tqdm.tqdm(base_args, total=self.W, bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}', position=0, leave=True)

        if self.base_multithread:
            with get_context("spawn").Pool(self.n_jobs) as pool:
                B = np.array(pool.starmap(self._predict_proba_base_model, base_args))
        else:
            B = np.array([self._predict_proba_base_model(b,X) for b, X in base_args])

        B = np.swapaxes(B, 0, 1)

        return B

    def predict(self, X: ArrayLike) -> ArrayLike:
        B = self.predict_proba(X)
        return np.argmax(B, axis=-1)
        
    def evaluate(self, X: ArrayLike = None, y: ArrayLike = None, B: ArrayLike = None) -> Tuple[float, float]:

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