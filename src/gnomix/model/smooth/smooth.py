from pyexpat import model
from gnomix.model.smooth.smoothmodel import SmootherModel
import numpy as np
# TODO: Moved to a separate script
# from sklearn.metrics import accuracy_score, balanced_accuracy_score
# NOTE: Deprecated
# from gnomix.model.smooth.utils import mode_filter
# from gnomix.model.smooth.Calibration import Calibrator
# TODO: Moved the timing logic to caller script
# from time import time


# When a new type is added, please add the key and logic to initialize it here
_VALID_TYPES = ["xgb","crf","cnn"]

class Smoother():

    def __init__(self,
                 n_windows: int,
                 num_ancestry: int, 
                 smooth_window_size: int, 
                 model_type: str,
                 n_jobs:int=32,
                 seed: int=94305,
                 verbose:bool=False):

        self.W = n_windows
        self.A = num_ancestry
        self.S = smooth_window_size if smooth_window_size %2 else smooth_window_size-1
        assert self.W >= 2*self.S, "Smoother size to large for given window size. "

        model = None
        if model_type == "xgb":
            from gnomix.model.smooth.xgb import XGB, XGBSmootherConfig
            xgb_config = XGBSmootherConfig(num_classes = self.A, 
                                           n_jobs=n_jobs,
                                           seed=seed,
                                           verbose=verbose)
            model = XGB(xgb_config)
        
        elif model_type == "crf":
            from gnomix.model.smooth.crf import CRF, CRFSmootherConfig
            crf_config = CRFSmootherConfig(verbose=verbose)
            model = CRF(crf_config)

        elif model_type == "cnn":
            from gnomix.model.smooth.cnn import CNN, CNNSmootherConfig
            cnn_config = CNNSmootherConfig(num_classes=self.A,
                                           num_features=self.S,
                                           verbose=verbose)
            model = CNN(cnn_config)

        else:
            print("Warning! smoother.model is None. Make sure to assign a smoother.model before proceeding")

        self.model = model

        ## Removed parameters (moved to caller function)
        # self.gnofix = False
        # self.n_jobs = n_jobs
        # self.seed = seed
        # self.verbose = verbose

        ## Deprecated parameters v1.0
        # self.mode_filter = mode_filter
        # self.calibrate = calibrate
        # self.calibrator = None


    # def process_base_proba(self,B,y=None):
    #     return B, y # smoother doesn't pre-process by default

    # This is for researching new smoother models
    @staticmethod
    def from_model(n_windows: int,
                  num_ancestry: int,
                  smooth_window_size: int,
                  model: SmootherModel):

        sm = Smoother(n_windows,num_ancestry,smooth_window_size,model_type="")
        assert type(model) == SmootherModel
        sm.model = model
        print("Smoother.model assigned using from_model constructor")
        return sm

    def train(self,B,y):

        assert len(np.unique(y)) == self.A, "Smoother training data does not include all populations"
        self.model.fit(B, y)
        

    def predict_proba(self, B):

        proba = self.model.predict_proba(B)
        return proba.reshape(-1, self.W, self.A)

    def predict(self, B):

        proba = self.predict_proba(B)
        y_pred = np.argmax(proba, axis=-1)
        return y_pred

    # def evaluate(self,B=None,y=None,y_pred=None):

    #     round_accr = lambda accr : round(np.mean(accr)*100,2)

    #     if B is not None:
    #         y_pred = self.predict(B)
    #     elif y_pred is None:
    #         print("Error: Need either Base probabilities or y predictions.")

    #     accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
    #     accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

    #     return accr, accr_bal