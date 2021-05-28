import numpy as np
from src.Smooth.utils import mode_filter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.Smooth.Calibration import Calibrator
from time import time

class Smoother():

    def __init__(self, n_windows, num_ancestry, smooth_window_size=75, model=None,
                calibrate=None, n_jobs=None, seed=None, mode_filter=0, verbose=False):

        self.W = n_windows
        self.A = num_ancestry
        self.S = smooth_window_size if smooth_window_size %2 else smooth_window_size-1
        self.model = model
        self.calibrate = calibrate
        self.calibrator = None
        self.mode_filter = mode_filter
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.time = {}

    def process_base_proba(self,B,y=None):
        return B, y # smoother doesn't pre-process by default

    def train(self,B,y):

        t = time()

        B_s, y_s = self.process_base_proba(B, y)
        self.model.fit(B_s, y_s)
        
        self.time["train"] = time() - t

    def predict_proba(self, B):

        t = time()

        B_s, _ = self.process_base_proba(B)

        # TODO:
        # if phase:
        #     X_phased, y_phased = self.phase(tt, calibrate=rtn_calibrated)
        #     return y_phased

        proba = self.model.predict_proba(B_s)
        
        if self.calibrate:
            if self.calibrator is None:
                print("No calibrator found, returning original probabilities.")
            else:
                proba = self.calibrator.transform(proba)
                
        self.time["inference"] = time() - t

        return proba.reshape(-1, self.W, self.A)

    def predict(self, B):

        proba = self.predict_proba(B)
        y_pred =  np.argmax(proba, axis=-1)

        if self.mode_filter != 0:
            y_pred = np.apply_along_axis(func1d=mode_filter, axis=1, arr=y_pred, size=self.mode_filter)
        return y_pred

    def evaluate(self,B=None,y=None,y_pred=None):

        round_accr = lambda accr : round(np.mean(accr)*100,2)

        if B is not None:
            y_pred = self.predict(B)
        elif y_pred is None:
            print("Error: Need either Base probabilities or y predictions.")

        accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
        accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

        return accr, accr_bal

    def train_calibrator(self, B, y, frac=0.05):

        # get pure probabilities
        calibrate = self.calibrate
        self.calibrate = False
        idxs = np.random.choice(len(B),int(frac*len(B)),replace=False)
        proba = self.predict_proba(B[idxs]).reshape(-1,self.A)
        self.calibrate = calibrate

        # calibrate
        self.calibrator = Calibrator(self.A)
        self.calibrator.fit(proba, y[idxs].reshape(-1))

