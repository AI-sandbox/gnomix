import numpy as np
from Utils.Smooth.utils import mode_filter
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class Smoother():

    def __init__(self, n_windows, smooth_window_size, num_ancestry, model=None,
                calibrate=None, n_jobs=None, seed=None, mode_filter=0, verbose=False):

        self.S = (1+smooth_window_size)//2
        self.W = n_windows
        self.A = num_ancestry
        self.model = model
        self.calibrate = calibrate
        self.mode_filter = mode_filter
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

    def process_base_proba(self,B,y=None):
        return B, y # smoother doesn't pre-process by default

    def train(self,B,y):
        B_s, y_s = self.process_base_proba(B, y)
        self.model.fit(B_s, y_s)

    def predict_proba(self, B):

        B_s, _ = self.process_base_proba(B)

        # TODO:
        # if phase:
        #     X_phased, y_phased = self.phase(tt, calibrate=rtn_calibrated)
        #     return y_phased

        proba = self.model.predict_proba(B_s)
        
        if self.calibrate:
            proba = apply_calibration(proba, self.calibrator)

        return proba.reshape(-1, self.W, self.A)

    def predict(self, B):

        proba = self.predict_proba(B)
        y_pred =  np.argmax(proba, axis=-1)
        if self.mode_filter != 0:
            y_pred = np.apply_along_axis(func1d=mode_filter, axis=1, arr=y_pred, size=self.mode_filter)
        return y_pred

    def evaluate(self,B,y):

        round_accr = lambda accr : round(np.mean(accr)*100,2)

        y_pred = self.predict(B)
        accr = round_accr( accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )
        accr_bal = round_accr( balanced_accuracy_score(y.reshape(-1), y_pred.reshape(-1)) )

        return accr, accr_bal

    def train_calibrator(self, B, y, frac=0.05):
        idxs = np.random.choice(len(B),int(frac*len(B)),replace=False)
        proba = self.model.predict_proba(B[idxs],rtn_calibrated=False).reshape(-1,self.A)
        self.calibrator = calibrator_module(proba, y[idxs].reshape(-1), self.A, method ='Isotonic') 