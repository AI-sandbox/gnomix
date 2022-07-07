from dataclasses import dataclass
from pickle import FALSE
from xgboost import XGBClassifier
import numpy as np

from gnomix.model.smooth.smoothmodel import SmootherModel

@dataclass
class XGBSmootherConfig:

    num_classes: int
    n_jobs: int = 32
    seed: int = 94305
    verbose: bool = False

def slide_window(B, S, y=None):
    """
    inputs
        - B (np.array: base probabilities of shape (N,W,A)
        - S (int): smoother window size 

    """
    N, W, A = B.shape

    # pad it.
    pad = (S+1)//2
    pad_left  = np.flip(B[:,0:pad,:],axis=1)
    pad_right = np.flip(B[:,-pad:,:],axis=1)
    B_padded = np.concatenate([pad_left,B,pad_right],axis=1)

    # window it.
    X_slide = np.zeros((N,W,A*S),dtype="float32")
    for ppl, dat in enumerate(B_padded):
        for w in range(W):
            X_slide[ppl,w,:] = dat[w:w+S].ravel()

    # reshape
    X_slide = X_slide.reshape(N*W,A*S)
    y_slide = None if y is None else y.reshape(N*W)

    return X_slide, y_slide

class XGB(SmootherModel):
    
    def __init__(self, config: XGBSmootherConfig):
        self.xgb_classifer = XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, reg_lambda=1, reg_alpha=0,
            nthread=config.n_jobs, random_state=config.seed,
            num_class=config.A, 
            use_label_encoder=False, objective='multi:softprob', eval_metric="mlogloss"
        )


    def train(self,B,y):
        B_s, y_s = self._process_base_proba(B, y)
        self.xgb_classifer.fit(B_s, y_s)

    def predict_proba(self, B):
        B_s, _ = self._process_base_proba(B)
        proba = self.xgb_classifer.predict_proba(B_s)
        return proba

    def _process_base_proba(self,B,y=None):
        B_slide, y_slide = slide_window(B, self.S, y)
        return B_slide, y_slide
