import numpy as np

class Smoother:

    def __init__(
        self,
        n_windows: int,
        num_ancestry: int,
        smooth_window_size: int = 75,
        model=None,
        n_jobs=None,
        seed=None,
        verbose=False
    ) -> None:

        self.W = n_windows
        self.A = num_ancestry
        self.S = smooth_window_size if smooth_window_size % 2 else smooth_window_size - 1
        self.model = model
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

    def process_base_proba(self,B,y=None):
        return B, y # smoother doesn't pre-process by default

    def train(self,B,y):

        assert len(np.unique(y)) == self.A, "Smoother training data does not include all populations"

        B_s, y_s = self.process_base_proba(B, y)
        self.model.fit(B_s, y_s)
        
    def predict_proba(self, B):

        B_s, _ = self.process_base_proba(B)
        proba = self.model.predict_proba(B_s)
        
        return proba.reshape(-1, self.W, self.A)

    def predict(self, B):

        proba = self.predict_proba(B)
        y_pred = np.argmax(proba, axis=-1)

        return y_pred

