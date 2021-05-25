from Utils.Smooth.smooth import Smoother

from Utils.Smooth.utils import slide_window

from xgboost import XGBClassifier
from Utils.Smooth.crf import CRF
from Utils.Smooth.cnn import CNN

class XGB_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = XGBClassifier(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                                    nthread=self.n_jobs, random_state=self.seed, num_class=self.A)

    def process_base_proba(self,B,y=None):
        B_slide, y_slide = slide_window(B, self.S, y)
        return B_slide, y_slide


class CRF_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CRF(verbose=self.verbose)


class CNN_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNN(num_classes=self.A, num_features=self.S, verbose=self.verbose)