from src.model.smooth.smooth import Smoother

from src.model.smooth.utils import slide_window

from xgboost import XGBClassifier
from src.model.smooth.crf import CRF

class XGB_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnofix = True
        assert self.W >= 2*self.S, "Smoother size to large for given window size. "
        self.model = XGBClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, reg_lambda=1, reg_alpha=0,
            nthread=self.n_jobs, random_state=self.seed,
            num_class=self.A, 
            use_label_encoder=False, objective='multi:softprob', eval_metric="mlogloss"
        )

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
        
        from src.model.smooth.cnn import CNN # This is to avoid requiring the installation of pytorch

        self.model = CNN(num_classes=self.A, num_features=self.S, verbose=self.verbose)