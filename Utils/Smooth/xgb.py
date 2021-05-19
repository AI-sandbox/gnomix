import xgboost as xgb

from Utils.Smooth.smooth import Smoother
from Utils.Smooth.utils import slide_window

class XGB_Smoother(Smoother):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #params
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, reg_lambda=1, reg_alpha=0,
                                        nthread=self.n_jobs, random_state=self.seed, num_class=self.A)

    def process_base_proba(self,B,y=None):
        B_slide, y_slide = slide_window(B, self.S, y)
        return B_slide, y_slide
