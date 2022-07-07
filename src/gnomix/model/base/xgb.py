from dataclasses import dataclass
from xgboost import XGBClassifier
from gnomix.model.base.basemodel import SingleBase

@dataclass
class XGBBaseConfig:
    seed:int=94305
    missing_encoding:int=2

class XGBBase(SingleBase):

    def __init__(self,  config: XGBBaseConfig):
        self.config = config


    def model_factory(self):
        return XGBClassifier(n_estimators=20, 
                             max_depth=4, 
                             learning_rate=0.1, 
                             reg_lambda=1, 
                             reg_alpha=0,
                             thread=1, 
                             missing=self.config.missing_encoding, 
                             random_state=self.config.seed)

    def is_multithreaded(self):
        # It is multithreaded and n_jobs is always 1
        return True
