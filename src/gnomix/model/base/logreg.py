
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from gnomix.model.base.basemodel import SingleBase


@dataclass
class LogRegBaseConfig:
    penalty="l2"

class LogReg(SingleBase):

    def __init__(self, config):
        self.config = config

    def is_multithreaded(self):
        return True

    def model_factory(self):
        return LogisticRegression(penalty=self.config.penalty, C = 3., solver="liblinear", max_iter=1000)


