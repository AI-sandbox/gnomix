from sklearn.linear_model import LogisticRegression

from Utils.Base.base import Base

class LogisticRegressionBase(Base):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        # each model from the model factory has to have : fit, predict, predict_proba, classes_
        self.init_base_models(lambda : LogisticRegression(penalty="l2", C = 1., solver="liblinear", max_iter=1000))

    




