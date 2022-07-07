from abc import ABC, abstractmethod

class SmootherModel(ABC):
    def __init__(self,config):
        pass

    @abstractmethod
    def train(self,B,y):
        pass

    @abstractmethod
    def predict_proba(self,B):
        pass
