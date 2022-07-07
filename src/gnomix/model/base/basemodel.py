from abc import ABC, abstractmethod

class SingleBase(ABC):
    def __init__(self,config):
        pass

    @abstractmethod
    def is_multithreaded(self):
        pass

    @abstractmethod
    def model_factory(self):
        pass
