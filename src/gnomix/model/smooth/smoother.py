from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

class Smoother(ABC):

    @abstractmethod
    def train(self, B: ArrayLike, y: ArrayLike) -> None:
        """
        Takes in base probabilities and labels for each window and performs training.
        Inputs:
            - B: float of shape [N, W, A]
            - y: int of shape [N, W]
        """
        pass

    @abstractmethod
    def predict_proba(self, B: ArrayLike) -> ArrayLike:
        """
        Takes in base probabilities and estimates ancestry probabilites
        Inputs:
            - B: float of shape [N, W, A]
        Outputs:
            - y_proba: float of shape [N, W, A]
        """
        pass