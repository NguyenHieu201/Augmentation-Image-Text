from abc import ABC, abstractmethod
import numpy as np


class BaseTransform(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, img: np.ndarray) -> np.ndarray:
        pass
