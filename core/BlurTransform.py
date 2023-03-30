import cv2
import numpy as np

from core.BaseTransform import BaseTransform


class BlurTransform(BaseTransform):
    def __init__(self, kernel: tuple(int)) -> None:
        super().__init__()
        self.kernel = kernel

    def transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.blur(img, self.kernel)
