from random import choice

import cv2
import numpy as np

from .BaseTransform import BaseTransform


class PaddingTransform(BaseTransform):
    def __init__(self, W, H) -> None:
        super().__init__()
        self.W = W
        self.H = H

    def _available_paddding(self, height: int, width: int) -> bool:
        return (height < self.W) & (width < self.H)

    def transform(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        if self._available_paddding(height, width):
            new_img = np.full((self.H, self.W, 3), 0, dtype=np.uint8)
            h_pos, w_pos = choice(
                list(range(self.H - height))), choice(list(range(self.W - width)))
            new_img[h_pos:h_pos+height, w_pos:w_pos+width] = img
            return new_img

        return img
