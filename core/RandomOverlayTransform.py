import os
from random import choice, choices

import numpy as np
from numpy.random import default_rng
import cv2
import skimage.exposure

from .BaseTransform import BaseTransform


class RandomOverlayTransform(BaseTransform):
    _DIRNAME = os.path.dirname(__file__)
    _FOLDER = os.path.join(_DIRNAME, "..", "background")
    _BACKGROUNDS = os.listdir(_FOLDER)

    def __init__(self, alpha=1, beta=0.2) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def _random_pattern(self, height: int, width: int) -> np.ndarray:
        rng = default_rng(seed=choice(list(range(1000))))
        noise = rng.integers(0, 255, (height, width), np.uint8, True)
        blur = cv2.GaussianBlur(
            noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT
        )
        stretch = skimage.exposure.rescale_intensity(
            blur, in_range="image", out_range=(0, 255)
        ).astype(np.uint8)
        thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask, mask, mask])
        return mask

    def transform(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        mask = self._random_pattern(height, width)
        return cv2.add(img, mask)
