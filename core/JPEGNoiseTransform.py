import cv2
import numpy as np
import random

from core.BaseTransform import BaseTransform
from utils.ImageUtils import ImageUtils


class JPEGNoiseTransform(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, img: np.ndarray) -> np.ndarray:
        quality_factor = random.randint(30, 95)
        # img = cv2.cvtColor(ImageUtils.single2uint(img), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        )
        img = cv2.imdecode(encimg, 1)
        return img
