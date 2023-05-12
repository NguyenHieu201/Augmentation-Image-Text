import numpy as np
import random

from core.BaseTransform import BaseTransform
from utils.ImageUtils import ImageUtils


class ColorJitterTransform(BaseTransform):
    def __init__(
        self,
        brightness: float = 0.5,
        constrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.1,
    ) -> None:
        super().__init__()
        self.brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        self.constrast_factor = random.uniform(max(0, 1 - constrast), 1 + constrast)
        self.saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        self.hue_factor = random.uniform(-hue, hue)

    def transform(self, img: np.ndarray) -> np.ndarray:
        result = ImageUtils.adjust_brightness(img, self.brightness_factor)
        result = ImageUtils.adjust_contrast(result, self.constrast_factor)
        result = ImageUtils.adjust_saturation(result, self.saturation_factor)
        result = ImageUtils.adjust_hue(result, self.hue_factor)
        return result
