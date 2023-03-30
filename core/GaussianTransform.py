import numpy as np

from .BaseTransform import BaseTransform


class GaussianTransform(BaseTransform):
    def __init__(self, noise: int) -> None:
        """

        """
        super().__init__()
        self.noise_std = noise

    def transform(self, img: np.ndarray) -> np.ndarray:
        gasusian_noise = np.random.normal(
            loc=0, scale=self.noise_std, size=(img.shape))

        transform_img = img + gasusian_noise * 255
        transform_img = np.clip(transform_img, 0, 255)
        return transform_img.astype(np.uint8)
