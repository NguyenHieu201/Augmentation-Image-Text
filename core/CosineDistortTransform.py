import numpy as np

from .BaseTransform import BaseTransform


class CosineDistortTransform(BaseTransform):
    def __init__(self, orientation="hor", x_scale=0.05, y_scale=5.0) -> None:
        super().__init__()
        self.orientation = orientation == "ver"
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.func = np.sin

    def _shift(self, t):
        return int(self.y_scale * self.func(np.pi * t * self.x_scale))

    # TODO: improve time of function: now ~ O(N) with image: N x N
    def transform(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-1] == 3:
            for c in range(3):
                for i in range(img.shape[self.orientation]):
                    if self.orientation:
                        img[:, i, c] = np.roll(img[:, i, c], self._shift(i))
                    else:
                        img[i, :, c] = np.roll(img[i, :, c], self._shift(i))
        else:
            raise NotImplementedError()
        return img
