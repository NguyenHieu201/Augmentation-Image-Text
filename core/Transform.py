import typing
from multiprocessing import Process, Pool

import numpy as np
from .BaseTransform import BaseTransform


class Transform(object):
    def __init__(self, methods: list[BaseTransform]) -> None:
        self.methods = methods

    def transform(self, img: np.ndarray) -> np.ndarray:
        transform_img = img.copy()
        for method in self.methods:
            transform_img = method.transform(transform_img)

        return transform_img

    def mp_transform(self, imgs: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
        # p = Process(target=self.transform, args=(img))
        pool = Pool(processes=batch_size)
        res = pool.map(self.transform, imgs)
        pool.close()
        return res

    def save(self, img, path):
        pass
