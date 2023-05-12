import typing
from multiprocessing import Process, Pool
import random

import numpy as np
from .BaseTransform import BaseTransform


class Transform(object):
    def __init__(self, methods: list[BaseTransform]) -> None:
        self.methods = methods

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Using all method to transform"""
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

    # TODO: implement save function
    def save(self, img, path):
        pass

    def random_select_transform(self, img: np.ndarray) -> np.ndarray:
        """Randomly select a method for transform"""
        transform_img = img.copy()
        selected_method = random.choice(self.methods)
        transform_img = selected_method.transform(transform_img)

        return transform_img

    def random_select_multiple_transform(
        self, img: np.ndarray, k: int = 2
    ) -> np.ndarray:
        """Randomly select k methods for transform"""
        transform_img = img.copy()
        methods = random.sample(self.methods, k)
        for method in methods:
            transform_img = method.transform(transform_img)
        return transform_img

    def random_apply_with_probs(
        self, img: np.ndarray, probs: list[float]
    ) -> np.ndarray:
        """Methods will be applied if binomial draw true
        Args:
            probs: list contain probability for each method occur
        """
        if len(self.methods) != len(probs):
            raise Exception("Give probs for each method")
        transform_img = img.copy()
        for i, method in enumerate(self.methods):
            prob = probs[i]
            is_apply = np.random.binomial(size=1, p=prob, n=1) == 1
            if is_apply:
                transform_img = method.transform(transform_img)
        return transform_img
