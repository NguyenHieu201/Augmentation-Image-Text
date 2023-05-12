import time

import cv2
from matplotlib import pyplot as plt
import numpy as np

from core.Transform import Transform
from core.GaussianTransform import GaussianTransform
from core.CosineDistortTransform import CosineDistortTransform
from core.RandomOverlayTransform import RandomOverlayTransform
from core.PaddingTransform import PaddingTransform
from core.ElasticDistortionTransform import ElasticDistortionTransform
from core.JPEGNoiseTransform import JPEGNoiseTransform
from core.RandomRotationTransform import RandomRotationTransform
from core.ColorJitterTransform import ColorJitterTransform


if __name__ == "__main__":
    img = cv2.imread("./sample-image/20221212_143203_84.png")
    transform = Transform([ColorJitterTransform(), RandomRotationTransform()])
    # new_img = transform.transform(img)
    new_img = transform.random_apply_with_probs(img, probs=[0.1, 0.5])
    cv2.imwrite("./result-image/result.jpg", new_img)
