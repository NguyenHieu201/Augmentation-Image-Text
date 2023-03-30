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


if __name__ == "__main__":

    img = cv2.imread("./sample-image/20221212_143203_84.png")
    transform = Transform([ElasticDistortionTransform(
        alpha=img.shape[0] * 2, sigma=img.shape[0] * 0.08, alpha_affine=img.shape[0] * 0.08)])
    new_img = transform.transform(img)
    cv2.imwrite("./result-image/result.jpg", new_img)
