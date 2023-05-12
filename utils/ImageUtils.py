import numpy as np
from PIL import Image, ImageEnhance
import cv2


class ImageUtils(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def single2uint(img):
        return np.uint8((img.clip(0, 1) * 255.0).round())

    @staticmethod
    def uint2single(img):
        return np.float32(img / 255.0)

    @staticmethod
    def adjust_brightness(img, brightness_factor):
        """Adjust brightness of an Image.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.
        Returns:
            numpy ndarray: Brightness adjusted image.
        """
        table = (
            np.array([i * brightness_factor for i in range(0, 256)])
            .clip(0, 255)
            .astype("uint8")
        )
        # same thing but a bit slower
        # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        if img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)

    @staticmethod
    def adjust_contrast(img, contrast_factor):
        """Adjust contrast of an mage.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.
        Returns:
            numpy ndarray: Contrast adjusted image.
        """
        # much faster to use the LUT construction than anything else I've tried
        # it's because you have to change dtypes multiple times

        # input is RGB
        if img.ndim > 2 and img.shape[2] == 3:
            mean_value = round(cv2.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0])
        elif img.ndim == 2:
            # grayscale input
            mean_value = round(cv2.mean(img)[0])
        else:
            # multichannel input
            mean_value = round(np.mean(img))

        table = (
            np.array(
                [(i - mean_value) * contrast_factor + mean_value for i in range(0, 256)]
            )
            .clip(0, 255)
            .astype("uint8")
        )
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(contrast_factor)
        if img.ndim == 2 or img.shape[2] == 1:
            return cv2.LUT(img, table)[:, :, np.newaxis]
        else:
            return cv2.LUT(img, table)

    @staticmethod
    def adjust_saturation(img, saturation_factor):
        """Adjust color saturation of an image.
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            saturation_factor (float):  How much to adjust the saturation. 0 will
                give a black and white image, 1 will give the original image while
                2 will enhance the saturation by a factor of 2.
        Returns:
            numpy ndarray: Saturation adjusted image.
        """
        # ~10ms slower than PIL!
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return np.array(img)

    def adjust_hue(img, hue_factor):
        """Adjust hue of an image.
        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.
        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.
        See `Hue`_ for more details.
        .. _Hue: https://en.wikipedia.org/wiki/Hue
        Args:
            img (numpy ndarray): numpy ndarray to be adjusted.
            hue_factor (float):  How much to shift the hue channel. Should be in
                [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                HSV space in positive and negative direction respectively.
                0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                with complementary colors while 0 gives the original image.
        Returns:
            numpy ndarray: Hue adjusted image.
        """
        # After testing, found that OpenCV calculates the Hue in a call to
        # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

        # This function takes 160ms! should be avoided
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError("hue_factor is not in [-0.5, 0.5].".format(hue_factor))
        img = Image.fromarray(img)
        input_mode = img.mode
        if input_mode in {"L", "1", "I", "F"}:
            return np.array(img)

        h, s, v = img.convert("HSV").split()

        np_h = np.array(h, dtype=np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            np_h += np.uint8(hue_factor * 255)
        h = Image.fromarray(np_h, "L")

        img = Image.merge("HSV", (h, s, v)).convert(input_mode)
        return np.array(img)
