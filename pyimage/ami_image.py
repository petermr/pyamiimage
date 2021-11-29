"""
Wrappers and other methods for image manipulation
"""
import numpy as np
from skimage import io, color, morphology
from skan.pre import threshold


class AmiImage:
    """
    instance and class methods for holding and converting images
    """
    def __init__(self):
        """
        for when we need to store state
        :return:
        """
        pass

# ========== legacy methods that need integrating

# PMR
    @classmethod
    def create_grayscale_from_file(cls, path):
        """
        Reads an image from path and creates a grayscale (w. skimage)
        May throw image exceptions (not trapped)
        :param path:
        :return: single channel grayscale
        """
        assert path is not None
        image = io.imread(path)
        gray_image = AmiImage.create_gray_image_from_image(image)
        return gray_image

    @classmethod
    def create_gray_image_from_image(cls, image):
        gray_image = color.rgb2gray(image)
        return gray_image

    @classmethod
    def create_white_skeleton_image_from_file(cls, path):
        """
        the image may be inverted so the highlights are white

        :param path: path with image
        :return: AmiSkeleton
        """
        # image = io.imread(file)
        assert path is not None
        assert path.exists() and not path.is_dir(), f"{path} should be existing file"
        image = AmiImage.create_grayscale_from_file(path)
        assert image is not None, f"cannot create image from {path}"
        skeleton_image = AmiImage.create_white_skeleton_from_image(image)
        return skeleton_image

    @classmethod
    def create_white_skeleton_from_image(cls, image):
        """
        create skeleton_image based on white components of image

        :param image:
        :return: AmiSkeleton
        """
        assert image is not None
        binary = AmiImage.create_white_binary_from_image(image)
        skeleton_image = morphology.skeletonize(binary)
        return skeleton_image

    @classmethod
    def create_white_binary_from_image(cls, image):
        """
        Create a thresholded, binary image from a grayscale

        :param image: grayscale image
        :return: binary with white pixels as signal
        """
        binary, thresh = AmiImage.create_thresholded_image_and_value(image)
        binary = np.invert(binary)
        return binary, thresh

    @classmethod
    def create_thresholded_image_and_value(cls, image):
        """
        Thresholded image and (attempt) to get threshold
        The thresholded image is OK but the threshold value may not yet work

        :param image: grayscale
        :return: thresholded image, threshold value (latter may not work)
        """

        t_image = threshold(image)
        tt = np.where(t_image > 0)  # above threshold
        return t_image, tt

# Anuv