"""
Wrappers and other methods for image manipulation
mainly classmathods, but may need some objects to preserve statementfor expensive functioms

TODO authors Anuv Chakroborty and Peter Murray-Rust, 2021
Apache2 Open Source licence
"""
import numpy as np
from skimage import io, color, morphology
import skimage
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
        gray_image = AmiImage.create_grayscale_0_1_float_from_image(image)
        return gray_image

    @classmethod
    def create_grayscale_0_1_float_from_image(cls, image):
        gray_image = color.rgb2gray(image)
        return gray_image

    @classmethod
    def create_white_skeleton_from_file(cls, path):
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
        binary, _ = AmiImage.create_white_binary_from_image(image)
        skeleton_image = morphology.skeletonize(binary)
        return skeleton_image

    @classmethod
    def create_white_binary_from_image(cls, image):
        """
        Create a thresholded, binary image from a grayscale

        :param image: grayscale image
        :return: binary with white pixels as signal
        """
        binary, thresh = AmiImage.create_auto_thresholded_image_and_value(image)
        binary = np.invert(binary)
        return binary, thresh

    @classmethod
    def create_auto_thresholded_image_and_value(cls, image):
        """
        Thresholded image and (attempt) to get threshold
        The thresholded image is OK but the threshold value may not yet work
        uses skan.pre

        :param image: grayscale
        :return: thresholded image, threshold value (latter may not work)
        """

        t_image = threshold(image)
        tt = np.where(t_image > 0)  # above threshold
        return t_image, tt

    @classmethod
    def invert(cls, image):
        """Inverts the brightness values of the image
        uses skimage.util.invert
        not yet tested

        :param image: ype not yet defined
        :return: inverted image
        """
        inverted = skimage.util.invert(image)
        return inverted

    @classmethod
    def skeletonize(cls, image):
        """Returns a skeleton of the image
        uses skimage.morphology.skeletonize

         result white signal black background?
         TODO NOT YET TESTED
         :param image: constraints?
         :return: binary image white signal black background
         """
        mask = morphology.skeletonize(image)
        skeleton = np.zeros(image.shape)
        skeleton[mask] = 1
        return skeleton

    @classmethod
    def threshold(cls, image, threshold=None):
        """"Returns a binary image using a threshold value
        threshold defaults to skimage.filters.threshold_otsu
        :param image: grayscale (0-255?)
        :param threshold: integer or float?
        :return: integer 0/1 binary image
        """
        # check if image is grayscale
        if len(image.shape) > 2:
            # convert to grayscale if not grayscale
            gray = cls.create_grayscale_0_1_float_from_image(image)
        
        # if no threshold is provided, assume default threshold: otsu
        if threshold is None:
            threshold = skimage.filters.threshold_otsu(gray)
        
        binary_image = np.where(gray >= threshold, 1, 0)
        return binary_image

    @classmethod
    def invert_threshold_skeletonize(cls, image):
        """Inverts Thresholds and Skeletonize a single channel grayscale image
        :show: display images for invert, threshold and skeletonize
        :return: skeletonized image
        """
        inverted_image = AmiImage.invert(image)
        binary_image = AmiImage.threshold(inverted_image)
        binary_image = binary_image.astype(np.uint16)
        skeleton = AmiImage.skeletonize(binary_image)

        return skeleton

#    TODO def get_image_type
#    should return
#       RGB 0-1
#       binary 0-255 , etc
# maybe is library somewhere
