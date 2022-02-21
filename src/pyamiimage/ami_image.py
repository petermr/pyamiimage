"""
Wrappers and other methods for image manipulation
mainly classmathods, but may need some objects to preserve statementfor expensive functioms
TODO authors Anuv Chakroborty and Peter Murray-Rust, 2021
Apache2 Open Source licence
"""
from distutils import extension
import numpy as np
from skimage import io, color, morphology
import skimage
from pathlib import Path
import os
import matplotlib.pyplot as plt


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
        gray_image = cls.create_grayscale_from_image(image)
        return gray_image

    @classmethod
    def create_grayscale_from_image(cls, image):
        """
        creates grayscale from input image
        accepts following types:
        shape = (*,*,4) assumed rgba and will convert to rgb
        shape = (*,*,3) assumed rgb and will convert to grayscale
        shape = (*,*) assumed grayscale or binary, no action
        :param image:
        :return: grayscale or possibly binary image
        """
        gray = None
        if cls.has_gray_shape(image):
            gray = image
        elif cls.has_alpha_channel_shape(image):
            image = color.rgba2rgb(image)

        if gray is None and AmiImage.has_rgb_shape(image):
            gray = color.rgb2gray(image)

        gray = skimage.img_as_ubyte(gray)
        return gray


    @classmethod
    def create_rgb_from_rgba(cls, image_rgba):
        assert cls.has_alpha_channel_shape(image_rgba)
        image_rgb = color.rgba2rgb(image_rgba)
        assert not cls.has_alpha_channel_shape(image_rgb), f"converted rgb should have lost alpha channel"
        assert cls.has_rgb_shape(image_rgb), f"converted rgb does not have rgb_shape"
        return image_rgb

    @classmethod
    def create_inverted_image(cls, image):
        """Inverts the brightness values of the image
        uses skimage.util.invert
        not yet tested
        :param image: ype not yet defined
        :return: inverted image
        """
        inverted = skimage.util.invert(image)
        return inverted


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

        # grayscale = AmiImage.create_grayscale_from_file(path)
        # assert grayscale is not None, f"cannot create grayscale image from {path}"
        # skeleton_image = AmiImage.create_white_skeleton_from_image(grayscale)
        image = io.imread(path)
        # print(f"path {path} has shape: {image.shape}")
        # print(f"AmiImage.has_alpha_channel_shape() {AmiImage.has_alpha_channel_shape(path)} for {path} ")
        skeleton_image = cls.create_white_skeleton_from_image(image)

        return skeleton_image

    @classmethod
    def create_white_skeleton_from_image(cls, image):
        """
        create skeleton_image based on white components of image
        :param image:
        :return: skeleton image
        """
        assert image is not None

        binary = AmiImage.create_white_binary_from_image(image)
        binary = binary/255
        mask = morphology.skeletonize(binary)
        skeleton = np.zeros(image.shape)
        skeleton[mask] = 255
        return skeleton

    @classmethod
    def create_white_binary_from_file(cls, path):
        assert path is not None
        image = io.imread(path)
        binary = AmiImage.create_white_binary_from_image(image)
        return binary

    @classmethod
    def create_white_binary_from_image(cls, image, threshold=None):
        """"Returns a binary image using a threshold value
        threshold defaults to skimage.filters.threshold_otsu
        :param image: grayscale (0-255?)
        :param threshold: integer or float?
        :return: integer 0/1 binary image
        """
        # check if image is grayscale
        if len(image.shape) > 2:
            # convert to grayscale if not grayscale
            image = cls.create_grayscale_from_image(image)

        # if no threshold is provided, assume default threshold: otsu
        if threshold is None:
            threshold = skimage.filters.threshold_otsu(image)

        binary_image = np.where(image >= threshold, 255, 0)
        return binary_image

    @classmethod
    def invert_binarize_skeletonize(cls, image):
        """Inverts Thresholds and Skeletonize a single channel grayscale image
        :show: display images for invert, threshold and skeletonize
        :return: skeletonized image
        """
        inverted_image = cls.create_inverted_image(image)
        # skeletonize has inbuilt binarization
        skeleton = cls.create_white_skeleton_from_image(inverted_image)
        assert np.max(skeleton) == 255, f"skeleton should have max 255 , found {np.max(skeleton)}"

        return skeleton

    @classmethod
    def check_binary_or_grayscale(cls, gray_image, image):
        if cls.heuristic_check_binary(image):
            pass
        elif cls.check_grayscale(image):
            pass
        else:
            raise ValueError(f"not a gray or binary image {image.shape}")
        assert not (gray_image.dtype == np.floating and np.max(gray_image) == 1.0) and \
               not (gray_image.dtype == int and np.max(gray_image) == 255), \
            f"checking range {gray_image.dtype} {np.max(gray_image)}"

    @classmethod
    def has_alpha_channel_shape(cls, image):
        return type(image) is np.ndarray and len(image.shape) == 3 and image.shape[2] == 4

    @classmethod
    def has_rgb_shape(cls, image):
        return type(image) is np.ndarray and len(image.shape) == 3 and image.shape[2] == 3

    @classmethod
    def has_gray_shape(cls, image):
        """
        checks if 2-D ndarray
        :param image:
        :return:
        """
        return type(image) is np.ndarray and len(image.shape) == 2

    @classmethod
    def get_image_dtype(cls, image):
        """get numpy """
        if type(image) is not np.ndarray or image.size == 0:
            return None
        return image.dtype

    @classmethod
    def heuristic_check_binary(cls, image):
        """
        Ths is very hacky, don't rely on it
        if image is 2-D and max val is 1 or boolean retrun True
        :param image:
        :return:
        """
        if type(image) is not np.ndarray or len(image.shape) != 2:
            return False
        if image.dtype is bool:
            return True
        if image.dtype is int:
            if np.max(image) == 1:
                return True
        if image.dtype is np.floating:
            pass
        return False

    @classmethod
    def check_grayscale(cls, image):
        """
        Very crude, currently just checks is 2-D and not bool
        :param image:
        :return:
        """
        if type(image) is not np.ndarray or len(image.shape) != 2:
            return False
        if image.dtype is bool:
            return False
        return True

    @classmethod
    def pre_plot_image(cls, image_file, erode_rad=0):
        """
        matplotlib plots the gray image, optionally with erosion
        runs plt.imshow, so will need plt.show() afterwards.
        :param erode_rad: erosion disk radius (0 => no erode)
        :param image_file:
        :return:
        """
        assert image_file.exists(), f"{image_file} does not exist"
        rgb = io.imread(image_file)
        img = color.rgb2gray(rgb)
        if erode_rad > 0:
            disk = morphology.disk(erode_rad)
            img = morphology.erosion(img, disk)
        plt.imshow(img, cmap='gray')

    @classmethod
    def write(cls, path, image, mkdir=False, overwrite=True):
        """
        Will throw io errors if cannot write file
        :param image:
        :param path:
        :param mkdir: if True will mkdir for parent
        :param overwrite: if True will overwrite existing file
        :return:
        """
        path = Path(path)
        print(f"image: {type(image)}")
        assert type(image) is np.ndarray and image.ndim >= 2, f"not an image: {type(image)}"
        if not mkdir:
            assert path.parent.exists(), f"parent directory must exist {path.parent}"
        else:
            path.parent.mkdir()
        if path.exists() and overwrite:
            os.remove(path)
        io.imsave(path, image)

    @classmethod
    def write_image_group(cls, dir, images, filename="default", mkdir=True, overwrite=True):
        """write multiple(n) images with the same filename numbered from 1-n"""
        file_extension = ".png"
        for index, image in enumerate(images):
            path = Path(dir, filename + str(index) + file_extension)
            print(path)
            AmiImage.write(path, image, mkdir, overwrite)

class AmiImageDTO():
    """
    Data Transfer Object for images and downstream artefacts
    """
    def __init__(self):
        self.image = None
        self.image_binary = None
        self.nx_graph = None
        self.ami_graph = None
        self.hocr = None
        self.hocr_html_element = None

#    TODO def get_image_type
#    should return
#       RGB 0-1
#       binary 0-255 , etc
# maybe is library somewhere

"""python REPL commands
replay these from this directory (resources) to see the distrib of gray values
sniprgbafile = Path(os.getcwd(), "snippet_rgba.png")
snipgrayim = io.imread(sniprgbafile)
sniprgbim = skimage.color.rgba2rgb(sniprgbaim)
snipgrayim = skimage.color.rgb2gray(sniprgbim)
plt.hist(snipgray)
plt.title("gray values")
plt.ylabel("count")
plt.xlabel("whiteness")
plt.show()
# HISTORY
import readline; print('\n'.join([str(readline.get_history_item(i + 1)) for i in range(readline.get_current_history_length())]))
"""

# deprecated
    # @classmethod
    # #
    # def create_white_binary_from_image(cls, image):
    #     """
    #     Create a thresholded, binary image from a grayscale

    #     :param image: grayscale image
    #     :return: binary with white pixels as signal (thresh is discarded)
    #     """
    #     gray = AmiImage.create_grayscale_from_image(image)
    #     binary, thresh = AmiImage.create_auto_thresholded_image_and_value(gray)
    #     binary = np.invert(binary)
    #     return binary  # discard thresh

    # @classmethod
    # # TODO mark deprecated
    # def create_auto_thresholded_image_and_value(cls, image):
    #     """
    #     Thresholded image and (attempt) to get threshold
    #     The thresholded image is OK but the threshold value may not yet work
    #     uses skan.pre

    #     :param image: grayscale
    #     :return: thresholded image, threshold value (latter may not work)
    #     """
    #     print(f"shape thresh {image.shape}")
    #     t_image = threshold(image)
    #     tt = np.where(t_image > 0)  # above threshold
    #     return t_image, tt


    # @classmethod
    # def skeletonize(cls, image):
    #     """Returns a skeleton of the image
    #     uses skimage.morphology.skeletonize

    #      result white signal black background?
    #      TODO NOT YET TESTED
    #      :param image: constraints?
    #      :return: binary image white signal black background
    #      """
    #     mask = morphology.skeletonize(image)
    #     skeleton = np.zeros(image.shape)
    #     skeleton[mask] = 255
    #     return skeleton

        # @classmethod
    # def create_grayscale_from_image(cls, image):
    #     # requires 2 separate conversions
    #     gray_image = cls.create_gray_from_image(image)
    #     # TODO comment in
    #     # cls.check_binary_or_grayscale(gray_image, image)
    #     return gray_image