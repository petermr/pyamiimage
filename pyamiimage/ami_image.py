"""
Wrappers and other methods for image manipulation
mainly classmathods, but may need some objects to preserve statementfor expensive functioms
TODO authors Anuv Chakroborty and Peter Murray-Rust, 2021
Apache2 Open Source licence
"""
import logging

import numpy as np
from skimage import io, color, morphology
import skimage
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import enum
import imageio.v3 as iio

from py4ami.util import Util

TEMP_DIR = Path(__file__).parent.parent
TEMP_DIR = Path(TEMP_DIR, "temp")

FILE = "file"
RGBA = "rgba"
RGB = "rgb"
GRAY = "gray"
# print(f"TEMP {TEMP_DIR}")

class AmiImage:
    """
    instance and class methods for holding and converting images
    """

    def __init__(self):
        """
        for when we need to store state
        :return:
        """
        self.image_dict = dict()

# ========== legacy methods that need integrating

    @classmethod
    def show(cls, image):
        io.imshow(image)
        io.show()

    @classmethod
    def read(cls, path):
        image = io.imread(path)
        # image = AmiImageReader().read_image(path)
        return image
    
    def read_file(self, file):
        assert file, "file should not be None"
        assert file.exists(), "{file} must exist"
        try:
            image = AmiImageReader().read_image(file)
        except Exception as e:
            raise e
        self.image_dict[FILE] = file
        self.store_image(image)

    def store_image(self, image):
        if image is None:
            return
        if self.has_alpha_channel_shape(image):
            self.image_dict[RGBA] = image
        elif self.has_rgb_shape(image):
            self.image_dict[RGB] = image
        elif self.has_gray_shape(image):
            self.image_dict[GRAY] = image

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
        image = AmiImageReader.read_image(path)
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
        white_count = np.sum(gray == 255)
        black_count = np.sum(gray == 0)
        gray_count = gray.shape[0] * gray.shape[1] - (white_count + black_count)
        if gray_count > 0:
            logging.warning(f"{gray_count} pixels other than 0/255 found")
        return gray


    @classmethod
    def create_rgb_from_rgba(cls, image_rgba):
        if image_rgba is None:
            logging.error("cannot create RGB from None")
            return None
        if not cls.has_alpha_channel_shape(image_rgba):
            print(f"channel does not have alpha, found {image_rgba.shape}")
            return image_rgba
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
        assert path is not None
        assert path.exists() and not path.is_dir(), f"{path} should be existing file"

        # grayscale = AmiImage.create_grayscale_from_file(path)
        # assert grayscale is not None, f"cannot create grayscale image from {path}"
        # skeleton_image = AmiImage.create_white_skeleton_from_image(grayscale)
        image = AmiImageReader.read_image(path)

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

        image_white = np.sum(image == 255)
        image_black = np.sum(image == 0)
        binary = AmiImage.create_white_binary_from_image(image)
        white = np.sum(binary == 255)
        black = np.sum(binary == 0)
        binary = binary/255
        logging.warning(f"binary {binary}")
        mask = morphology.skeletonize(binary)
        skeleton = np.zeros(image.shape)
        skeleton[mask] = 255
        return skeleton

    @classmethod
    def create_white_binary_from_file(cls, path):
        assert path is not None
        image = AmiImageReader.read_image(path)
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

        # maybe binary already
        if len(image.shape) == 2:
            size = image.shape[0] * image.shape[1]
            image_white = np.sum(image == 255)
            image_black = np.sum(image == 0)
            if size == image_black + image_white:
                # invert if signal is white
                if image_white > image_black:
                    image = np.where(image == 255, 0, 255)
                return image

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
    def invert_binarize_skeletonize(cls, image, invert=True):
        """Inverts Thresholds and Skeletonize a single channel grayscale image
        :show: display images for invert, threshold and skeletonize
        :param invert: if True invert the image (def=True)
        :return: skeletonized image
        """
        inverted_image = image
        if invert:
            inverted_image = cls.create_inverted_image(image)
        # this is just a debug
        # path = Path(Resources.TEMP_DIR,"inverted.png")
        # print(f"path: {path.absolute()}")
        # io.imsave(path, inverted_image)
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
        rgb = AmiImageReader.read_image(image_file)

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
    def kmeans(cls, raw_image, n_colors, background):
        """finds kmeans in colour space and projects image into each mean
        :param raw_image: raw multicolor image, gets reshaped
        :param n_colors: number of kmeans to extract
        :param background: to add in extracted images
        :return: (labels, centers_i, quantized_images) ;
            labels are per-pixel ints (don't know what)
            centers_i are RGB values at kmeans-centers,
            quantized_images are single colour+background
        :except: may throw "cannot reshape"
        """
        print(f"raw {raw_image.shape}")
        col_layers = raw_image.shape[2]  # find colour layers

        if AmiImage.has_alpha_channel_shape(raw_image):
            fname = str(Path(TEMP_DIR, "junk_rgba.png"))
            io.imsave(fname, raw_image)
            raw_image = AmiImage.create_rgb_from_rgba(raw_image)
            fname = str(Path(TEMP_DIR, "junk_rgb.png"))
            # this is awful, don't know why the rgb image can't be analysed
            # save the image, and then re-read
            io.imsave(fname, raw_image)
            raw_image = AmiImageReader.read_image(fname)

        col_layers = raw_image.shape[2]
        print(f"COLORS: {col_layers} {raw_image.shape}")
        # this might raise error
        reshaped_image = raw_image.reshape((-1, col_layers))
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(reshaped_image)
        labels = kmeans.labels_
        color_centers = kmeans.cluster_centers_
        centers_i = [[int(center[0]), int(center[1]), int(center[2])] for center in color_centers]
        colors_image = color_centers[labels].reshape(raw_image.shape).astype('uint8')
        quantized_images = []
        for i, center_i in enumerate(centers_i):
            quantized_images.append(np.where(colors_image == centers_i[i], [centers_i[i]], background))
        return (labels, centers_i, quantized_images)


    @classmethod
    def write_image_group(cls, dir, images, filename="default", mkdir=True, overwrite=True):
        """write multiple(n) images with the same filename numbered from 1-n"""
        file_extension = ".png"
        for index, image in enumerate(images):
            path = Path(dir, filename + str(index) + file_extension)
            print(path)
            AmiImage.write(path, image, mkdir, overwrite)


class ImageReaderOptions(enum.Enum):
    CV2 = 1,
    SKIMAGE = 2,
    IMAGEIO = 3,

class AmiImageReader:
    """
    wrapper for *.imread() as I don't trust skimage
    """
    def __init__(self):
        pass

    @classmethod
    def read_image(cls, imagefile, reader=ImageReaderOptions.CV2, debug=False):
        """
        reads image using standard libraries. Allows for chioce of library
        as some are giving load trouble.
        :param imagefile: file to read, checks for existence (might be Path)
        :param reader: library to read with (default = CV2)
        :param debug: print image values
        :return: image as array
        """
        if imagefile is None:
            return None
        if not Path(imagefile).exists():
            raise FileNotFoundError(f"no file {imagefile}")
        try:
            if reader == ImageReaderOptions.CV2:
                img = cv2.imread(str(imagefile))
            else:
                img = io.imread(str(imagefile))
            if debug:
                print(f" img {img}")
            return img
        except Exception as e:
            """this is probably a abd idea!"""
            try:
                print(f" retrying as grayscale")
                img = io.imread(imagefile, as_gray=True) # grayscale, but this is not required
            except Exception as e1:
                print(f"cannot read imagefile {imagefile} in grayscale mode because {e1}")
                raise e1
        return img
            # Util.print_stacktrace(e)


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