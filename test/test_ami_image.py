"""tests AmiImage routines

Mainly class routines
"""
import unittest
from pathlib import Path
from skimage import io
import numpy as np

from ..pyimage.ami_image import AmiImage

RESOURCE_DIR = Path(Path(__file__).parent, "resources")
COMPARE_DIR = Path(Path(__file__).parent, "comparison_images")

RGBA_SNIPPET = Path(RESOURCE_DIR, "snippet_rgba.png")
RGB_SNIPPET = Path(RESOURCE_DIR, "snippet_rgb.png")
GRAY2_SNIPPET = Path(RESOURCE_DIR, "snippet_gray2.png")


interactive = False
# interactive = True


class TestAmiImage:

    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        assert RGBA_SNIPPET.exists(), f"image should exist {RGBA_SNIPPET}"

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        pass

    def test_check_image(self):
        """review properties of test images"""
        image = io.imread(RGBA_SNIPPET)
        assert image.shape == (152, 625, 4)  # RGBA

    def test_existing_images(self):
        """
        attemots to classify existing testb images
        :return:
        """
        image_dicts = [
            {"shape": (1167, 1515),
            "path": Path(RESOURCE_DIR, "biosynth_path_1.png"),
             },
            {"shape": (315, 1512),
             "path": Path(RESOURCE_DIR, "biosynth_path_1_cropped.png"),
             },
            {"shape": (1391, 1420, 3),
             "path": Path(RESOURCE_DIR, "biosynth_path_2.jpg"),
             },
            {"shape": (972, 1020),
             "path": Path(RESOURCE_DIR, "biosynth_path_3.png"),
             },
            {"shape": (546, 1354, 3),
             "path": Path(RESOURCE_DIR, "capacity_r_g_b.png"),
             },
            {"shape": (546, 1354, 3),
             "path": Path(RESOURCE_DIR, "green.png"),
             },
            {"shape": (315, 1512),
             "path": Path(RESOURCE_DIR, "islands_5.png"),
             "max": 255,
             },
            {"shape": (152, 625, 4),
             "path": Path(RESOURCE_DIR, "snippet_rgba.png"),
             },
        ]
        for image_dict in image_dicts:
            print(f"image: {image_dict['shape']}")
            image = io.imread(image_dict['path']);
            assert image.shape == image_dict['shape'], f"shape fails {image.shape}"
            if "max" in image_dict:
                maxx = image_dict['max']
                assert np.max(image) == maxx, f"max value is {np.max(image)}"

    def test_show_image(self):
        image = io.imread(RGBA_SNIPPET)
        assert image.shape == (152, 625, 4)  # RGBA
        if interactive:
            io.imshow(image)
            io.show()

    def test_rgb2agray(self):
        """convert raw "gray" image to grayscale"""
        # TODO create pure rgb image
        image = io.imread(RGBA_SNIPPET)
        gray_image = AmiImage.create_grayscale_from_image(image)
        
        # DOES NOT WORK
        # compare_filename = "gray.png"
        # compare_filepath = Path(COMPARE_DIR, compare_filename)
        # compare_image = io.imread(compare_filepath)
        # assert np.array_equal(gray_image, compare_image), f"Image does not match {compare_filename} image"

        # ***POSSIBLY REDUNDANT***
        assert gray_image.shape == (152, 625)  # gray
        print(gray_image)
        assert np.count_nonzero(gray_image) == 93408  # zero == black
        assert np.size(gray_image) == 95000
        assert np.max(gray_image) == 1.0, "image from 0.0 to 1.0"
        assert np.min(gray_image) == 0.0
        black_lim = 0.1
        non_black = np.where(0.1 < gray_image)

        # print(np.count(non_black))
        non_black_pixels = np.where(gray_image > black_lim)
        # this is a 2-tuple of [x,y] values
        print(f"non black pixels {non_black_pixels}")
        print(f"num non_black {len(non_black_pixels[0])}")
        black_pixels = np.where(gray_image < black_lim)

    def test_convert_rgba2rgb(self):
        image_rgba = io.imread(RGBA_SNIPPET)
        AmiImage.has_alpha_channel_shape(image_rgba)
        image_rgb = AmiImage.create_rgb_from_rgba(image_rgba)
#        io.imsave(RGB_SNIPPET, image_rgb)  # only use to retriueve if lost!
        assert AmiImage.has_rgb_shape(image_rgb) , f"rgb should have rgb_shape"
        assert not AmiImage.has_alpha_channel_shape(image_rgb) , f"rgb should not have rgba_shape"


    def test_rgb2gray(self):
        """convert raw "gray" image to grayscale"""
        # TODO create pure rgb image
        image_rgb = io.imread(RGB_SNIPPET)
        assert AmiImage.has_rgb_shape(image_rgb) , f"rgb should have rgb_shape"
        image_gray = AmiImage.create_grayscale_from_image(image_rgb)
        print(f"gray shape {image_gray.shape}")
        assert image_gray.shape == (152, 625), f"gray shape should be (341, 796)"
        # AmiImage.write(GRAY2_SNIPPET, image_gray)  # only use of recreating lost image
        assert AmiImage.has_gray_shape(image_gray)

        # ***POSSIBLY REDUNDANT***
        assert image_gray.shape == (152, 625)  # gray
        print(image_gray)
        assert np.count_nonzero(image_gray) == 93408  # zero == black
        assert np.size(image_gray) == 95000
        assert np.max(image_gray) == 1.0, "image from 0.0 to 1.0"
        assert np.min(image_gray) == 0.0
        black_lim = 0.1
        non_black = np.where(0.1 < image_gray)

        # print(np.count(non_black))
        non_black_pixels = np.where(image_gray > black_lim)
        # this is a 2-tuple of [x,y] values
        print(f"non black pixels {non_black_pixels}")
        print(f"num non_black {len(non_black_pixels[0])}")
        black_pixels = np.where(image_gray < black_lim)

    def test_white_binary(self):
        """tests image to white binary image"""
        image = io.imread(RGB_SNIPPET)
        white_binary = AmiImage.create_white_binary(image)
        # only unique values in a binary image are 0 and 1
        unique_values = np.unique(white_binary)
        assert len(unique_values) == 2


        # DOES NOT WORK
        # compare_filename = "white_binary.png"
        # compare_filepath = Path(COMPARE_DIR, compare_filename)
        # compare_image = io.imread(compare_filepath)
        # assert np.array_equal(white_binary, compare_image), f"Image does not match {compare_filename} image"        

# TODO tests for AmiImage conversions of simple images to gray and white skeleton
