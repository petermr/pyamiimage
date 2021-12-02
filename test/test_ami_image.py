"""tests AmiImage routines

Mainly class routines
"""
import unittest
from pathlib import Path
from skimage import io
import numpy as np

from ..pyimage.ami_image import AmiImage

RESOURCE_DIR = Path(Path(__file__).parent, "resources")
RGB_SNIPPET = Path(RESOURCE_DIR, "biosynth_3_snippet_1.png")


interactive = False


class TestAmiImage:

    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        assert RGB_SNIPPET.exists(), f"image should exist {RGB_SNIPPET}"

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        pass

    def test_check_image(self):
        """review properties of test images"""
        image = io.imread(RGB_SNIPPET)
        assert image.shape == (341, 796, 4)  # RGBA

    @unittest.skipUnless(interactive, "ignore unless interactive")
    def test_show_image(self):
        image = io.imread(RGB_SNIPPET)
        assert image.shape == (341, 796, 4)  # RGBA
        io.imshow(image)
        io.show()

    def test_rgb2gray(self):
        """convert raw "gray" image to grayscale"""
        image = io.imread(RGB_SNIPPET)
        gray_image = AmiImage.create_grayscale_0_1_float_from_image(image)
        assert gray_image.shape == (341, 796)  # gray
        print(gray_image)
        assert np.count_nonzero(gray_image) == 270842  # zero == black
        assert np.size(gray_image) == 271436
        assert np.max(gray_image) == 1.0, "image from 0.0 to 1.0"
        assert np.min(gray_image) == 0.0
        non_black_pixels = np.where(gray_image != 0.0)
        print(f"non black pixels {non_black_pixels}")
        black_pixels = np.any(gray_image == 0.0)


# TODO tests for AmiImage conversions of simple images to gray and white skeleton
