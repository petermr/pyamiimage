from pyimage.preprocessing import ImageProcessor
import pytest
import numpy
from pathlib import Path

"""
These tests are desinged to test the preprocessing module
containing the ImageProcessing class 

These tests are for Test Driven Development
"""

class TestImageCode():

    # apparently doesn't like init in Test class
    # def __init__(self):
    #     self.image_processor = None
    #     self.default_image = None
    #     self.default_image_gray = None

    TEST_DIR = Path(__file__).parent
    PYAMIIMAGE_DIR = TEST_DIR.parent
    ASSETS_DIR = Path(PYAMIIMAGE_DIR, "assets").absolute()
    assert ASSETS_DIR.exists(), "assets dir should exist"
    OCIMUM_IMAGE = Path(ASSETS_DIR, "purple_ocimum_basilicum.png")
    assert OCIMUM_IMAGE.exists(), "ocimum image should exist"
    RED_BLACK_IMAGE = Path(ASSETS_DIR, "red_black_cv.png")
    assert RED_BLACK_IMAGE.exists(), "red_black_image should exist"

    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.image_processor = ImageProcessor()
        self.default_image = self.image_processor.load_image(self.OCIMUM_IMAGE)
        assert self.default_image is not None
        self.default_image_gray = self.image_processor.to_gray()
        assert self.default_image_gray is not None


    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        self.image_processor = None
        self.default_image = None
        self.default_image_gray = None


    # @pytest.fixture()
    # def resource():
    #     print("setup")
    #     yield "resource"
    #     print("teardown")

    def test_load_image(self):
        self.image_processor.load_image(self.OCIMUM_IMAGE)
        assert self.image_processor.image is not None

    def test_to_gray_type(self):
        assert self.default_image_gray is not None
        assert type(self.default_image_gray) is numpy.ndarray

    def test_to_gray_shape(self):
        # converts a 3 dimensional array(length, width & channel) to 2 dimensional array (length, width)
        assert self.default_image_gray is not None
        assert len(self.default_image_gray.shape) == 2
        height = 923
        width = 709
        assert self.default_image_gray.shape == (height, width)


    def test_interactive_show_image(self):
        """displays grayscale image and blocks on user """
        interactive = False
        if interactive:
            # will block on user input
            assert self.image_processor.show_image()
        assert True, "finished display"

# NOTES
# progress on extracting graph?
# https://stackoverflow.com/questions/57029293/calculating-a-graph-from-a-skeletonized-image
# https://github.com/Image-Py/sknw#find-path
# https://stackoverflow.com/questions/63653267/how-to-create-a-graph-with-an-images-pixel