"""tests AmiImage routines

Mainly class routines
"""
from pathlib import Path

import numpy as np
from skimage import io, morphology
from skimage import data
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt

from ..pyimage.ami_image import AmiImage
from ..test.resources import Resources

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
        # assert RGBA_SNIPPET.exists(), f"image should exist {RGBA_SNIPPET}"
        # self.image = io.imread(RGBA_SNIPPET)
        assert RGB_SNIPPET.exists(), f"image should exist {RGB_SNIPPET}"
        self.image = io.imread(RGB_SNIPPET)

        assert COMPARE_DIR.exists(), "Comparison directory does not exist, run generate_compare_files.py"
    
    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        pass

    # def test_check_image_shape(self):
    #     """review properties of test images"""
    #     assert self.image.shape == (341, 796, 3)  # RGB

    def test_check_image_values(self):
        """check the values of imported image against comparison"""

        TestAmiImage.assert_image_equals_repository_image_file(self.image, "original.png", COMPARE_DIR)

    def test_rgb2gray(self):
        """convert raw "gray" image to grayscale"""
        # TODO create pure rgb image
        gray_image = AmiImage.create_grayscale_from_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(gray_image, "gray.png", COMPARE_DIR)

    def test_white_binary(self):
        """tests image to white binary image"""
        white_binary = AmiImage.create_white_binary_from_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(white_binary, "white_binary.png", COMPARE_DIR)

    def test_create_inverted_image(self):
        """tests image to inverted image"""
        inverted = AmiImage.create_inverted_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(inverted, "inverted.png", COMPARE_DIR)

    def test_create_white_skeleton_from_image(self):
        """tests image to skeleton image"""
        inverted = AmiImage.create_inverted_image(self.image)
        white_skeleton = AmiImage.create_white_skeleton_from_image(inverted)

        TestAmiImage.assert_image_equals_repository_image_file(white_skeleton, "white_skeleton.png", COMPARE_DIR)


    def test_sharpen_iucr(self):
        """shows effect of sharpening (interactive display)
        """
        image = io.imread(str(Path(Resources.YW5003_5)))
        assert image is not None
        print (image.shape)
        image = AmiImage.create_rgb_from_rgba(image)
        image = AmiImage.create_grayscale_from_image(image)

        result_1 = unsharp_mask(image, radius=1, amount=1)
        result_2 = unsharp_mask(image, radius=5, amount=2)
        result_3 = unsharp_mask(image, radius=20, amount=1)


        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 sharex=True, sharey=True, figsize=(10, 10))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original image')
        ax[1].imshow(result_1, cmap=plt.cm.gray)
        ax[1].set_title('Enhanced image, radius=1, amount=1.0')
        ax[2].imshow(result_2, cmap=plt.cm.gray)
        ax[2].set_title('Enhanced image, radius=5, amount=2.0') # best
        ax[3].imshow(result_3, cmap=plt.cm.gray)
        ax[3].set_title('Enhanced image, radius=20, amount=1.0')
        print(f"shape {result_3}")

        if interactive:
            plt.show()


# ========== helper methods ==============
    @classmethod
    def assert_image_equals_repository_image_file(cls, original_image, expected_filename, repository_dir):
        """
        uses assert_image_equals to compare images
        expected images are in a communal directory
        if images are not equal, fails assert

        :param expected_filename: filename of expected image
        :param original_image: image to compare with expected
        :param repository_dir: in which expected filename is
        :return: None
        """
        compare_filepath = Path(repository_dir, expected_filename)
        assert compare_filepath.exists(), f"{expected_filename} does not exist, have you run generate_compare_files.py?"
        compare_image = io.imread(compare_filepath)
        cls.assert_image_equals(original_image, compare_image, expected_filename)

    @classmethod
    def assert_image_equals(cls, comparable_image, compare_image, expected_filename=None):
        """
        convenience method to compare 2 images using np.array_equal

        :param comparable_image:
        :param compare_image:
        :param expected_filename: optional filename for error message
        :return:
        """
        msg = f"{expected_filename} image" if expected_filename is not None else ""
        assert np.array_equal(comparable_image, compare_image), f"Image does not match {msg}"

