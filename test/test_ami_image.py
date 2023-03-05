"""tests AmiImage routines

Mainly class routines
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io, morphology
from skimage.filters import unsharp_mask
import imageio as iio

from pyamiimage.ami_image import AmiImage, AmiImageReader
from resources import Resources
from ami_test_lib import AmiAnyTest
from ami_plot import AmiPlotter

RESOURCE_DIR = Path(Path(__file__).parent, "resources")
COMPARE_DIR = Path(Path(__file__).parent, "comparison_images")

RGBA_SNIPPET = Path(RESOURCE_DIR, "snippet_rgba.png")
RGB_SNIPPET = Path(RESOURCE_DIR, "snippet_rgb.png")
GRAY2_SNIPPET = Path(RESOURCE_DIR, "snippet_gray2.png")


interactive = False
# interactive = True


class TestAmiImage(
    # AmiAnyTest
):
    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        # assert RGBA_SNIPPET.exists(), f"image should exist {RGBA_SNIPPET}"
        assert RGB_SNIPPET.exists(), f"image should exist {RGB_SNIPPET}"
        self.image = AmiImageReader.read_image(RGB_SNIPPET)

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

        TestAmiImage.assert_image_equals_repository_image_file(
            self.image, "original.png", COMPARE_DIR
        )

    def test_rgb2gray(self):
        """convert raw "gray" image to grayscale"""
        # TODO create pure rgb image
        gray_image = AmiImage.create_grayscale_from_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(
            gray_image, "gray.png", COMPARE_DIR
        )

    def test_white_binary(self):
        """tests image to white binary image"""
        white_binary = AmiImage.create_white_binary_from_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(
            white_binary, "white_binary.png", COMPARE_DIR
        )

    def test_create_inverted_image(self):
        """tests image to inverted image"""
        inverted = AmiImage.create_inverted_image(self.image)

        TestAmiImage.assert_image_equals_repository_image_file(
            inverted, "inverted.png", COMPARE_DIR
        )

    def test_create_white_skeleton_from_image(self):
        """tests image to skeleton image"""
        inverted = AmiImage.create_inverted_image(self.image)
        white_skeleton = AmiImage.create_white_skeleton_from_image(inverted)

        TestAmiImage.assert_image_equals_repository_image_file(
            white_skeleton, "white_skeleton.png", COMPARE_DIR
        )

    def test_sharpen_iucr(self):
        """shows effect of sharpening (interactive display)"""
        image = AmiImageReader.read_image(str(Path(Resources.YW5003_5_RAW)))
        assert image is not None
        print(image.shape)
        image = AmiImage.create_rgb_from_rgba(image)
        image = AmiImage.create_grayscale_from_image(image)

        result_1 = unsharp_mask(image, radius=1, amount=1)
        result_2 = unsharp_mask(image, radius=5, amount=2)
        result_3 = unsharp_mask(image, radius=20, amount=1)
        images = [image, result_1, result_2, result_3]
        titles = [
            "Original image",
            "Enhanced image, radius=1, amount=1.0",
            "Enhanced image, radius=5, amount=2.0",
            "Enhanced image, radius=20, amount=1.0",
        ]
        cmaps = [
            plt.cm.gray,
            plt.cm.gray,
            plt.cm.gray,
            plt.cm.gray,
        ]

        plotter = AmiPlotter(nrows=2, ncols=2, sharex="all", sharey="all", figsize=(10, 10))
        for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
            plotter.imshow(image=image, axis=i, title=title, cmap=cmap)
        plotter.show(interactive)

        AmiPlotter().imshow(image=image, title="single plot").show(interactive)


    def test_plot_array(self):
        points = np.array(
            [
                [1.0, 1.0],
                [1.1, 2.0],
                [0.9, 3.0],
                [2.0, 2.9],
                [3.0, 3.0],
            ]
        )
        plotter = AmiPlotter(nrows=2, ncols=1, sharex="all", sharey="all", figsize=(10, 10))
        plotter.plot(points, title="Single Plot").show(interactive)

    # ========== helper methods ==============
    @classmethod
    def assert_image_equals_repository_image_file(
        cls, original_image, expected_filename, repository_dir
    ):
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
        assert (
            compare_filepath.exists()
        ), f"{expected_filename} does not exist, have you run generate_compare_files.py?"
        compare_image = io.imread(compare_filepath)
        cls.assert_image_equals(original_image, compare_image, expected_filename)

    @classmethod
    def assert_image_equals(
        cls, comparable_image, compare_image, expected_filename=None
    ):
        """
        convenience method to compare 2 images using np.array_equal

        :param comparable_image:
        :param compare_image:
        :param expected_filename: optional filename for error message
        :return:
        """
        msg = f"{expected_filename} image" if expected_filename is not None else ""
        assert np.array_equal(
            comparable_image, compare_image
        ), f"Image does not match {msg}"

    def test_can_read_grayscale(self):
        """
        tests that images can be read.
        (There is/has_been a problem with reading in some complex situations)
        """
        img = io.imread(GRAY2_SNIPPET)
        print(f"read io img {img.shape}\n{img}")
        img = iio.imread(GRAY2_SNIPPET)
        print(f"read iio img {img.shape}\n{img}")

        img = AmiImageReader.read_image(GRAY2_SNIPPET)
        print(f"img ami {img.shape} \n{img}")

