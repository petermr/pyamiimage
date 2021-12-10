# tests existing routines in image_lib

from pyimage.old_code.image_lib import ImageLib
import numpy

"""
These tests are for experimenting with ImageLib class 
"""


def test_circle_points():
    """Old circle_points generates an array of numpy points on a circle
    This is primarily to test thetb the test system is working
    """
    image_lib = ImageLib()
    # circle_points(200, [80, 250], 80)[:-1]
    resolution = 200  # ? number of points?
    center = [80, 250]
    radius = 80
    points = image_lib.circle_points(resolution, center, radius)[:-1]
    assert type(points) is numpy.ndarray
    assert len(points) == 199
    assert points[0][0] == 330.
    assert points[0][1] == 80.


def test_image_import():
    # """Check that ImageLib reads an image and retains the shape"""
    # image_lib = ImageLib()
    # image_lib.image_import()
    # assert image_lib.image is not None
    # #  use == for equality, not "is"
    # assert image_lib.image.shape == (923, 709, 3)
    pass
