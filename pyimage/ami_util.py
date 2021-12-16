"""
Utilities (mainly classmethods)
"""
from pathlib import Path
import numpy as np
import numpy.linalg as LA


class AmiUtil:

    @classmethod
    def check_type_and_existence(cls, target, expected_type):
        """
        asserts not None for object and its type
        if path asserts existence


        :param target: object to check
        :param expected_type: type of object
        :return: None
        """
        assert target is not None
        typ = type(target)
        assert typ is expected_type, f"type {typ} should be {expected_type}"
        if expected_type is Path:
            assert target.exists(), f"{target} should exist"

    @classmethod
    def is_ordered_numbers(cls, limits2):
        """
        check limits2 is a numeric 2-tuple in increasing order
        :param limits2:
        :return: True tuple[1] > tuple[2]
        """
        return limits2 is not None and len(limits2) == 2 \
               and AmiUtil.is_number(limits2[0]) and AmiUtil.is_number(limits2[1]) \
               and limits2[1] > limits2[0]

    @classmethod
    def is_number(cls, s):
        """
        test if s is a number
        :param s:
        :return: True if float(s) succeeds
        """
        try:
            float(s)
            return True
        except ValueError:
            return False

    @classmethod
    def get_xy_from_sknw_centroid(cls, yx):
        """
        yx is a 2-array and coords need swapping
        :param yx:
        :return: [x,y]
        """
        assert yx is not None
        assert len(yx) == 2 and type(yx) is np.ndarray, f"xy was {yx}"
        return [yx[1], yx[0]]

    @classmethod
    def make_numpy_assert(cls, numpy_array, shape=None, maxx=None, dtype=None):
        """
        Asserts properties of numpy_array
        :param numpy_array:
        :param shape:
        :param maxx: max value (e.g. 255, or 1.0 for images)
        :param dtype:
        :return:
        """
        assert numpy_array is not None, f"numpy array should not be None"
        if type(numpy_array) is not np.ndarray:
            print(f"object should be numpy.darray, found {type(numpy_array)} \n {numpy_array}")
        if shape:
            assert numpy_array.shape == shape, f"shape should be {numpy_array.shape}"
        if maxx:
            assert np.max(numpy_array) == maxx, f"max should be {np.max(numpy_array)}"
        if dtype:
            assert numpy_array.dtype == dtype, f"dtype should be {numpy_array.dtype}"


class Vector2:

    def __init__(self, v2):
        # self.vec2 = np.array()
        self.vec2 = v2

    @classmethod
    def angle_to(cls, vv0, vv1):
        """

        :param vv0: Vector2
        :param vv1: Vector2
        :return: angle(rad) between vectors (maybe unsigned??)
        """
        v0 = vv0.vec2
        v1 = vv1.vec2

        inner = np.inner(v0, v1)
        norms = LA.norm(v0) * LA.norm(v1)

        cos = inner / norms
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        return rad
