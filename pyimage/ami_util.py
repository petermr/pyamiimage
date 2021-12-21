"""
Utilities (mainly classmethods)
"""
from pathlib import Path
import numpy as np
import numpy.linalg as LA
import math


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
        assert type(numpy_array) is np.ndarray, \
            f"object should be numpy.darray, found {type(numpy_array)} \n {numpy_array}"
        if shape:
            assert numpy_array.shape == shape, f"shape should be {numpy_array.shape}"
        if maxx:
            assert np.max(numpy_array) == maxx, f"max should be {np.max(numpy_array)}"
        if dtype:
            assert numpy_array.dtype == dtype, f"dtype should be {numpy_array.dtype}"

    @classmethod
    def get_angle(cls, p0, p1, p2):
        '''
        signed angle p0-p1-p2
        :param p0:
        :param p1: centre point
        :param p2:
        '''
        AmiUtil.assert_is_float_array(p0)
        AmiUtil.assert_is_float_array(p1)
        AmiUtil.assert_is_float_array(p2)

        # print(f"p0 {p0} {p0[0]} {type(p0[0])} p1 {p1} {p1[0]} {type(p1[0])} p2 {p2} {p2[0]} {type(p2[0])}")
        linal = False
        # print(f"{p0}  {np.array(p0)} {np.array(p1)}")
        if linal:
            np0 = np.array(p0, dtype=np.uint8)
            np1 = np.array(p1, dtype=np.uint8)
            np2 = np.array(p2, dtype=np.uint8)
            # print(f"{p0} {np0} {p1} {np1} {p2} {np2}")
            v0 = np0 - np1
            v1 = np2 - np1
            # print(f"{v0}, {v1}")
            angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        else:
            dx0 = p0[0] - p1[0]
            dy0 = p0[1] - p1[1]
            # print(f"dx0 {dx0} {type(dx0)} dx0 {dy0} {type(dy0)}")
            v01 = [p0[0] - p1[0], p0[1] - p1[1]]
            v21 = [p2[0] - p1[0], p2[1] - p1[1]]
            # print(f"v01 {v01} v21 {v21}")
            ang01 = math.atan2(v01[1], v01[0])
            ang21 = math.atan2(v21[1], v21[0])
            # print(f" ang01 {ang01} ang21 {ang21}")
            angle = ang21 - ang01
            if angle > math.pi:
                angle -= 2 * math.pi

        return angle

    @classmethod
    def assert_is_float_array(cls, arr, length=2):
        """
        assert arr[0] is float and has given length
        :param arr:
        :param length:
        :return:
        """

        assert len(arr) == length and type(arr[0]) is float, f"arr must be 2-vector float {arr}"

    @classmethod
    def float_list(cls, int_lst):
        """
        converts a list of ints or np.uint16 or np.uint8 to floats
        :param int_lst: 
        :return: 
        """
        assert int_lst is not None and type(int_lst) is list and len(int_lst) > 0, f"not a list: {int_lst}"
        tt = type(int_lst[0])
        assert tt is int or tt is np.uint8 or tt is np.uint16, f"expected int, got {tt}"
        return [float(i) for i in int_lst]

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
