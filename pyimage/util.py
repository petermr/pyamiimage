"""
Utilities (mainly classmethods
"""
from pathlib import Path
import numpy as np


class Util:

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
            and Util.is_number(limits2[0]) and Util.is_number(limits2[1]) \
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
        assert yx is not None;
        assert len(yx) == 2 and type(yx) is np.ndarray, f"xy was {yx}"
        return [yx[1], yx[0]]

