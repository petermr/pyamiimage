import pytest

from ..pyimage.ami_util import Vector2

class TestAmiUtil:
    def test_angle(self):
        """
        test angle between 2 vectors
        :return:
        """
        v0 = Vector2([1, 2])
        v1 = Vector2([1, 0])
        ang = Vector2.angle_to(v0, v1)
        assert pytest.approx(ang) == 1.107149
