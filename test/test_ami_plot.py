import unittest

from ..pyimage.ami_plot import AmiLineTool


@unittest.skip("Not implemented")
class TestAmiLineTool:
    """ test AmilineTool"""

    def test_empty_polyline(self):
        line_tool = AmiLineTool()
        assert line_tool.points == [], f"found {line_tool.points}"

    def test_single_points(self):
        line_tool = AmiLineTool()
        line_tool.add_point([1, 2])
        assert line_tool.points == [[1, 2]], f"found {line_tool}"

    @unittest.skip("Not implemented")
    def test_single_segment(self):
        line_tool = AmiLineTool()
        line_tool.add_segment([[1, 2], [2, 3]])
        assert line_tool.points == [[1, 2], [2, 3]]

    @unittest.skip("Not implemented")
    def test_multiple_segments(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7]]

    @unittest.skip("Not implemented")
    def test_fail_multiple_segments(self):
        line_tool = AmiLineTool()
        try:
            line_tool.add_segments([[[1, 2], [2, 3]], [[5, 8], [5, 7]]])
            raise ValueError(f"should raise ValueError ppoints do not overlap")
        except ValueError:
            pass

    @unittest.skip("Not implemented")
    def test_add_right_point(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        line_tool.add_point([9, 5])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7], [9, 5]]

    @unittest.skip("prepending to a simple list is not allowed")
    def test_add_left_point(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7]]
        line_tool.insert_point(0, [12, 3])
        assert line_tool.points == [[12, 3], [1, 2], [2, 3], [5, 7]]

    @unittest.skip("Not implemented")
    def test_add_right_segment(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        line_tool.add_segment([[5, 7], [9, 5]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7], [9, 5]]

    @unittest.skip("Not implemented")
    def test_fail_right_segment(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        try:
            line_tool.add_segment([[15, 7], [9, 5]])
            raise ValueError("should raise error as points don't overlap")
        except ValueError:
            pass

    @unittest.skip("Not implemented")
    def test_add_left_segment(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7]]
        line_tool.insert_segment(0, [[25, 37], [1, 2]])
        assert line_tool.points == [[25, 37], [1, 2], [2, 3], [5, 7]]

    def test_add_points_to_empty(self):
        line_tool = AmiLineTool()
        line_tool.add_points([[1, 2], [2, 3], [5, 7]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7]]

    @unittest.skip("Not implemented")
    def test_add_points_to_existing(self):
        line_tool = AmiLineTool()
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7]]
        line_tool.add_points([[12, 3], [7, 2]])
        assert line_tool.points == [[1, 2], [2, 3], [5, 7], [12, 3], [7, 2]]
