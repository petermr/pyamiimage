import unittest

import context
from pyamiimage.ami_plot import POLYGON, POLYLINE, AmiLine, AmiLineTool, X, Y


class TestAmiLineTool:
    """test AmilineTool"""

    def test_empty_polyline(self):
        line_tool = AmiLineTool()
        assert line_tool.line_points_list == []
        assert line_tool.polygons == []
        assert line_tool.mode == POLYLINE

    def test_single_segment(self):
        # creates fresh polyline
        line_tool = AmiLineTool()
        line_tool.add_segment([[1, 2], [2, 3]])
        assert line_tool.line_points_list == [
            [[1, 2], [2, 3]]
        ], f"found {line_tool.line_points_list}"

    def test_multiple_segments(self):
        line_tool = AmiLineTool(xy_flag=X)
        segments = [[[1, 2], [2, 3]], [[2, 3], [5, 7]]]
        line_tool.add_segments(segments)
        assert line_tool.line_points_list == [
            [[1, 2], [2, 3], [5, 7]]
        ], f"found {line_tool.line_points_list}"

    def test_multiple_non_overlapping_segments(self):
        line_tool = AmiLineTool(xy_flag=X)
        # already sorted
        line_tool.add_segments([[[1, 2], [2, 3]], [[5, 8], [5, 7]]])
        assert line_tool.line_points_list == [[[1, 2], [2, 3]], [[5, 8], [5, 7]]]
        # not sorted
        line_tool = AmiLineTool(xy_flag=X)
        line_tool.add_segments([[[5, 8], [5, 7]], [[1, 2], [2, 3]]])
        assert line_tool.line_points_list == [[[5, 8], [5, 7]], [[1, 2], [2, 3]]]

    def test_add_right_segment(self):
        line_tool = AmiLineTool(xy_flag=X)
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        assert line_tool.line_points_list == [[[1, 2], [2, 3], [5, 7]]]
        line_tool.add_segment([[5, 7], [9, 5]])
        assert line_tool.line_points_list == [[[1, 2], [2, 3], [5, 7], [9, 5]]]

    def test_fail_right_segment(self):
        line_tool = AmiLineTool(xy_flag=X)
        line_tool.add_segments([[[1, 2], [2, 3]], [[2, 3], [5, 7]]])
        try:
            line_tool.add_segment([[15, 7], [9, 5]])
            raise ValueError("should raise error as points don't overlap")
        except ValueError:
            pass

    def test_make_unclosed_box_and_close(self):
        line_tool = AmiLineTool(xy_flag=X, mode=POLYGON)
        line_tool.add_segments(
            [[[10, 20], [10, 30]], [[10, 30], [40, 30]], [[40, 30], [40, 20]]]
        )
        assert line_tool.line_points_list == [[[10, 20], [10, 30], [40, 30], [40, 20]]]
        # close box
        line_tool.add_segments([[[10, 20], [40, 20]]])
        assert line_tool.polygons == [[[10, 20], [10, 30], [40, 30], [40, 20]]]

    def test_make_closed_box(self):
        line_tool = AmiLineTool(xy_flag=X, mode=POLYGON)
        line_tool.add_merge_polyline_to_poly_list(
            [[10, 20], [10, 30], [40, 30], [40, 20]]
        )
        assert line_tool.line_points_list == [[[10, 20], [10, 30], [40, 30], [40, 20]]]
        line_tool.add_segments([[[10, 20], [40, 20]]])
        assert line_tool.polygons == [[[10, 20], [10, 30], [40, 30], [40, 20]]]

    def test_joining_lines_head_tail(self):
        """head-tail"""
        line_tool = AmiLineTool()
        polyline0 = [[0, 10], [0, 20], [10, 20]]
        line_tool.add_merge_polyline_to_poly_list(polyline0)
        polyline1 = [[10, 20], [10, 30], [20, 30]]
        line_tool.add_merge_polyline_to_poly_list(polyline1)
        assert line_tool.line_points_list == [
            [[0, 10], [0, 20], [10, 20], [10, 30], [20, 30]]
        ]

    def test_joining_lines_head_head(self):
        """head-head"""

        line_tool = AmiLineTool()
        polyline0 = [[0, 10], [0, 20], [10, 20]]
        line_tool.add_merge_polyline_to_poly_list(polyline0)
        polyline1 = [[20, 30], [10, 30], [10, 20]]
        line_tool.add_merge_polyline_to_poly_list(polyline1)
        assert line_tool.line_points_list == [
            [[0, 10], [0, 20], [10, 20], [10, 30], [20, 30]]
        ]

    def test_joining_lines_tail_tail(self):
        """tail-tail"""
        line_tool = AmiLineTool()
        polyline0 = [[10, 20], [0, 20], [0, 10]]
        line_tool.add_merge_polyline_to_poly_list(polyline0)
        polyline1 = [[10, 20], [10, 30], [20, 30]]
        line_tool.add_merge_polyline_to_poly_list(polyline1)
        assert line_tool.line_points_list == [
            [[20, 30], [10, 30], [10, 20], [0, 20], [0, 10]]
        ]

    def test_joining_lines_tail_head(self):
        """tail-head"""
        line_tool = AmiLineTool()
        polyline0 = [[10, 20], [0, 20], [0, 10]]
        line_tool.add_merge_polyline_to_poly_list(polyline0)
        polyline1 = [[20, 30], [10, 30], [10, 20]]
        line_tool.add_merge_polyline_to_poly_list(polyline1)
        assert line_tool.line_points_list == [
            [[20, 30], [10, 30], [10, 20], [0, 20], [0, 10]]
        ]
