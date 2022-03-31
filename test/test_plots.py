import glob
import os
import unittest
from collections import Counter
from pathlib import Path

import imageio

from pyamiimage.ami_graph_all import AmiGraph, AmiEdge, AmiLine
from pyamiimage.ami_plot import AmiPlot, PlotSide, X, Y, TickMark
from pyamiimage.ami_util import AmiUtil
from pyamiimage.bbox import BBox
from pyamiimage.image_exploration import Exploration
from pyamiimage.tesseract_hocr import TesseractOCR
# local
from resources import Resources
from test_ami_skeleton import TestAmiSkeleton


class TestPlots:

    def setup_class(self):
        """
        resources are created once only in self.resources.create_ami_graph_objects()
        Make sure you don't corrupt them
        we may need to add a copy() method
        :return:
        """
        self.resources = Resources()
        self.resources.create_ami_graph_objects()

    def setup_method(self, method):
        self.satish_047q_ami_graph = self.resources.satish_047q_ami_graph

        return self

    def test_satish_erode_dilate(self):
        img_dir = Resources.SATISH_DIR
        os.chdir(img_dir)
        img_files = glob.glob("*.png")
        assert len(img_files) > 0
        min_edge_len = 200
        interactive = False
        for img_file in sorted(img_files):
            img_path = Path(img_file)
            Exploration().explore_dilate_1(img_path, interactive=interactive)
            img = TestAmiSkeleton.create_skeleton_from_file(img_path)
            out_path = Path(Resources.TEMP_DIR, img_path.stem + ".png")
            imageio.imwrite(out_path, img)
            print(f"writing {type(img)} {out_path} {img.shape}")
            ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(img_path)
            ami_islands = ami_graph.get_or_create_ami_islands(mindim=50)
            for island in ami_islands:
                plot_edge = None
                print(f"{island.get_or_create_bbox()} n, {len(island.get_ami_nodes())} "
                      f"e {len(island.get_or_create_ami_edges())} coords {len(island.get_or_create_coords())}")
                if len(island.get_ami_nodes()) <= 2:
                    print(f"edge?")
                    print(f"{[(edge.first_point, edge.last_point) for edge in island.get_or_create_ami_edges()]}")
                    plot_edge = island.get_or_create_ami_edges()[0]
                elif len(ami_islands) == 1:  # (higher numbers means the edge has been identified elsewhere
                    print("box?")
                    for edge in island.get_or_create_ami_edges():
                        pixlen = edge.pixel_length()
                        if pixlen < min_edge_len:
                            continue
                        start_node = edge.get_start_ami_node()
                        end_node = edge.get_end_ami_node()
                        if not start_node or not end_node:
                            print(f"cannot find nodes for long edge")
                            continue
                        print(f"{edge} {pixlen} {start_node} {end_node}")
                        print(
                            f"start {len(start_node.get_or_create_ami_edges())} end {len(end_node.get_or_create_ami_edges())}")
                        if len(start_node.get_or_create_ami_edges()) == 1 or len(
                                end_node.get_or_create_ami_edges()) == 1:
                            plot_edge = edge
                            break
                if plot_edge:
                    xy_array = plot_edge.points_xy
                    csv_path = Path(Resources.TEMP_DIR, img_path.stem + ".csv")
                    AmiUtil.write_xy_to_csv(xy_array, csv_path)

    def test_box_and_ticks(self):
        """creates axial box and ticks
        """

        ami_graph = self.satish_047q_ami_graph
        # ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.SATISH_047Q_RAW)

        plot_islands = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)
        if len(plot_islands) == 0:
            raise ValueError(f" cannot find islands in {len(ami_graph.get_or_create_ami_islands())}")
        plot_island = plot_islands[0]
        ami_edges = plot_island.get_or_create_ami_edges()

        horizontal_edges = AmiEdge.get_horizontal_edges(ami_edges, tolerance=2)
        vertical_edges = AmiEdge.get_vertical_edges(ami_edges, tolerance=2)
        horiz_ami_lines = AmiEdge.get_single_lines(horizontal_edges)
        vert_ami_lines = AmiEdge.get_single_lines(vertical_edges)

        island_bbox = plot_island.get_or_create_bbox()
        assert island_bbox.xy_ranges == [[82, 449], [81, 352]]
        assert len(horizontal_edges) == 8, f"horizontal edges (3 vertical ticks and 5 x-axis segments)"
        assert len(vertical_edges) == 8, f"vertical edges (5 horizontal ticks and 3 y-axis segments)"
        assert len(horiz_ami_lines) == 8
        assert len(vert_ami_lines) == 8

        # ticks
        ami_plot = AmiPlot(bbox=island_bbox)

        vert_box = ami_plot.get_axial_box(side=PlotSide.LEFT)
        vert_box.change_range(1, 3)
        y_ticks = TickMark.get_tick_marks(horiz_ami_lines, vert_box, Y)
        # TODO sort this list

        assert vert_box.xy_ranges == [[72, 92], [78, 355]]
        TickMark.assert_ticks([
            [[82, 87], [151, 151]],
            [[82, 87], [285, 285]],
            [[82, 87], [83, 83]],
            [[82, 87], [218, 218]],
        ], y_ticks)  # pick up ticks above and characters below

        horiz_box = ami_plot.get_axial_box(side=PlotSide.BOTTOM, high_margin=25)
        assert type(horiz_box) is BBox
        horiz_box.change_range(1, 3)
        assert horiz_box.xy_ranges == [[82, 449], [339, 380]]
        x_ticks = TickMark.get_tick_marks(vert_ami_lines, horiz_box, X)
        x_tick_exp = [
            [[149, 149], [347, 352]],
            [[216, 216], [347, 352]],
            [[282, 282], [347, 352]],
            [[349, 349], [347, 352]],
            [[415, 415], [347, 352]],
        ]
        TickMark.assert_ticks(x_tick_exp, x_ticks)

        # axial polylines
        tolerance = 2
        axial_polylines = AmiEdge.get_axial_polylines(ami_edges, tolerance=tolerance)
        assert len(axial_polylines) == 2, f"axial polylines {len(axial_polylines)}"
        assert type(axial_polylines) is list
        assert type(axial_polylines[0]) is list
        assert type(axial_polylines[0][0]) is AmiLine
        assert len(axial_polylines[0]) == 2
        assert len(axial_polylines[1]) == 3
        # I don't like the str(...) but how to compare lists of coords? probably need a polyline class
        assert str(axial_polylines[0][0]) == str([[82, 285], [82, 351]])
        assert str(axial_polylines[0]) == str([[[82, 285], [82, 351]], [[82, 351], [149, 352]]])

        for axial_polyline in axial_polylines:
            for ami_line in axial_polyline:
                if ami_line.is_vertical(tolerance=tolerance):
                    vert_ami_lines.append(ami_line)
                elif ami_line.is_horizontal(tolerance=tolerance):
                    horiz_ami_lines.append(ami_line)
                else:
                    raise ValueError(f"line {ami_line} must be horizontal or vertical")

        vert_dict = AmiLine.get_horiz_vert_counter(vert_ami_lines, xy_index=0)
        assert type(vert_dict) is Counter
        assert vert_dict == Counter({82: 4, 149: 1, 216: 1, 282: 1, 349: 1, 415: 1, 448: 1}), \
            f"found {vert_dict}"
        horiz_dict = AmiLine.get_horiz_vert_counter(horiz_ami_lines, xy_index=1)
        assert horiz_dict == Counter({352: 4, 351: 2, 151: 1, 285: 1, 83: 1, 218: 1, 82: 1}), f"found {horiz_dict}"

# this image doesn't give good words
        image_file = Resources.SATISH_047Q_RAW

        expected_bboxes = [
            [[34, 45], [203, 261]],
            [[34, 48], [173, 197]],
            [[246, 300], [20, 35]],
            [[82, 450], [76, 87]],
            [[79, 86], [81, 353]],
            [[446, 453], [80, 353]],
            [[82, 450], [348, 357]],
            [[144, 156], [358, 367]],
            [[276, 289], [358, 367]],
            [[230, 266], [393, 407]],
            [[271, 302], [393, 407]],
            [[409, 422], [358, 367]],
            [[461, 491], [91, 102]],
            [[496, 525], [92, 104]]
        ]

        word_numpys, words = TesseractOCR.extract_numpy_box_from_image(image_file)
        word_bboxes = [BBox.create_from_numpy_array(word_numpy) for word_numpy in word_numpys]
        # horiz_text2coord_list = self.match_scale_text2ticks(word_bboxes, horiz_box, words, x_ticks)

        assert words == ['Hardness', '(Hv)', 'Jominy', ' ', ' ', ' ', ' ', '10', '30', 'Depth',
                         '(mm)', '50', 'eo', '0479']

    def test_create_plot_box_042a(self):
        """creates axial box and ticks
        rescaled image is twice the size
        """

        # ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.SATISH_042A_RAW)
        path042a = Path(Resources.TEST_RESOURCE_DIR, "042A", "raw.png")
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(path042a)
        mindim = 50
        islands = ami_graph.get_or_create_ami_islands(mindim=mindim)
        assert len(islands) == 2, f"should be box and the plot inside it"
        # get biggest box
        island_index = 0 if islands[0].get_or_create_bbox().get_height() > islands[1].get_or_create_bbox().get_height() else 1

        ami_plot = AmiPlot(image_file=path042a)
        ami_plot.create_scaled_plot_box(island_index=island_index, mindim=mindim)
        print(f"left text2coord {ami_plot.left_scale.text2coord_list}")
        print(f"bottom text2coord {ami_plot.bottom_scale.text2coord_list}")
        assert ami_plot.bottom_scale.text2coord_list[0][0] == '10'
        # assert ami_plot.bottom_scale.text2coord_list[0][1].bbox.xy_ranges == [[149, 149], [347, 352]]

        assert ami_plot.bottom_scale.user_to_plot_scale == 19.575
        assert ami_plot.bottom_scale.user_num_to_plot_offset == 225.25

        assert not ami_plot.left_scale.text2coord_list

    def test_create_plot_box_005b(self):
        """creates axial box and ticks
        """
        # this image doesn't give good words
        ami_plot = AmiPlot(image_file=Resources.SATISH_005B_RAW)
        ami_plot.create_scaled_plot_box(island_index=0, mindim=30)
        print(f"left {ami_plot.axial_box_by_side['LEFT']}")
        print(f"left {ami_plot.left_scale.text2coord_list}")
        print(f"bottom {ami_plot.bottom_scale.text2coord_list}")

    def test_create_plot_box_many(self):
        os.chdir(Resources.SATISH_DIR)
        files = glob.glob("*.png")
        for image_file in files:
            try:
                print(f"{image_file}")
                ami_plot = AmiPlot(image_file=Path(image_file))
                ami_plot.create_scaled_plot_box(island_index=0)
                print(f"left {ami_plot.left_scale.text2coord_list}")
                print(f" {image_file} left {ami_plot.left_scale.text2coord_list}")
                print(f" {image_file} bottom {ami_plot.bottom_scale.text2coord_list}")
            except Exception as e:
                print (e)

