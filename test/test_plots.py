from collections import Counter
import glob
from pathlib import Path
import os
import imageio
import numpy as np

# local
from resources import Resources
from pyamiimage.ami_graph_all import AmiGraph, AmiEdge, AmiLine
from pyamiimage.ami_plot import AmiPlot, PlotSide, X, Y
from pyamiimage.image_exploration import Exploration
from pyamiimage.ami_util import AmiUtil
from pyamiimage.tesseract_hocr import TesseractOCR
from pyamiimage.bbox import BBox
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
        MIN_EDGE_LEN = 200

        for img_file in sorted(img_files):
            img_path = Path(img_file)
            Exploration().explore_dilate_1(img_path)
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
                        if pixlen < MIN_EDGE_LEN:
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
        plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        island_bbox = plot_island.get_or_create_bbox()
        assert island_bbox.xy_ranges == [[82, 449], [81, 352]]
        ami_edges = plot_island.get_or_create_ami_edges()

        horizontal_edges = AmiEdge.get_horizontal_edges(ami_edges, tolerance=2)
        assert len(horizontal_edges) == 8, f"horizontal edges (3 vertical ticks and 5 x-axis segments)"
        vertical_edges = AmiEdge.get_vertical_edges(ami_edges, tolerance=2)
        assert len(vertical_edges) == 8, f"vertical edges (5 horizontal ticks and 3 y-axis segments)"
        horiz_ami_lines = AmiEdge.get_single_lines(horizontal_edges)
        assert len(horiz_ami_lines) == 8
        vert_ami_lines = AmiEdge.get_single_lines(vertical_edges)
        assert len(vert_ami_lines) == 8

        # ticks
        ami_plot = AmiPlot(bbox=island_bbox)
        vert_box = ami_plot.get_axial_box(side=PlotSide.LEFT)
        vert_box.change_range(1, 3)
        assert vert_box.xy_ranges == [[72, 92], [78, 355]]
        y_ticks = AmiLine.get_tick_coords(horiz_ami_lines, vert_box, Y)
        assert y_ticks == [83, 151, 218, 285], f"y ticks"
        horiz_box = ami_plot.get_axial_box(side=PlotSide.BOTTOM)
        horiz_box.change_range(1, 3)
        assert horiz_box.xy_ranges == [[82, 449], [339, 365]]
        x_ticks = AmiLine.get_tick_coords(vert_ami_lines, horiz_box, X)
        assert x_ticks == [149, 216, 282, 349, 415], f"x ticks"

        # bottom_box = ami_plot.get_axial_box(side=PlotSide.BOTTOM)
        # bottom_box.change_range(1, 3)
        # assert bottom_box.xy_ranges == [[82, 449], [339, 365]]

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

# this images doesn't give good words
        bboxes, words = TesseractOCR.extract_numpy_box_from_image(Resources.SATISH_047Q_RAW)
        assert len(bboxes) == 14
        print(words)

        expected_boxes = [
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
        bboxes = [BBox.create_from_numpy_array(numpy_bbox) for numpy_bbox in bboxes]
        for i, bbox in enumerate(bboxes):
            assert bbox.xy_ranges == expected_boxes[i]
            print(f"{bbox} {words[i]}")

        assert words == ['Hardness', '(Hv)', 'Jominy', ' ', ' ', ' ', ' ', '10', '30', 'Depth',
                         '(mm)', '50', 'eo', '0479']

    def test_box_and_ticks_2(self):
        """creates axial box and ticks
        """
        ami_graph = self.satish_047q_ami_graph
        plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        island_bbox = plot_island.get_or_create_bbox()
        assert island_bbox.xy_ranges == [[82, 449], [81, 352]]
        ami_edges = plot_island.get_or_create_ami_edges()

        horizontal_edges = AmiEdge.get_horizontal_edges(ami_edges, tolerance=2)
        assert len(horizontal_edges) == 8, f"horizontal edges (3 vertical ticks and 5 x-axis segments)"
        vertical_edges = AmiEdge.get_vertical_edges(ami_edges, tolerance=2)
        assert len(vertical_edges) == 8, f"vertical edges (5 horizontal ticks and 3 y-axis segments)"
        horiz_ami_lines = AmiEdge.get_single_lines(horizontal_edges)
        assert len(horiz_ami_lines) == 8
        vert_ami_lines = AmiEdge.get_single_lines(vertical_edges)
        assert len(vert_ami_lines) == 8

        # ticks
        ami_plot = AmiPlot(bbox=island_bbox)
        vert_box = ami_plot.get_axial_box(side=PlotSide.LEFT)
        vert_box.change_range(1, 3)
        assert vert_box.xy_ranges == [[72, 92], [78, 355]]
        y_ticks = AmiLine.get_tick_coords(horiz_ami_lines, vert_box, Y)
        assert y_ticks == [83, 151, 218, 285], f"y ticks"
        horiz_box = ami_plot.get_axial_box(side=PlotSide.BOTTOM)
        horiz_box.change_range(1, 3)
        assert horiz_box.xy_ranges == [[82, 449], [339, 365]]
        x_ticks = AmiLine.get_tick_coords(vert_ami_lines, horiz_box, X)
        assert x_ticks == [149, 216, 282, 349, 415], f"x ticks"

        # bottom_box = ami_plot.get_axial_box(side=PlotSide.BOTTOM)
        # bottom_box.change_range(1, 3)
        # assert bottom_box.xy_ranges == [[82, 449], [339, 365]]

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

# this images doesn't give good words
        bboxes, words = TesseractOCR.extract_numpy_box_from_image(Resources.SATISH_047Q_RAW)
        assert len(bboxes) == 14
        print(words)

        expected_boxes = [
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
        bboxes = [BBox.create_from_numpy_array(numpy_bbox) for numpy_bbox in bboxes]
        for i, bbox in enumerate(bboxes):
            assert bbox.xy_ranges == expected_boxes[i]
            print(f"{bbox} {words[i]}")

        assert words == ['Hardness', '(Hv)', 'Jominy', ' ', ' ', ' ', ' ', '10', '30', 'Depth',
                         '(mm)', '50', 'eo', '0479']
