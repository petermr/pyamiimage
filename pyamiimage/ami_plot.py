import logging
import re
from collections import Counter
# local
from pyamiimage.ami_util import AmiUtil
from pyamiimage.bbox import BBox
from pyamiimage.tesseract_hocr import TesseractOCR
from pyamiimage.ami_ocr import AmiOCR

# TODO move AmiLine (and maybe others) into ami_graph_all - causes import problems
#  and doesn't really belong here

logger = logging.getLogger(__name__)


HEAD = "head"

DASHED = "dashed"
POLY = "poly"  # polyline or polygon
POLYGON = "polygon"
POLYLINE = "polyline"
TARGETS = [POLY, POLYGON, POLYLINE]

SEGMENT = "segment"
TAIL = "tail"
X = 0
Y = 1
MAX_DELTA_TICK = 10


class PlotSide:
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    SIDES = [LEFT, TOP, RIGHT, BOTTOM]


class ScaleText:
    """holds text and bounding box for a scale value

     """

    def __init__(self, text, bbox):
        """creates scake text

        :param text: e.g. '20.0'
        :param bbox: e.g. BBox (xy_ranges = [[20,30], [40, 47]]
        """
        self.text = text
        self.bbox = bbox

    def centroid_coord(self, axis):
        """

        :param axis: X or Y
        :return:
        """
        assert axis == X or axis == Y
        return self.bbox.centroid[axis]


class TickMark:

    def __init__(self, bbox):
        """holds a tick mark

        Use a class in case we annotate later (e.g. with Major/Minor)
        :param bbox: BBox exactly containing the tick mark (may be zero width)
        """
        assert bbox is not None, f"must not be None"
        assert type(bbox) is BBox, f"expected BBox found {bbox}"
        # bbox of the tick mark line
        self.bbox = bbox
        # text (or None) for tick
        self.user_text = None
        # value computed from user_text
        self.user_num = None

    def __repr__(self):
        return str(self.bbox)+" "+str(self.user_text)+" "+str(self.user_num)

    def __str__(self):
        return str(self.bbox)

    @property
    def coord(self):
        """coordinate of line

        :return: coordinate perpendicular to tick mark
        """
        return self.centroid[X] if self.orientation is Y else X

    @property
    def centroid(self):
        """centroid of bbox
        :return: centroid of surrounding bbox created from elsewhere (e.g. OCR)
        """
        return self.bbox.centroid

    @property
    def orientation(self):
        """direction of tick

        X has for horizontal ticks, Y for vertical ticks
        :return: X if tick line is horizontal else Y
        """
        return X if self.bbox.get_width() > self.bbox.get_height() else Y

    @property
    def perpendicular(self):
        """direction perpendicular to tick

        Y for horizontal ticks, X for vertical ticks
        :return: Y if tick line is horizontal else X
        """
        return Y if self.bbox.get_width() > self.bbox.get_height() else X

    @classmethod
    def get_tick_marks(cls, ami_lines, including_box, axis):
        """
        get changing tick coordinates for lines

        :param ami_lines: horizontal or vertical lines
        :param including_box: box which must totally include tick marks
        :param axis: X or Y
        :return: tick_marks
        """
        assert ami_lines is not None and len(ami_lines) >= 1, f"must have at least one tick"
        assert including_box is not None
        assert axis == X or axis == Y, f"must have X or Y axis"
        return [TickMark(line.bbox) for line in ami_lines if including_box.contains_bbox(line.bbox)]

    @classmethod
    def assert_ticks(cls, tick_exp, x_ticks):
        for i, x_tick in enumerate(x_ticks):
            assert x_tick.bbox.xy_ranges == tick_exp[i], f"tick found {x_tick.bbox.xy_ranges} expected {tick_exp[i]}"

    @classmethod
    def match_ticks_to_text(cls, scale_texts, ticks, max_delta=MAX_DELTA_TICK):
        """
        matches centroids of ScaleTexts to TickMark coordinates

        :param scale_texts: list of ScaleTexts to match (normally numbers, but not guaranteed)
        :param ticks: list of TickMarks to match
        :param max_delta:maximum deviation of coordinates between text and ticks
        :return: list of (scale_text, tick) tuples (unsorted)
        """
        scale_text2tick_list = []
        for tick in ticks:
            assert tick.perpendicular == X or tick.perpendicular == Y, f"perp {tick.perpendicular}"
            for scale_text in scale_texts:
                scale_coord = scale_text.centroid_coord(tick.perpendicular)
                if abs(tick.coord - scale_coord) < max_delta:
                    scale_text2tick_list.append((scale_text.text, tick))

        return scale_text2tick_list


class AmiPlot:

    def __init__(self, bbox=None, image_file=None, ami_graph=None):
        self.bbox = bbox
        self.image_file = image_file
        self.ami_graph = ami_graph
        self.axial_box_by_side = dict()
        self.plot_island = None
        # resources
        self.ami_edges = None
        self.horiz_ami_lines = None
        self.vert_ami_lines = None
        self.words = None
        self.word_bboxes = None
        # scales
        self.bottom_scale = None
        self.left_scale = None
        self.top_scale = None
        self.right_scale = None

        if image_file:
            from pyamiimage.ami_graph_all import AmiGraph # TODO resolve AmiPlot and AmiGraph
            self.ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(image_file)
        return

    def get_axial_box(self, side=PlotSide.LEFT, low_margin=10, high_margin=10):
        """
        create a box for the axial spine and optional margins

        if margins are zero this is a zero-width box exactly covering the axial spine
        It may be advisable to expand the box along the spine to catch ticks with same
        coordinate as spine ends

        :param side: side of box (see PlotSide)
        :param low_margin: decrease lowvalue of non-side range
        :param high_margin: increase highvalue of non-side range
        """
        assert side in PlotSide.SIDES
        axial_bbox = self.axial_box_by_side.get(side)
        if self.bbox and not axial_bbox:
            xrange = self.bbox.get_xrange()
            yrange = self.bbox.get_yrange()
            xy_ranges_ = None
            if side == PlotSide.LEFT:
                xy_ranges_ = [[xrange[0] - low_margin, xrange[0] + high_margin], yrange]
            elif side == PlotSide.RIGHT:
                xy_ranges_ = [[xrange[1] - low_margin, xrange[1] + high_margin], yrange]
            elif side == PlotSide.TOP:
                xy_ranges_ = [xrange, [yrange[0] - low_margin, yrange[0] + high_margin]]
            elif side == PlotSide.BOTTOM:
                xy_ranges_ = [xrange, [yrange[1] - low_margin, yrange[1] + high_margin]]

            axial_bbox = BBox(xy_ranges=xy_ranges_)
            self.axial_box_by_side[side] = axial_bbox
        return axial_bbox

    def clear_axial_boxes(self):
        self.axial_box_by_side = dict()

    def add_axial_polylines_to_ami_lines(self, ami_edges, horiz_ami_lines, vert_ami_lines, tolerance=2):
        # TODO not in right class?
        from pyamiimage.ami_graph_all import AmiEdge  # horrible TODO have to fix this
        axial_polylines = AmiEdge.get_axial_polylines(ami_edges, tolerance=tolerance)
        for axial_polyline in axial_polylines:
            for ami_line in axial_polyline:
                if ami_line.is_vertical(tolerance=tolerance):
                    vert_ami_lines.append(ami_line)
                elif ami_line.is_horizontal(tolerance=tolerance):
                    horiz_ami_lines.append(ami_line)
                else:
                    raise ValueError(f"line {ami_line} must be horizontal or vertical")

    def create_scaled_plot_box(self, island_index=0, maxmindim=10000, mindim=0):
        from pyamiimage.ami_graph_all import AmiEdge  # TODO resolve imports

        plot_islands = self.ami_graph.get_or_create_ami_islands(mindim=mindim, maxmindim=maxmindim)
        if len(plot_islands) <= island_index:
            raise ValueError(f"not enough islands {len(plot_islands)}, wanted {island_index}")
        self.plot_island = plot_islands[island_index]
        self.bbox = self.plot_island.get_or_create_bbox()
        self.ami_edges = self.plot_island.get_or_create_ami_edges()
        self.horiz_ami_lines = AmiEdge.get_horizontal_lines(self.ami_edges)
        if len(self.horiz_ami_lines) == 0:
            print(f"cannot find any horiz lines, edges {self.edges}")
        self.vert_ami_lines = AmiEdge.get_vertical_lines(self.ami_edges)

        self.left_scale = AmiScale()
        self.left_scale.box = self.get_axial_box(side=PlotSide.LEFT, low_margin=100, high_margin=50)
        self.left_scale.box.change_range(1, 3)
        self.left_scale.ticks = TickMark.get_tick_marks(self.horiz_ami_lines, self.left_scale.box, Y)

        print (f"ticks {self.left_scale.ticks}")
        self.bottom_scale = AmiScale()
        self.bottom_scale.box = self.get_axial_box(side=PlotSide.BOTTOM, high_margin=50)
        self.bottom_scale.box.change_range(1, 3)
        self.bottom_scale.ticks = TickMark.get_tick_marks(self.vert_ami_lines, self.bottom_scale.box, X)

        print(f"bottom ticks {self.bottom_scale.ticks}")

        # axial polylines can be L- or U-shaped
        self.add_axial_polylines_to_ami_lines(self.ami_edges, self.horiz_ami_lines, self.vert_ami_lines)

        # word_numpys, self.words = TesseractOCR.extract_numpy_box_from_image(self.image_file)
        # self.word_bboxes = [BBox.create_from_numpy_array(word_numpy) for word_numpy in word_numpys]
        # print(f" wordzz {self.words}")

        # defaults to easyocr
        ami_ocr = AmiOCR(self.image_file)
        text_boxes = ami_ocr.get_textboxes()
        # assert 20 <= len(text_boxes) <= 22, f"text_boxes found {len(text_boxes)}"
        for text_box in text_boxes:
            print(f"text box {text_box}")


        self.bottom_scale.text2coord_list = self.bottom_scale.match_scale_text_box2ticks(text_boxes)
        self.left_scale.text2coord_list = self.left_scale.match_scale_text_box2ticks(text_boxes)
        self.bottom_scale.get_numeric_ticks()
        self.bottom_scale.calculate_offset_scale()


class AmiScale:

    def __init__(self):
        self.text2coord_list = None
        self.scale_text2tick_list = []
        self.box = None
        self.ticks = None
        self.text_values = None
        self.numeric_ticks = []

        """
        to convert from user coords to plot coords
        plot_coord = self.user_num_to_plot_offset + user_to_plot_scale * user_coord
        """
        self.user_to_plot_scale = None
        self.user_num_to_plot_offset = None

    def match_scale_text2ticks(self, word_bboxes, words, del_regex=None):
        # TODO get rid of self.scale_text2tick_list (maybe a dict())
        self.text_values = []
        self.scale_text2tick_list = []
        if not self.ticks:
            print(f"no ticks")
            return
        if len(self.ticks) < 2:
            print(f"only one tick")
            return

        for bbox, word in zip(word_bboxes, words):
            if self.box.contains_bbox(bbox):
                self.text_values.append(ScaleText(word, bbox))
        self.scale_text2tick_list = TickMark.match_ticks_to_text(self.text_values, self.ticks)
        for scale_text2tick in self.scale_text2tick_list:
            scale_text = scale_text2tick[0]
            if del_regex:
                scale_text = re.sub(del_regex, '', scale_text)
            scale_text2tick[1].user_num = AmiUtil.get_float(scale_text)
            scale_text2tick[1].user_text = scale_text

        return self.scale_text2tick_list

    def match_scale_text_box2ticks(self, text_boxes):
        # TODO get rid of self.scale_text2tick_list (maybe a dict())
        self.text_values = []
        self.scale_text2tick_list = []
        if not self.ticks:
            print(f"no ticks")
            return
        if len(self.ticks) < 2:
            print(f"only one tick")
            return

        for text_box in text_boxes:
            if self.box.contains_bbox(text_box.bbox):
                self.text_values.append(ScaleText(text_box.text, text_box.bbox))
        self.scale_text2tick_list = TickMark.match_ticks_to_text(self.text_values, self.ticks)
        for scale_text2tick in self.scale_text2tick_list:
            scale_text = scale_text2tick[0]
            scale_text2tick[1].user_num = AmiUtil.get_float(scale_text)
            scale_text2tick[1].user_text = scale_text

        return self.scale_text2tick_list

    def get_numeric_ticks(self):
        self.numeric_ticks = [stt[1] for stt in self.scale_text2tick_list if stt[1].user_num is not None]
        self.numeric_ticks = sorted(self.numeric_ticks, key=lambda n : n.user_num)
        # print(f"numericR {self.numeric_ticks}")

    def calculate_offset_scale(self):
        if not self.numeric_ticks:
            print(f" no numeric ticks")
            return
        if len(self.numeric_ticks) == 1:
            print(f" only one numeric tick")
            return
        tick_lo = self.numeric_ticks[0]
        tick_hi = self.numeric_ticks[-1]
        delta_plot = tick_hi.coord - tick_lo.coord
        delta_user = tick_hi.user_num - tick_lo.user_num
        self.user_to_plot_scale = delta_plot / delta_user
        self.user_num_to_plot_offset = tick_lo.coord - tick_lo.user_num * self.user_to_plot_scale

    def convert_plot_coords_to_user(self, plot_coord):
        return (plot_coord - self.user_num_to_plot_offset) / self.user_to_plot_scale

class AmiLine:
    """This will probably include a third-party tool supporting geometry for lines

    It may also be a polyline
    """

    def __init__(self, xy12=None, ami_edge=None):
        """xy12 of form [[x1, y1], [x2, y2]]
        direction is xy1 -> xy2 if significant"""
        self.ami_edge = ami_edge
        self.xy1 = None
        self.xy2 = None
        if xy12 is not None:
            if len(xy12) != 2 or len(xy12[0]) != 2 or len(xy12[1]) != 2:
                raise ValueError(f"bad xy pair for line {xy12}")
            self.xy1 = [xy12[0][X], xy12[0][Y]]
            self.xy2 = [xy12[1][X], xy12[1][Y]]
        self._bbox = None
        # ami_nodes = []

    def __repr__(self):
        return str([self.xy1, self.xy2])

    def __str__(self):
        return str([self.xy1, self.xy2])

    def set_ami_edge(self, ami_edge):
        self.ami_edge = ami_edge

    @property
    def vector(self):
        """vector between end points
        :return: xy2 - xy1
        """
        return None if (self.xy1 is None or self.xy2 is None) else \
            (self.xy2[X] - self.xy1[X], self.xy2[Y] - self.xy1[Y])

    @property
    def xy_mid(self):
        """get midpoint of line
        :return: 2-array [x, y] of None if coords not set"""

        if self.xy1 is not None and self.xy2 is not None:
            return [(self.xy1[X] + self.xy2[X]) / 2, (self.xy1[Y] + self.xy2[Y]) // 2]
        return None

    @property
    def bbox(self):
        """get bbox for line
        """
        if not self._bbox and self.xy1 and self.xy2:
            self._bbox = BBox([[self.xy1[X], self.xy2[X]], [self.xy1[Y], self.xy2[Y]]])

        return self._bbox

    def is_horizontal(self, tolerance=1) -> int:
        return abs(self.vector[Y]) <= tolerance < abs(self.vector[X])

    def is_vertical(self, tolerance=1) -> int:
        return abs(self.vector[X]) <= tolerance < abs(self.vector[Y])

    @classmethod
    def get_horiz_vert_counter(cls, ami_lines, xy_index) -> Counter:
        """
        counts midpoint coordinates of lines (normally ints)

        :param ami_lines: horiz or vert ami_lines
        :param xy_index: 0 (x) 0r 1 (y) (normally 0 for vert lines, 1 for horiz)
        :return: Counter
        """
        hv_dict = Counter()
        for ami_line in ami_lines:
            xy_mid = ami_line.xy_mid[xy_index]
            if xy_mid is not None:
                hv_dict[int(xy_mid)] += 1
        return hv_dict

    def get_min(self, xy_flag):
        return None if xy_flag is None else min(self.xy1[xy_flag], self.xy2[xy_flag])

    def get_max(self, xy_flag):
        return None if xy_flag is None else max(self.xy1[xy_flag], self.xy2[xy_flag])


class AmiPolyline:
    """polyline. can represent solid or dashed (NYI) lines. contains attachment points
    """

    def __init__(self, points_list=None, ami_edge=None, tolerance=1):
        """
        polyline contains a list of connected points and also
        :param points_list:
        :param ami_edge:
        :param tolerance:
        """
        self.ami_edge = ami_edge
        self.points_list = []  # 2D coords
        self.bbox = None
        if points_list:
            for point in points_list:
                self.points_list.append([point[0], point[1]])
        self.tolerance = tolerance

    @property
    def id(self):
        idd = None
        if self.points_list:
            p0 = self.points_list[0]
            idd = str(p0[0]) + "_" + str(p0[1])
            idd += "__"
            p1 = self.points_list[-1]
            idd += str(p1[0]) + "_" + str(p1[1])
        return idd

    def get_bounding_box(self):
        if not self.bbox and self.points_list:
            self.bbox = BBox()
            for point in self.points_list:
                self.bbox.add_coordinate(point)
            if self.tolerance and self.tolerance > 0:
                self.bbox.expand_by_margin(self.tolerance)
        return self.bbox

    def __repr__(self):
        return str(self.points_list)

    def __str__(self):
        return str(self.points_list)

    @property
    def vector(self):
        """vector between end points
        :return: xy2 - xy1
        """
        pl = self.points_list
        return [pl[0][X] - pl[-1][X], pl[1][Y] - pl[1][Y]]

    @property
    def xy_mid(self):
        """get midpoint of line
        :return: 2-array [x, y] of None if coords not set"""

        if self.points_list and len(self.points_list) > 1:
            pl = self.points_list
            return [(pl[0][X] - pl[-1][X]) // 2, (pl[1][Y] - pl[1][Y]) // 2]
        return None

    def is_horizontal(self, tolerance=1) -> int:
        return abs(self.vector[Y]) <= tolerance < abs(self.vector[X])

    def is_vertical(self, tolerance=1) -> int:
        return abs(self.vector[X]) <= tolerance < abs(self.vector[Y])

    @classmethod
    def get_horiz_vert_counter(cls, ami_lines, xy_index) -> Counter:
        """
        NYI
        counts midpoint coordinates of lines (normally ints)

        :param ami_lines: horiz or vert ami_lines
        :param xy_index: 0 (x) 0r 1 (y) (normally 0 for vert lines, 1 for horiz)
        :return: Counter
        """
        hv_dict = Counter()
        for ami_line in ami_lines:
            xy_mid = ami_line.xy_mid[xy_index]
            if xy_mid is not None:
                hv_dict[int(xy_mid)] += 1
        return hv_dict

    def range(self, tolerance=None):
        """range from min to max; only works for axial lines
        if start/end points are the same return None

        """
        if tolerance is None:
            tolerance = self.tolerance
        if not self.points_list or len(self.points_list) < 2:
            return None
        dx = self.points_list[-1][0] - self.points_list[0][0]
        dy = self.points_list[-1][1] - self.points_list[0][1]

        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return None
        if abs(dx) <= tolerance:
            ll = [self.points_list[-1][1], self.points_list[0][1]] if dy < 0 else [self.points_list[0][1],
                                                                                   self.points_list[-1][1]]
            return ll
        if abs(dy) <= tolerance:
            ll = [self.points_list[-1][0], self.points_list[0][0]] if dy < 0 else [self.points_list[0][0],
                                                                                   self.points_list[-1][0]]
            return ll
        return None
        # raise ValueError(f"cannot calculate range {self.points_list}")

    def get_attachment_points(self):
        if self.points_list and len(self.points_list) >= 2:
            return self.points_list[1:-1]
        return None

    def get_cartesian_length(self):
        """gets length for axial polylines
        :return: abs distance in axial coordinate else NaN"""
        range_ = self.range()
        return float("NaN") if range_ is None else abs(range_[0] - range_[1])

    def find_points_in_box(self, bbox):
        """iterates over all points including ends in polyline
        :param bbox: BBox within which point must fit
        :return: """
        points_in_box = []
        size = len(self.points_list)
        for i, point in enumerate(self.points_list):
            if bbox.contains_point(point):
                points_in_box.append((i, i - size, point))
        return points_in_box

    def number_of_points(self):
        """
        :return: 0 if no points_list else number of points
        """
        return len(self.points_list) if self.points_list else 0

    def split_line(self, point_triple):
        """split polyline at point
        :param point_triple: triple created by AmiPolyline.find_points_in_box() (index_left, index_right, coords)
        :return: two lines, if one is length 0, nul and orginal polyline
        """
        lines = [None, None]
        ll = self.number_of_points()
        if point_triple[0] == 0:
            lines[1] = self
        elif point_triple[0] == -1:
            lines[0] = self
        else:
            lines[0] = self.sub_polyline(0, point_triple[0])
            lines[1] = self.sub_polyline(point_triple[0], ll - 1)

        return lines

    def sub_polyline(self, index0, index1):
        """slice line at points , keeping both
        not pythonic
        """
        polyline = AmiPolyline(self.points_list[index0:index1 + 1])
        return polyline


class AmiLineTool:
    """joins points or straight line segments
    final target can be:
    may contain tools for further analysis
    still being developed.
    end
    mode=SEGMENT assembles segments into one of more polylines
    """

    def __init__(self, tolerance=1, points=None, ami_edge=None, mode=POLYLINE, xy_flag=None):
        """build higher-order objects frome line segments or points
        :param tolerance: for determing overlapping points
        :param points: points to make polyline/gons
        :param ami_edge: possibly obsolete
        :param mode: POLYLINE (default)
        :"""
        self.point_set = set()
        self.points = []
        self.tolerance = tolerance
        self.mode = mode
        self.xy_flag = xy_flag
        if mode not in TARGETS:
            raise ValueError(f"unknown target {mode}")
        if ami_edge is not None:
            raise ValueError("emi_edges not yet supported")
        if points is not None:
            for point in points:
                self.add_point(point)
        ami_polylines = []  # NYI
        self.line_points_list = []  # obsolete?
        self.polygons = []  # obsolete?

    def __repr__(self):
        return str(self.points)

    def __str__(self):
        return str(self.points)

    def add_point(self, point):
        """add point
        :param point:
        """
        AmiLineTool._validate_point(point)
        # if point in self.point_set:
        #     raise ValueError("Cannot add point twice {point}")
        if point in self.points:
            raise ValueError("Cannot add point twice {point}")
        self.points.append(point)

    def points(self):
        return self.points

    def add_segment(self, segment):
        """segment tail must match head

        Segment is of form [  [x1,y1], [x2, y2] ]
        Polyline is [[x1,y1], [x2, y2], [x3, y3] ... ]
        poly_list - is [[[x1,y1], [x2, y2], [x3, y3] ... ], [[x4,y4], [x5, y5], [x6, y6] ... ]]
        """
        if type(segment) is AmiLine:
            segment = [segment.xy1, segment.xy2]
        self._validate_segment(segment)
        if self.mode == POLYLINE or self.mode == POLYGON:
            polyline = [segment[0], segment[1]]
            self.add_merge_polyline_to_poly_list(polyline)
        else:
            # if len(self.points) == 0:
            #     self.add_point(segment[0])
            #     self.add_point(segment[1])
            # elif self._are_coincident_points(self.points[-1], segment[0]):
            #     self.add_point(segment[1])
            # else:
            #     raise ValueError("Cannot add segment {segment} (no overlap)")
            pass

    def add_segments(self, segments):
        """
        add segments piecewise to self.polylines
        recommend sorting highest first e.g.
        [
        [[2, 3], [5, 7]],
        [[1, 2], [2, 3]]
        ]
        rather than
        [
        [[1, 2], [2, 3]],
        [[2, 3], [5, 7]]
        ]
        :param segments:
        :return:
        """
        for segment in segments:
            self.add_segment(segment)

    def _are_coincident_points(self, point1, point2):
        """

        overlaps if either dx or dy >= self.tolerance
        :param point1: [x, y]
        :param point2: [x, y]
        :return:
        """
        AmiLineTool._validate_point(point1)
        AmiLineTool._validate_point(point2)
        return AmiUtil.are_coincident(point1, point2, self.tolerance)

    def _validate_segment(self, segment):
        if type(segment) is list:
            AmiLineTool._validate_point(segment[0])
            AmiLineTool._validate_point(segment[1])
        else:
            assert type(segment) is AmiLine, f"segment was {type(segment)}"
            AmiLineTool._validate_point(segment[0])
            AmiLineTool._validate_point(segment[1])

    @classmethod
    def _validate_point(cls, point, message=""):
        assert type(point) is list, f"{message} found {type(point)}"
        assert len(point) == 2, f"{message} found {len(point)}"
        assert AmiUtil.is_number(point[0])

    def _copy_point(self, point):
        AmiLineTool._validate_point(point)
        return [point[0], point[1]]

    def copy_segment(self, segment):
        return [self._copy_point(segment[0]), self._copy_point(segment[1])]

    def add_points(self, points):
        assert type(points) is list
        for point in points:
            self.add_point(point)

    # def insert_point(self, pos, point):
    #     AmiLineTool._validate_point(point)
    #     self.points.insert(0, [33, 44])
    #     self.points.insert(pos, point)

    def add_merge_polyline_to_poly_list(self, polyline_to_add):
        """
        adds/merges polylines using copy semantics.
        assumes the lines to be added are suitable for merging into polylines or polygons
        (i.e. all final nodes will be 1 or 2 - connected


        :param polyline_to_add: polyline to add (may be anynumber of points, 1, 2, many
        """
        if not self.line_points_list:
            polylinex = []
            for point in polyline_to_add:
                polylinex.append([point[0], point[1]])
            self.line_points_list.append(polylinex)
        else:
            added_polyline = None
            # have to consider directions of lines
            for polyline in self.line_points_list:
                added_polyline = self.join_heads_and_tails(polyline, polyline_to_add)
                if added_polyline is not None:
                    break

            if added_polyline is not None:
                self.find_cycles(added_polyline, polyline)
            else:
                self.copy_and_append(polyline_to_add)

    def find_cycles(self, added_polyline, polyline):
        """is the new polyline cyclic
        if so, remove polyline and convert to polygon
        """
        if self._are_coincident_points(added_polyline[0], added_polyline[-1]):
            logger.debug(f"CYCLIC {added_polyline}")
            if self.mode == POLYGON:
                self.line_points_list.remove(polyline)
                point0 = polyline[0]
                polyline.remove(point0)
                self.polygons.append(polyline)

    def join_heads_and_tails(self, polyline, polyline_to_add):
        """join two polylines with
         :param polyline: existing loyline in self.polylines
         :param polyline_to_add: polyline being added

         Each line has a tail (0) and head (-1) There are 4 possible
         joinings HH, HT, TH, TT (or none).

         """
        added_polyline = None
        if self._are_coincident_points(polyline_to_add[-1], polyline[0]):
            # tail-head
            logger.debug(f"add truncated {polyline_to_add} to {polyline}")
            self.copy_prepend(polyline_to_add[:-1][::-1], polyline)
            added_polyline = polyline

        elif self._are_coincident_points(polyline[0], polyline_to_add[0]):
            # tail-tail
            logger.debug(f"tail-tail {polyline_to_add} to {polyline}")
            logger.debug(f"{polyline_to_add} => {polyline_to_add[1:]} to {polyline}")
            self.copy_prepend(polyline_to_add[1:], polyline)
            added_polyline = polyline

        elif self._are_coincident_points(polyline_to_add[0], polyline[-1]):
            # head-tail
            self.copy_append(polyline_to_add[1:], polyline)
            added_polyline = polyline

        elif self._are_coincident_points(polyline_to_add[-1], polyline[-1]):
            # head-head
            add_ = polyline_to_add[::-1][1:]
            logger.debug(f" adding {polyline_to_add} => {add_} to {polyline}")
            self.copy_append(add_, polyline)
            added_polyline = polyline

        else:
            logger.debug(f" polyline_to_add {polyline_to_add} does not overlap {polyline}")

        return added_polyline

    @classmethod
    def copy_append(cls, line_points_to_add, target_line_points):
        logger.debug(f"adding {line_points_to_add} to {target_line_points}")
        for point in line_points_to_add:
            point1 = [point[0], point[1]]
            target_line_points.append(point1)

    @classmethod
    def copy_prepend(cls, line_points_to_add, line_points_list):
        """prepends polyline_to_add to front of list
        uses insert(0, ...) which is SLOW unless lines are deques
        """
        for point in line_points_to_add:
            point1 = [point[0], point[1]]
            line_points_list.insert(0, point1)

    def copy_and_append(self, polyline_to_add):
        """
        copies into new polyline and adds this to self.polylines
        :param polyline_to_add: polyline to add may have 1,2, more points
        """
        line_points = []
        for point in polyline_to_add:
            line_points.append([point[0], point[1]])
        self.line_points_list.append(line_points)


class AmiEdgeTool:
    """refines edges (join, straighten, break, corners, segments, curves, etc some NYI)
    Still being actively developed
    """

    def __init__(self, ami_graph=None, ami_edges=None, ami_nodes=None):
        """Best to create this from the factory method create_tool"""
        self.ami_graph = ami_graph
        self.ami_edges = ami_edges
        self.ami_nodes = ami_nodes

    @classmethod
    def create_tool(cls, ami_graph, ami_edges=None, ami_nodes=None):
        """preferred method of instantiating tool
        :param ami_graph: required graph
        :param ami_edges: edges to process
        :param ami_nodes: optional nodes (if none uses ends of edges)
        """
        edge_tool = AmiEdgeTool(ami_graph, ami_edges=ami_edges, ami_nodes=ami_nodes)
        if not ami_nodes:
            edge_tool.create_ami_nodes_from_edges()

        return edge_tool

    def analyze_topology(self):
        """counts nodes and edges by recursively iterating over
        noes and their edges -> edges and their nodes
        also only includes start_id < end_id
        (mainly a check)

         :return: nodes, edges"""

        if self.ami_edges is None:
            logger.error(f"no edges, possible error")
        new_ami_edges = set()
        new_ami_nodes = set()
        while self.ami_nodes:
            ami_node = self.ami_nodes.pop()
            new_ami_nodes.add(ami_node)
            node_ami_edges = ami_node.get_or_create_ami_edges()
            for ami_edge in node_ami_edges:
                if ami_edge.has_start_lt_end():
                    if ami_edge not in self.ami_edges:
                        logger.error(f" cannot find {ami_edge} in edges")
                    else:
                        new_ami_edges.add(ami_edge)
        return new_ami_nodes, new_ami_edges

    def create_ami_nodes_from_edges(self):
        """generates unique ami_nodes from node_ids at ends of edges"""
        if not self.ami_nodes:
            self.ami_nodes = set()
            node_ids = set()
            for ami_edge in self.ami_edges:
                node_ids.add(ami_edge.start_id)
                node_ids.add(ami_edge.end_id)
            self.ami_nodes = self.ami_graph.create_ami_nodes_from_ids(node_ids)
