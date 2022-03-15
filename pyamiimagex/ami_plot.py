import logging
from collections import Counter
# local
from pyamiimage.ami_util import AmiUtil
from pyamiimage.bbox import BBox

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


class AmiPlot:
    pass


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
            l = [self.points_list[-1][1], self.points_list[0][1]] if dy < 0 else [self.points_list[0][1], self.points_list[-1][1]]
            return l
        if abs(dy) <= tolerance:
            l = [self.points_list[-1][0], self.points_list[0][0]] if dy < 0 else [self.points_list[0][0], self.points_list[-1][0]]
            return l
        return None
        # raise ValueError(f"cannot calculate range {self.points_list}")

    def get_attachment_points(self):
        if self.points_list and len(self.points_list) >= 2:
            return self.points_list[1:-1]
        return None

    def get_cartesian_length(self):
        """gets length for axial polylines
        :return: abs distance in axial coordinate else NaN"""
        range = self.range()
        return float("NaN") if range is None else abs(range[0] - range[1])

    def find_points_in_box(self, bbox):
        """iterates over all points including ends in polyline
        :param bbox: BBox within which point must fit
        :return: """
        points_in_box = []
        size = len(self.points_list)
        for i, point in enumerate(self.points_list):
            if bbox.contains_point(point):
                points_in_box.append((i, i-size, point))
        return points_in_box

    def number_of_points(self):
        """
        :return: 0 if no points_list else number of points
        """
        return len(self.points_list) if self.points_list else 0

    def split_line(self, point_triple):
        """split polyline at point
        :param point_triple: triple created by AmiPolyline.find_points_in_box() (index_left, index_right, coords)
        :param polyline: to split
        :return: two lines, if one is length 0, nul and orginal polyline
        """
        lines = [None, None]
        l = self.number_of_points()
        if point_triple[0] == 0:
            lines[1] = self
        elif point_triple[0] == -1:
            lines[0] = self
        else:
            lines[0] = self.sub_polyline(0, point_triple[0])
            lines[1] = self.sub_polyline(point_triple[0], l-1)

        return lines

    def sub_polyline(self, index0, index1):
        """slice line at points , keeping both
        not pythonic
        """
        polyline = AmiPolyline(self.points_list[index0:index1+1])
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
        if self.line_points_list == []:
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
