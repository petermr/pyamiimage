import logging
from collections import Counter
# local
from ..pyimage.ami_util import AmiUtil

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
        self.polylines = []

    def __repr__(self):
        return str(self.points)

    def __str__(self):
        return str(self.points)

    def add_point(self, point):
        """add point
        :param point:
        """
        self._validate_point(point)
        # if point in self.point_set:
        #     raise ValueError("Cannot add point twice {point}")
        if point in self.points:
            raise ValueError("Cannot add point twice {point}")
        self.points.append(point)

    def points(self):
        return self.points

    def add_segment(self, segment):
        """segment tail must match head"""
        self._validate_segment(segment)
        if self.mode == POLYLINE:
            self.add_segment_to_poly_list(segment)
        else:
            if len(self.points) == 0:
                self.add_point(segment[0])
                self.add_point(segment[1])
            elif self._are_overlapping(self.points[-1], segment[0]):
                self.add_point(segment[1])
            else:
                raise ValueError("Cannot add segment {segment} (no overlap)")

    def add_segments(self, segments):
        for segment in segments:
            self.add_segment(segment)

    def insert_segment(self, pos, segment):
        self._validate_segment(segment)
        if not self._are_overlapping(segment[1], self.points[0]):
            raise ValueError("non-overlapping insertion at pos {pos} {segment}")
        self.points.insert(pos, segment[0])

    def _are_overlapping(self, point1, point2):
        """

        overlaps if either dx or dy >= self.tolerance
        :param point1: [x, y]
        :param point2: [x, y]
        :return:
        """
        return abs(point1[X] - point2[X]) + abs(point1[Y] - point2[Y]) <= self.tolerance

    def _validate_segment(self, segment):
        assert type(segment) is AmiLine, f"segment was {type(segment)}"
        self._validate_point(segment.xy1)
        self._validate_point(segment.xy2)

    def _validate_point(self, point):
        assert type(point) is list, f"found {type(point)}"
        assert len(point) == 2, f"found {len(point)}"
        assert AmiUtil.is_number(point[0])

    def _copy_point(self, point):
        self._validate_point(point)
        return [point[0], point[1]]

    def copy_segment(self, segment):
        return [self._copy_point(segment[0]), self._copy_point(segment[1])]

    def add_points(self, points):
        assert type(points) is list
        for point in points:
            self.add_point(point)

    def insert_point(self, pos, point):
        self._validate_point(point)
        self.points.insert(0, [33, 44])
        self.points.insert(pos, point)

    def add_segment_to_poly_list(self, segment):
        """add segment, assuming current list is sorted
        polyline            polyline
        lo-----hi           lo----------hi
                   lo---hi
                   segment
        """
        # seg_low = segment.get_min(self.xy_flag)
        # seg_hi = segment.get_max(self.xy_flag)
        if self.polylines == []:
            self.polylines.append(segment)
            return
        last_polyline = None
        xyf = self.xy_flag
        inserted = False
        for i, polyline in enumerate(self.polylines):
            # does it overlap the next exactly?
            if self._are_overlapping(segment.xy1, polyline.xy1):
                if last_polyline is None:
                    # end of first polyline
                    self.prepend_segment_to_polyline(segment, polyline)
                    inserted = True
                elif self._are_overlapping(segment.xy1, last_polyline[1]):
                    # join in the middle of last_polyline and polyline
                    # join to predecessor
                    self.prepend_segment_to_polyline(segment, polyline)
                    # and then add last
                    self.append_polyline_to_polyline_and_remove(last_polyline, polyline)
                    # and clear the last
                    self.polylines.remove(last_polyline)
                    inserted = True

            elif segment.xy2[xyf] < polyline[0][xyf]:
                # falls behind preceding and leads last if any
                if last_polyline is None or segment.xy0[xyf] > (last_polyline[-1][xyf] + self.tolerance):
                    self.polylines.insert(i, segment)
                    inserted = True
                else:
                    raise ValueError(f"too big {segment} cannot insert")

            if inserted:
                break

            last_polyline = polyline

        if not inserted:
            if self._are_overlapping(last_polyline[-1], segment.xy2):
                self.polylines.append(last_polyline, segment.xy2)
            elif segment.xy1[xyf] < last_polyline[1][xyf]:
                raise f"cannot add segment at head {segment}"
            else:
                self.polylines.append(segment)

    def prepend_segment_to_polyline(self, segment, polyline):
        polyline.insert(0, segment.xy1)

    def append_polyline_to_polyline_and_remove(self, last_polyline, polyline):
        last_polyline.add(polyline)
        self.polylines.remove(polyline)


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
                        print(f" cannot find {ami_edge} in edges")
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
