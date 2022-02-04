import logging
# local
from ..pyimage.ami_plot import AmiLine
from ..pyimage.ami_plot import POLYLINE
from ..pyimage.ami_plot import AmiLineTool
from ..pyimage.ami_graph_all import AmiEdge

logger = logging.getLogger(__name__)

COORD = 0
COUNT = 1
X = 0
Y = 1
XY = ["X", "y"]

""" terms
AmiEdge is a graph-edge between 2 AmiNodes. It need not be straight (but normally cannot intersect itself)
There can be 2 or more AmiEdges between 2 AmiNodes. It normally contains a thread of pixels "points";
 
AmiLine represents straight lines between 2 points (often AmiNodes, but often unnamed). 
an AmiEdge containa 0, 1 or many AmiLines.
AmiLines should retain backreferences to the AmiEdges - not fully implemented.

X is horizontal
Y is vertical (normally increasing down the page, origin is top-left)

"""


class AmiEdgeAnalyzer:
    """
    contains and analyzes edges, building higher-level objects
    """

    def __init__(self, tolerance=1):
        self.horizontal_edges = None
        self.vertical_edges = None
        self.horiz_ami_lines = None
        self.vert_ami_lines = None

        self.axial_polylines = None
        self.vert_dict = []
        self.horiz_dict = []
        self.tolerance = tolerance

        self.line_x_coords_by_count = None
        self.line_y_coords_by_count = None

        # temporary work variables


    def read_edges(self, ami_edges):
        self.horizontal_edges = AmiEdge.get_horizontal_edges(ami_edges, tolerance=self.tolerance)
        self.vertical_edges = AmiEdge.get_vertical_edges(ami_edges, tolerance=self.tolerance)
        self.horiz_ami_lines = AmiEdge.get_single_lines(self.horizontal_edges)
        self.vert_ami_lines = AmiEdge.get_single_lines(self.vertical_edges)

        self.axial_polylines = AmiEdge.get_axial_polylines(ami_edges, tolerance=self.tolerance)

        self.extract_lines_from_polylines()

        self.vert_dict = AmiLine.get_horiz_vert_counter(self.vert_ami_lines, xy_index=0)
        self.horiz_dict = AmiLine.get_horiz_vert_counter(self.horiz_ami_lines, xy_index=1)

    def extract_lines_from_polylines(self):
        """find horizonatl and verticakl segments in polylines/edges"""
        logger.debug(f"extracting lines from axial polylines {len(self.axial_polylines)}")
        for axial_polyline in self.axial_polylines:
            logger.debug(f"axial polyline: {axial_polyline}")
            for ami_line in axial_polyline:
                if ami_line.is_vertical(tolerance=self.tolerance):
                    self.vert_ami_lines.append(ami_line)
                elif ami_line.is_horizontal(tolerance=self.tolerance):
                    self.horiz_ami_lines.append(ami_line)
                else:
                    raise ValueError(f"line {ami_line} must be horizontal or vertical")

    def merge_neighbouring_coords(self) -> tuple:
        """
        create Counters for vertical lines (x coord) and horizontal (y-coords)
        :return: x_counter, y_counter
        """
        self.line_x_coords_by_count = self._merge_close_bins(self.vert_dict.most_common())
        self.line_y_coords_by_count = self._merge_close_bins(self.horiz_dict.most_common())
        return self.line_x_coords_by_count, self.line_y_coords_by_count

    def _merge_close_bins(self, coord_counts):
        """merge counts of bins within tolerance

        :param coord_counts: Counter of count by coordinate
        :return: updated counter"""

        filtered_counts = []
        while len(coord_counts) > 0:
            coord_count = coord_counts[0]
            change = False
            for filtered_count in filtered_counts:
                if abs(filtered_count[COORD] - coord_count[COORD]) <= self.tolerance:
                    filtered_count[COUNT] += coord_count[COUNT]
                    change = True
                    break
            if not change:
                filtered_counts.append([coord_count[COORD], coord_count[COUNT]])
            coord_counts.remove(coord_count)
        return filtered_counts

    # TODO move to AmiLineTool
    def join_ami_lines(self, xy_flag):
        """
        join lines with constant coordinate (vert/X, or horiz/Y)

        :param xy_flag: 0 or 1 for X/Y
        :param gap_factor: gap allowed to next line as ratio of length
        :return: joined lines as list

        default is lines must touch within self.tolerance. gap_factor = 1.0 gives
        ___   ___   ___ (uniformly spaded white/black)

        """

        other = 1 - xy_flag
        line_coords_by_count = self.line_x_coords_by_count if xy_flag == X else self.line_y_coords_by_count
        line_tool = AmiLineTool(mode=POLYLINE, xy_flag=xy_flag)
        for this_coord, count in line_coords_by_count:
            ami_lines = self.find_sorted_ami_lines_with_coord(xy_flag, this_coord)
            for segment in ami_lines:
                line_tool.add_segment(segment)
        return line_tool


        # if False:
        #     polylines = []
        #     for segment in ami_lines:
        #         new_ami_line = self.create_new_line_with_ascending_coords(segment, xy_flag)
        #         polylines.append(new_ami_line)
        #
        #     # sort on second xy coord; x or y determined by xy_flag/other
        #     polylines = sorted(polylines, key=lambda ami_linex: ami_linex.xy2[other])
        #     logger.debug(f"new lines {polylines}")
        #     long_ami_line = None
        #     long_ami_lines = []
        #     for segment in polylines:
        #         long_ami_line = self.add_ami_lines_to_long_lines(segment, long_ami_line)
        #     pass

    @classmethod
    def _delta(cls, point0, point1):
        return min(abs(point0[0] - point1[0]), abs(point0[0] - point1[1]))

    @classmethod
    def _join(cls, long_ami_line, ami_line):
        """joins growing end of long_ami_line to ami_line
        """
        long_ami_line.xy2 = ami_line.xy2
        long_ami_line.mid_points.append(ami_line.xy1)
        return long_ami_line

    def add_ami_lines_to_long_lines(self, ami_line, long_ami_line):
        """

        :param ami_line:
        :param long_ami_line:
        :return: list of long_ami_lines

        """
        long_ami_lines = []
        if long_ami_line is None:
            long_ami_line = ami_line
            long_ami_lines.append(long_ami_line)
            long_ami_line.mid_points = []
            return
        else:
            point0 = long_ami_line.xy2  # joinable point
            point1 = ami_line.xy1  # incoming joinable point
            if self._delta(point0, point1) <= self.tolerance:
                # close enough? join
                long_ami_line = self._join(long_ami_line, ami_line)
            else:
                # no, create new line
                long_ami_line = None

        return long_ami_lines

    def create_new_line_with_ascending_coords(self, ami_line, xy_flag):
        """NOT TESTED"""
        # [[[66, 61], [66, 131]], [[66, 131], [66, 185]],...
        other = 1 - xy_flag
        swap = ami_line.xy1[other] < ami_line.xy2[other]
        xy1 = [ami_line.xy1[xy_flag], ami_line.xy1[other]] if swap else [ami_line.xy2[xy_flag], ami_line.xy2[other]]
        xy2 = [ami_line.xy2[xy_flag], ami_line.xy2[other]] if swap else [ami_line.xy1[xy_flag], ami_line.xy1[other]]
        return AmiLine(xy12=[xy1, xy2])

    def find_sorted_ami_lines_with_coord(self, xy_flag, coord):
        """
        finds axial lines with coord within self.tolerance
        :param xy_flag: X or Y
        :param coord:
        :return: list of lines
        """
        ami_lines = []
        lines_to_search = self.vert_ami_lines if xy_flag == X else self.horiz_ami_lines
        for line in lines_to_search:
            if abs(line.xy_mid[xy_flag] - coord) <= self.tolerance:
                ami_lines.append(line)
        other_coord = 1 - xy_flag
        ami_lines = sorted(ami_lines, key=lambda linex: linex.xy1[other_coord])
        return ami_lines

    def create_line_tools(self, island):
        """create horizontal and vertical line_tools
        :param island:
        :return: horizontal line tool, vertical line tool

        """
        ami_edges = island.get_or_create_ami_edges()
        self.read_edges(ami_edges)
        self.merge_neighbouring_coords()
        vert_line_tool = self.join_ami_lines(X)
        horiz_line_tool = self.join_ami_lines(Y)
        return horiz_line_tool, vert_line_tool
