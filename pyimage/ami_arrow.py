import logging
import math
# local
from ..pyimage.ami_graph_all import AmiNode, AmiEdge, AmiIsland
from ..pyimage.svg import SVGG, SVGLine, SVGCircle
from ..pyimage.ami_util import AmiUtil

logger = logging.getLogger(__name__)


class AmiArrow:
    HEAD_WOBBLE = 0.8 * math.pi  # allowed deviation from straightness between shaft and head
    BARB_ANGLE_DIFF = 0.1 * math.pi  # allowed asymmetry in barb angles
    MAX_BARB_TO_TAIL_ANGLE = 0.9 * math.pi / 2  # maximum angle of barbs to tail

    def __init__(self, ami_island=None):
        self.ami_island = ami_island
        if self.ami_island is None:
            raise ValueError(f"AmiArrow must have an island")
        self.point_id = None
        self.head_id = None
        self.tail_id = None
        self.barb_ids = None
        self.point_xy = None
        self.tail_xy = None

    def __str__(self):
        s = f"tail {self.tail_id} - head {self.head_id} > point {self.point_id} barbs {self.barb_ids}"
        return s

    def set_tail_xy(self, yx):
        self.tail_xy = AmiUtil.swap_yx_to_xy(yx)

    def set_point_xy(self, yx):
        self.point_xy = AmiUtil.swap_yx_to_xy(yx)

    # @classmethod
    # def annotate_graph(cls, nx_graph):
    #     cls.annotate_4_nodes(nx_graph)
    #     cls.annotate_3_nodes(nx_graph)
    #     cls.annotate_1_nodes(nx_graph)

    @classmethod
    def create_simple_arrow(cls, island):
        """
        create simple arrow from island
        one head, one tail
        :param island:
        :return: AmiArrow or none
        """
        if not 5 >= len(island.node_ids) >= 4:
            logger.warning(f"cannot create simple arrow from {island.node_ids}")
            return None
        node_dict = island.create_node_degree_dict()
        logger.debug(f"\nnode dict {node_dict}")
        neighbour_count = len(island.node_ids) - 1
        try:
            central_node_id = node_dict[neighbour_count][0]
            logger.debug(f"central node {central_node_id}")
        except Exception as e:
            return None

        edge_dict = island.create_edge_property_dikt(central_node_id)
        logger.debug(f"edge_dict {edge_dict}")

        longest_dict = cls._find_dict_with_longest_edge(edge_dict)
        longest_edge = AmiEdge(ami_graph=island.ami_graph, start=central_node_id, end=longest_dict[AmiNode.REMOTE], branch_id=0)
        logger.debug(f"longest edge {longest_edge}")
        logger.debug(f"longest dict {longest_dict}")

        ami_arrow = AmiArrow(island)
        ami_arrow.head_id = central_node_id
        ami_arrow.tail_id = longest_dict[AmiNode.REMOTE]

        # find short lines
        short_lines = [value for value in edge_dict.values() if value != longest_dict]
        logger.debug(f"short lines {short_lines} {len(short_lines)}")

        if len(short_lines) == 3:  # 5-node arrow - normally from thinned solid triangle

            # find straight on
            barbs1 = []
            arrow_point_line = None
            for line in short_lines:
                line_tuple = (central_node_id, line[AmiNode.REMOTE], 0)  # assume only one branch
                angle = island.ami_graph.get_interedge_tuple_angle(longest_edge.get_tuple(), line_tuple)
                assert angle is not None
                if abs(angle) > AmiArrow.HEAD_WOBBLE:
                    if arrow_point_line is not None:
                        raise ValueError(f"competing lines for head line {longest_edge} to {short_lines}")
                    arrow_point_line = line
                else:
                    barbs1.append(line)
            if arrow_point_line is None:
                raise ValueError(f"cannot find point line {longest_edge} to {short_lines}")
            ami_arrow.point_id = arrow_point_line[AmiNode.REMOTE]
            if ami_arrow.point_id is None:
                raise ValueError(f"cannot find point {arrow_point_line} , {central_node_id}")
        else:  # 4 node arrow
            barbs1 = short_lines
            ami_arrow.point_id = ami_arrow.head_id
        barbs = barbs1
        ami_arrow.set_point_xy(AmiNode.get_xy_for_node_id(island.ami_graph.nx_graph, ami_arrow.point_id))
        ami_arrow.set_tail_xy(AmiNode.get_xy_for_node_id(island.ami_graph.nx_graph, ami_arrow.tail_id))
        print(f"nodes head {ami_arrow.point_xy} tail {ami_arrow.tail_xy}")
        logger.debug(f"longest {longest_edge} point {ami_arrow.point_id} barbs {barbs}")
        if len(barbs) != 2:
            raise ValueError(f" require exactly 2 barbs on arrow {barbs}")
        ami_arrow.barb_ids = [barb[AmiNode.REMOTE] for barb in barbs]

        barb_angles = [island.ami_graph.get_angle_between_nodes(ami_arrow.tail_id, ami_arrow.head_id, barb_id) for barb_id in ami_arrow.barb_ids]
        logger.debug(f"barb angles {barb_angles}")
        if abs(barb_angles[0] + barb_angles[1]) > AmiArrow.BARB_ANGLE_DIFF:
            raise ValueError(f"barb angles not symmetric {barb_angles}")
        if abs(barb_angles[0] - barb_angles[1]) / 2 > AmiArrow.MAX_BARB_TO_TAIL_ANGLE:
            raise ValueError(f"barb angles not acute {barb_angles}")


        logger.debug(f"AMIARR {ami_arrow}")
        return ami_arrow

    @classmethod
    def _find_dict_with_longest_edge(cls, edge_dict):
        longest_dict = None
        for key, value in edge_dict.items():
            if longest_dict is None:
                longest_dict = value
            else:
                if value[AmiNode.PIXLEN] > longest_dict[AmiNode.PIXLEN]:
                    longest_dict = value
        return longest_dict

    def get_svg(self):
        """
        create line with arrowhead
        :return:
        """
        g = SVGG()
        line = SVGLine(self.point_xy, self.tail_xy)
        line.set_attribute("stroke", "red")
        line.set_attribute("width", "2.0")
        g.append(line)
        circle = SVGCircle(self.point_xy, rad=10)
        g.append(circle)
        return g


# ----------- utils -----------
