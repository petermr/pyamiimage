import logging
import math
# local
from ..pyimage.ami_graph_all import AmiNode, AmiEdge

logger = logging.getLogger(__name__)

class AmiArrow:
    HEAD_WOBBLE = 0.8 * math.pi  # allowed deviation from straightness between shaft and head
    BARB_ANGLE_DIFF = 0.1 * math.pi  # allowed asymmetry in barb angles
    MAX_BARB_TO_TAIL_ANGLE = 0.9 * math.pi / 2  # maximum angle of barbs to tail

    def __init__(self, ami_island=None):
        self.ami_island = ami_island
        if self.ami_island is None:
            raise ValueError(f"AmiArrow mush have an island")
        self.point_id = None
        self.head_id = None
        self.tail_id = None
        self.barb_ids = None

    def __str__(self):
        s = f"tail {self.tail_id} - head {self.head_id} > point {self.point_id} barbs {self.barb_ids}"
        return s

    @classmethod
    def annotate_graph(cls, nx_graph):
        cls.annotate_4_nodes(nx_graph)
        cls.annotate_3_nodes(nx_graph)
        cls.annotate_1_nodes(nx_graph)

    @classmethod
    def create_arrow(cls, island):
        if not 5 >= len(island.node_ids) >= 4:
            return None
        node_dict = island.create_node_degree_dict()
        print("\nnode dict", node_dict)
        neighbour_count = len(island.node_ids) - 1
        try:
            central_node_id = node_dict[neighbour_count][0]
            print("CN ", central_node_id)
        except Exception as e:
            return None

        edge_dict = island.create_edge_property_dikt(central_node_id)
        print("edge_dict ", edge_dict)

        longest_dict = None

        for key, value in edge_dict.items():
            if longest_dict is None:
                longest_dict = value
            else:
                if value[AmiNode.PIXLEN] > longest_dict[AmiNode.PIXLEN]:
                    longest_dict = value
        longest_edge = AmiEdge(ami_graph=island.ami_graph, start=central_node_id, end=longest_dict[AmiNode.REMOTE], branch_id=0)
        print("xLE" , longest_edge)

        print("xLD", longest_dict)

        # TODO need to check the angles
        ami_arrow = AmiArrow(island)
        ami_arrow.head_id = central_node_id
        ami_arrow.tail_id = longest_dict[AmiNode.REMOTE]

        # find short lines
        short_lines = []
        for key, value in edge_dict.items():
            if value != longest_dict:
                short_lines.append(value)
        print(f"short lines {short_lines} {len(short_lines)}")

        nshort = len(short_lines)
        arrow_point_line = None
        barbs = []
        if nshort == 3:
            # find straight on
            for line in short_lines:
                angle = island.ami_graph.get_interedge_angle(longest_edge, line)
                if abs(angle) > math.pi * AmiArrow.HEAD_WOBBLE:
                    if arrow_point_line is not None:
                        raise ValueError(f"competing lines for head line {longest_edge} to {short_lines}")
                    arrow_point_line = line
                else:
                    barbs.append(line)
            if arrow_point_line is None:
                raise ValueError(f"cannot find point line {longest_edge} to {short_lines}")
            ami_arrow.point_id = arrow_point_line.remote(central_node_id)
            if ami_arrow.point_id is None:
                print(f"cannot find point {arrow_point_line} , {central_node_id}")
        else:
            barbs = short_lines
            ami_arrow.point_id = ami_arrow.head_id

        print(f"longest {longest_edge} point {ami_arrow.point_id} barbs {barbs}")
        if len(barbs) != 2:
            raise ValueError(f" require exactly 2 barbs on arrow {barbs}")
        ami_arrow.barb_ids = [barb[AmiNode.REMOTE] for barb in barbs]

        barb_angles = []
        for barb_id in ami_arrow.barb_ids:
            angle = island.ami_graph.get_angle_between_nodes(ami_arrow.tail_id, ami_arrow.head_id, barb_id)
            barb_angles.append(angle)
        print(f"barb angles {barb_angles}")
        if abs(barb_angles[0] + barb_angles[1]) > AmiArrow.BARB_ANGLE_DIFF:
            raise ValueError(f"barb angles not symmetric {barb_angles}")
        if abs(barb_angles[0] - barb_angles[1]) / 2 > AmiArrow.MAX_BARB_TO_TAIL_ANGLE:
            raise ValueError(f"barb angles not acute {barb_angles}")



        print(f"AMIARR {ami_arrow}")
        return ami_arrow


# ----------- utils -----------
