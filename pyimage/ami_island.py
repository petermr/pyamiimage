"""AmiIsland is a set of node_ids that NetwworkX has listed as a "component"""
import logging

from pyimage.svg import BBox
from pyimage.util import Util


class AmiIsland:
    def __init__(self):
        # self.ami_skeleton = None
        self.node_ids = None
        self.ami_graph = None
        # self.nodes = []
        self.coords_xy = None
        self.bbox = None

    @classmethod
    def create_island(cls, node_ids, ami_graph, skeleton=None):
        """
        create from a list of node_ids (maybe from sknw)
        maybe should be instance method of ami_graph
        :param node_ids: set of node ids
        :param ami_graph: essential
        :param skeleton:
        :return:
        """
        assert type(node_ids) is set, "componente mus be of type set"
        assert len(node_ids) > 0 , "components cannot be empty"

        ami_island = AmiIsland()
        ami_island.node_ids = node_ids
        ami_island.ami_skeleton = skeleton
        ami_island.ami_graph = ami_graph
        print("ami_island", ami_island)
        return ami_island

    def __str__(self):
        s = "" + \
            f"node_ids: {self.node_ids}; \n" + \
            f"coords: {self.coords_xy}\n" + \
            "\n"
            # f"skeleton {self.ami_skeleton}\n" + \

        return s

    def get_raw_box(self):
        bbox = None
        return bbox

    def get_or_create_coords(self):

        coords = []
        if self.coords_xy is None:
            for node_id in self.node_ids:
                yx = self.ami_graph.nx_graph.nodes[node_id]["o"]
                xy = Util.get_xy_from_sknw_centroid(yx)
                coords.append(xy)
            # self.get_or_create_nodes()
            # self.coords_xy = []
            # for node in self.nodes:
            #     coord_xy = node.coords_xy
            #     self.coords_xy.append(coord_xy)

    # def get_or_create_nodes(self):
    #     if len(self.nodes) == 0 and self.node_ids is not None:
    #         self.nodes = [AmiNode(node_id) for node_id in self.node_ids]
    #     return self.nodes
        return coords

    def get_or_create_bbox(self):
        if self.bbox is None:
            coords_xy = self.get_or_create_coords()
            print(f"coords_xy {coords_xy}")
            self.bbox = BBox()
            for coord in coords_xy:
                self.bbox.add_coordinate(coord)
        return self.bbox



