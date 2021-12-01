"""a warpper for an sknw/nx node, still being developed"""

import copy
from pyimage.util import Util

class AmiNode:
    """Node holds coordinates
    ["o"] for centrois (AmiSkeleton.CENTROID)
    ["pts"] for multiple points (AmiSkeleton.POINTS)
    ALL COORDINATES COMMUNICATED BY/TO USER ARE X,Y
    (SKNW uses y,x coordinates)
    """
    CENTROID = "o"
    def __init__(self, node_id=None, ami_graph=None, nx_graph=None):
        """

        :param node_id: mandatory
        :param ami_graph: will use ami_graph.nx_graph
        :param nx_graph: else will use nx_graph
        """

        if len(str(node_id)) > 4:
            print(f"ami_graph {ami_graph}, nx_graph {nx_graph}")
            raise Exception(f"id should be simple {node_id}")

        self.ami_graph = ami_graph
        self.nx_graph = nx_graph
        if nx_graph is None and self.ami_graph is not None:
            self.nx_graph = self.ami_graph.nx_graph
        self.centroid_xy = None
        self.node_id = node_id

    def read_nx_node(self, node_dict):
        """read dict for node, contains coordinates
        typically: 'o': array([ 82., 844.]), 'pts': array([[ 82, 844]], dtype=int16)}
        dict ket
        """
        self.node_dict = copy.deepcopy(node_dict)

    def get_or_create_centroid_xy(self):
        """
        gets centroid from nx_graph.nodes[node_id]
        :return:
        """
        if self.centroid_xy is None and self.nx_graph is not None:
            assert len(str(self.node_id)) < 4, f"self.node_id {self.node_id}"
            self.centroid_xy = Util.get_xy_from_sknw_centroid(
                self.nx_graph.nodes[self.node_id][self.CENTROID])
        return self.centroid_xy

    def set_centroid_yx(self, point_yx):
        """
        set point in y,x, format
        :param point_yx:
        :return:
        """
        self.centroid_xy = [point_yx[1], point_yx[0]]  # note coords reverse in sknw
        return

    def __repr__(self):
        s = str(self.coords_xy) + "\n" + str(self.centroid_xy)
        return s

    def __str__(self):
        s = f"centroid {self.centroid_xy}"
        return s
    
