"""supports/wraps nx_graphs from N etworkX"""
import numpy as np
import networkx as nx
import copy
from networkx.algorithms import tree
from skimage import io
import sknw  # must pip install sknw
import logging
from pathlib import PosixPath

from pyimage.ami_image import AmiImage
from pyimage.util import Util
from pyimage.ami_skeleton import AmiSkeleton
from pyimage.svg import BBox


"""
==========================================================================
==============================GRAPH=======================================
==========================================================================
"""


class AmiGraph:
    """holds AmiNodes and AmiEdges
    may also hold subgraphs
    """

    logger = logging.getLogger("ami_graph")

    def __init__(self, nx_graph, generate_nodes=False, nd_skeleton=None):
        """create fro nodes and edges"""
        if nx_graph is None:
            raise Exception(f"nx_graph cannot be None")
        self.nx_graph = nx_graph

        self.ami_edges = None
        self.ami_nodes = None
        self.ami_island_list = None
        self.nd_skeleton = nd_skeleton
        self.ami_edge_dict = None
        self.generate_nodes = generate_nodes
        self.ami_node_dict = None

        self.read_nx_graph(nx_graph)
        assert self.nx_graph is not None, f"ami_graph.nx_graph should not be None"
        return

    def get_nx_graph(self):
        """try to avoid circular imports or attribute not found"""
        return self.nx_graph

    def add_raw_node(self, raw_node, fail_on_duplicate=False):
        """add a raw node either a string or string-indexed dict
        if already a dict, deepcopy it
        if a primitive make a node_dict and start it with raw_node as id
        :raw_node: node to add, must have key
        :fail_on_duplicate: if true fail if key already exists
        """
        if raw_node is not None:
            # ami_node = AmiNode()
            key = raw_node.key if type(raw_node) is dict else str(raw_node)
            key = "n" + str(key)
            if key in self.ami_node_dict and fail_on_duplicate:
                raise AmiGraphError(f"cannot add same node twice {key}")
            if type(raw_node) is dict:
                self.ami_node_dict[key] = copy.deepcopy(raw_node)
            else:
                self.ami_node_dict[key] = "node"  # store just the key at present
        else:
            self.logger.warning("node cannot be None")

    def read_edges(self, edges):
        # self.ami_edges = edges
        if len(self.ami_node_dict.keys()) == 0 and self.generate_nodes:
            self.generate_nodes_from_edges()
        for i, edge in enumerate(edges):
            idx = "e" + str(i)
            self.add_edge(edge, idx)

    def add_edge(self, raw_edge, idx, fail_on_duplicate=True):
        if fail_on_duplicate and idx in self.ami_edge_dict.keys():
            raise ValueError("duplicate edge")

        if raw_edge is None:
            raise AmiGraphError("cannot add edge=None")
        edge1 = ("n" + str(raw_edge[0]), "n" + str(raw_edge[1]))
        self.ami_edge_dict[idx] = edge1

    def generate_nodes_from_edges(self):
        if self.ami_edges is not None:
            for edge in self.ami_edges:
                self.add_raw_node(edge[0])
                self.add_raw_node(edge[1])

    @classmethod
    def create_ami_graph(cls, nd_skeleton):
        """Uses Sknw to create a graph object within a new AmiGraph"""
        # currently only called in a test
        nx_graph = sknw.build_sknw(nd_skeleton)
        ami_graph = AmiGraph(nx_graph, nd_skeleton=nd_skeleton)
        return ami_graph

    def ingest_graph_info(self):
        if self.nx_graph is None:
            self.logger.warning("Null graph")
            return
        nx_island_list = list(nx.connected_components(self.nx_graph))
        if nx_island_list is None or len(nx_island_list) == 0:
            self.logger.warning("No islands")
            return

        self.assert_nx_island_info(nx_island_list)
        nx_edgelist = self.get_edge_list_through_mininum_spanning_tree()
        self.debug_edges_and_nodes(nx_edgelist, debug_count=7)
        nodes = self.nx_graph.nodes
        self.ami_node_dict = {i: (nodes[node]["o"][0], nodes[node]["o"][1]) for i, node in enumerate(nodes)}

        self.ami_island_list = []
        for nx_island in nx_island_list:
            ami_island = self.create_ami_island(nx_island, skeleton=self.nd_skeleton)
            self.ami_island_list.append(ami_island)

        return

    def assert_nx_island_info(self, nx_island_list):
        nx_island0 = nx_island_list[0]
        assert type(nx_island0) is set
        assert len(nx_island0) > 0
        elem0 = list(nx_island0)[0]
        assert type(elem0) is int, f"island elem are {type(elem0)}"

    def debug_edges_and_nodes(self, nx_edgelist, debug_count=5):
        start_node = 0
        end_node = 1
        pts_index = 2
        for edge in nx_edgelist[:debug_count]:
            pts_ = edge[pts_index]['pts']
        edgelist_pts_ = nx_edgelist[0][2]['pts']
        for step in edgelist_pts_[:debug_count]:
            print("step", step)
            pass

    def get_edge_list_through_mininum_spanning_tree(self):
        mst = tree.maximum_spanning_edges(self.nx_graph, algorithm="kruskal", data=True)
        # mst = tree.minimum_spanning_tree(graph, algorithm="kruskal")
        nx_edgelist = list(mst)
        return nx_edgelist

    @classmethod
    def set_bbox_pixels_to_color(cls, bbox, image, colorx=255):
        """sets all pixels in box to uniform color

        :param bbox:
        :param image:
        :param colorx:
        :return: modified image
        """
        xx = bbox[0]
        yy = bbox[1]
        image[xx[0]:xx[1], yy[0]:yy[1]] = colorx
        return image

    def __str__(self):
        s = "nodes: " + str(self.ami_nodes) + \
            "\n edges: " + str(self.ami_edges)
        return s

    def read_nx_graph(self, nx_graph):
        """
        Read and unpack NetworkX graph.
        This may change as a result of changing data models
        the nx_graph may be tyhe fundamental data structure
        :param nx_graph:
        :return:
        """
        # self.nodes_as_dicts = [nx_graph.node[ndidx] for ndidx in (nx_graph.nodes())]
        # self.nodes_yx = [nx_graph.node[ndidx][AmiSkeleton.CENTROID] for ndidx in (nx_graph.nodes())]
        self.read_nx_edges(nx_graph)
        self.read_nx_nodes(nx_graph)
        # this may be the critical data structure and the others are convenience
        self.nx_graph = nx_graph
        ingest = False
        if ingest:
            self.ingest_graph_info()
        return

    def read_nx_nodes(self, nx_graph):
        self.ami_nodes = []
        node_ids = nx_graph.nodes()
        for node_id in node_ids:
            node_dict = node_ids[node_id]
            assert len(str(node_id)) < 4, f"node_id is {node_id}"
            ami_node = AmiNode(ami_graph=self, node_id=node_id, nx_graph=nx_graph)
            ami_node.set_centroid_yx(nx_graph.nodes[node_id][AmiSkeleton.CENTROID])
            ami_node.read_nx_node(node_dict)
            self.ami_nodes.append(ami_node)

    def read_nx_edges(self, nx_graph):
        self.ami_edges = []
        for (start, end) in nx_graph.edges():
            points_yx = nx_graph[start][end][AmiSkeleton.NODE_PTS]
            ami_edge = AmiEdge()
            ami_edge.read_nx_edge_points_yx(points_yx)
            self.ami_edges.append(ami_edge)

    def get_or_create_ami_islands(self):
        """
        AmiIslands are nx_graph 'components' with added functionality
        :return: list of AmiIslands
        """
        if self.ami_island_list is None and self.nx_graph is not None:
            self.ami_island_list = [self.create_ami_island(comp) for comp in
                                    nx.algorithms.components.connected_components(self.nx_graph)]
        return self.ami_island_list

    @classmethod
    def create_nx_graph_from_arbitrary_image_file(cls, path):
        assert path.exists() and not path.is_dir(), f"{path} should be existing file"
        Util.check_type_and_existence(path, PosixPath)
        image1 = io.imread(path)
        Util.check_type_and_existence(image1, np.ndarray)
        gray_image = AmiImage.create_grayscale_from_image(image1)
        skeleton_array = AmiImage.create_white_skeleton_from_image(gray_image)
        nx_graph = AmiSkeleton.create_nx_graph_from_skeleton_wraps_sknw_NX_GRAPH(skeleton_array)
        return nx_graph

    def get_ami_islands_from_nx_graph(self):
        """
        Get the pixel-disjoint "islands" as from NetworkX
        :return: list of AmiIslands
        """

        # self.get_coords_for_nodes_and_edges_from_nx_graph(self.nx_graph)
        assert self.nx_graph is not None
        ami_islands = []
        for node_ids in nx.algorithms.components.connected_components(self.nx_graph):
            ami_island = self.create_ami_island(node_ids)
            assert ami_island is not None
            assert type(ami_island) is AmiIsland
            ami_islands.append(ami_island)
        return ami_islands

    def extract_coords_for_node(self, idx):
        """
        gets coords for a single node with given id
        :param idx: normally an int
        :return:
        """
        node_data = self.nx_graph.nodes[idx]
        centroid = node_data[AmiSkeleton.CENTROID]
        centroid = (centroid[1], centroid[0])  # swap y,x as sknw seems to have this unusual order
        return centroid

    def create_ami_island(self, node_ids):

        """
        create from a list of node_ids (maybe from sknw)
        maybe should be instance method of ami_graph
        :param node_ids: set of node ids
        :param skeleton:
        :return: AmiIsland object
        """
        assert type(node_ids) is set, "componente mus be of type set"
        assert len(node_ids) > 0, "components cannot be empty"

        ami_island = AmiIsland()
        ami_island.node_ids = node_ids
        ami_island.ami_graph = self
        return ami_island


    # def create_islands(self):
    #     """
    #     needs nx_graph to exist
    #
    #     :return: list of islands
    #     """
    #
    #     assert self.nx_graph is not None
    #     islands = self.get_ami_islands_from_nx_graph()
    #     return islands


class AmiGraphError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


if __name__ == '__main__':
    pass

"""
==========================================================================
===============================EDGE=======================================
==========================================================================
"""

"""wrapper for edge from sknw/nx, still being developed"""


class AmiEdge:
    def __init__(self):
        self.points_xy = None
        self.bbox = None

    def read_nx_edge_points_yx(self, points_array_yx):
        """
        convert from nx_points (held as yarray, xarray) to array(x, y)
        :param points_array_yx:
        :return:
        """
        # points are in separate columns (y, x)
        assert points_array_yx is not None and points_array_yx.ndim == 2 and points_array_yx.shape[1] == 2
        self.points_xy = []
        for point in points_array_yx:
            self.points_xy.append([point[1], point[0]])

    def __repr__(self):
        s = ""
        if self.points_xy is not None:
            s = f"ami edge pts: {self.points_xy[0]} .. {len(str(self.points_xy))} .. {self.points_xy[-1]}"
        return s

    def get_or_create_bbox(self):
        if self.bbox is None and self.points_xy is not None:
            self.bbox = BBox()
            for point in self.points_xy:
                self.bbox.add_coordinate(point)

        return self.bbox


"""a warpper for an sknw/nx node, still being developed"""


"""
==========================================================================
===============================NODE=======================================
==========================================================================
"""


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
        self.coords_xy = None
        self.node_id = node_id
        self.node_dict = None  # may not be needed

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


# =====

"""AmiIsland is a set of node_ids that NetwworkX has listed as a "component"""

"""
==========================================================================
==============================ISLAND======================================
==========================================================================
"""


class AmiIsland:
    def __init__(self, ami_graph=None):
        # self.ami_skeleton = None
        self.node_ids = None
        self.ami_graph = ami_graph
        self.coords_xy = None
        self.bbox = None

    def __str__(self):
        s = "" + \
            f"node_ids: {self.node_ids}; \n" + \
            f"coords: {self.coords_xy}\n" + \
            f"ami_graph: {self.ami_graph}" + \
            "\n"

        return s

    def get_raw_box(self):
        bbox = None
        return bbox

    def get_or_create_coords(self):
        coords = []
        assert self.ami_graph is not None, "must have AmiGraph"
        if self.coords_xy is None:
            for node_id in self.node_ids:
                yx = self.ami_graph.nx_graph.nodes[node_id]["o"]
                xy = Util.get_xy_from_sknw_centroid(yx)
                coords.append(xy)

        return coords

    def get_or_create_bbox(self):
        """
        create BBox object if not exists.
        May give empty box if no coordinates
        :return: BBox
        """
        if self.bbox is None:
            coords_xy = self.get_or_create_coords()
            self.bbox = BBox()
            for coord in coords_xy:
                self.bbox.add_coordinate(coord)
        return self.bbox
