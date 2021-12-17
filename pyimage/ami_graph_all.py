"""supports/wraps nx_graphs from NetworkX"""
# library
import numpy as np
import networkx as nx
import copy
from networkx.algorithms import tree
from skimage import io
import sknw  # must pip install sknw
import logging
from pathlib import Path, PosixPath, WindowsPath, PurePath
from skimage.measure import approximate_polygon
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# local
from ..pyimage.ami_image import AmiImage
from ..pyimage.ami_util import AmiUtil
from ..pyimage.svg import BBox
from ..pyimage.text_box import TextBox
from ..pyimage.flood_fill import FloodFill


"""
==========================================================================
==============================GRAPH=======================================
==========================================================================
"""


class AmiGraph:
    """holds AmiNodes and AmiEdges
    may also hold subgraphs
    """
    """nx_graph is a NetworkX graph which holds nodes and edges
     and which can be used to compute oher graph functionality (e.g. edges on nodes
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

    @classmethod
    def create_ami_graph_from_arbitrary_image_file(cls, file):
        assert file.exists()
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(file)
        return AmiGraph(nx_graph)

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

        AmiGraph.assert_nx_island_info(nx_island_list)
        nx_edgelist = self.get_edge_list_through_mininum_spanning_tree()
        AmiGraph.debug_edges_and_nodes(nx_edgelist, debug_count=7)
        nodes = self.nx_graph.nodes
        self.ami_node_dict = {i: (nodes[node]["o"][0], nodes[node]["o"][1]) for i, node in enumerate(nodes)}

        self.ami_island_list = []
        for nx_island in nx_island_list:
            ami_island = self.create_ami_island(nx_island)
            self.ami_island_list.append(ami_island)

        return

    @classmethod
    def assert_nx_island_info(cls, nx_island_list):
        nx_island0 = nx_island_list[0]
        assert type(nx_island0) is set
        assert len(nx_island0) > 0
        elem0 = list(nx_island0)[0]
        assert type(elem0) is int, f"island elem are {type(elem0)}"

    @classmethod
    def debug_edges_and_nodes(cls, nx_edgelist, debug_count=5):
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
        # self.nodes_yx = [nx_graph.node[ndidx][AmiNode.CENTROID] for ndidx in (nx_graph.nodes())]
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
            ami_node.set_centroid_yx(nx_graph.nodes[node_id][AmiNode.CENTROID])
            ami_node.read_nx_node(node_dict)
            self.ami_nodes.append(ami_node)

    def read_nx_edges(self, nx_graph):
        self.ami_edges = []
        for (start, end) in nx_graph.edges():
            points_yx = nx_graph[start][end][AmiNode.NODE_PTS]
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
    def create_nx_graph_from_arbitrary_image_file(cls, path, interactive=False):
        assert path.exists() and not path.is_dir(), f"{path} should be existing file"
        assert isinstance(path, PurePath)                              

        image1 = io.imread(path)
        AmiUtil.check_type_and_existence(image1, np.ndarray)
        gray_image = AmiImage.create_grayscale_from_image(image1)
        assert AmiImage.check_grayscale(gray_image) == True

        if interactive:
            io.imshow(gray_image)
            io.show()
        skeleton_array = AmiImage.invert_binarize_skeletonize(gray_image)
        assert np.max(skeleton_array) == 255, f"max skeleton should be 255 found {np.max(skeleton_array)}"

        nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)
        return nx_graph

    @classmethod
    def create_nx_graph_from_skeleton(cls, skeleton_image):
        """
        DO NOT INLINE
        :param skeleton_image:
        :return:
        """
        AmiUtil.check_type_and_existence(skeleton_image, np.ndarray)
        nx_graph = sknw.build_sknw(skeleton_image)
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
        centroid = node_data[AmiNode.CENTROID]
        centroid = (centroid[1], centroid[0])  # swap y,x as sknw seems to have this unusual order
        return centroid

# -------- AmiIsland routines

    def create_ami_island(self, node_ids):

        """
        create from a list of node_ids (maybe from sknw)
        :param node_ids: set of node ids
        :return: AmiIsland object
        """
        assert type(node_ids) is set, "componente mus be of type set"
        assert len(node_ids) > 0, "components cannot be empty"

        ami_island = AmiIsland()
        ami_island.node_ids = node_ids
        ami_island.ami_graph = self
        ami_island.create_edges()
        return ami_island

# -------- AmiGraph/AmiNode routines
    def get_or_create_ami_node(self, node_index):
        """NYI """
        nodex = AmiNode(nx_graph=self.nx_graph, node_id=(list(self.nx_graph.nodes)[node_index]))

# -------- segmentation and plotting
    @classmethod
    def plot_all_lines(cls, nx_graph, lines, tolerance, nodes=None):

        assert type(lines) is list, f"lines should be list {lines}"
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        for line in lines:
            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
            for i, j in line:
                assert type(i) is int, f"i should be int {type(i)}"
                assert type(j) is int, f"j should be int {type(j)}"
                AmiGraph.douglas_peucker_plot_line(nx_graph, i, j, tolerance, ax1, ax2, nodes=nodes)
            plt.show()

    @classmethod
    def douglas_peucker_plot_line(cls, nx_graph, i, j, tolerance, ax1, ax2, nodes=None):
        points = nx_graph[i][j][AmiEdge.PTS]

        # original wiggly line
        # x and y are reversed in sknw
        ax1.plot(points[:, 1], -points[:, 0])  # negative since down the page
        points2 = approximate_polygon(points, tolerance=tolerance)

        # the line is not directed so find which end fits which node is best
        distij = cls.move_line_ends_to_closest_node(nx_graph, (i, j), points2, move=False)
        distji = cls.move_line_ends_to_closest_node(nx_graph, (j, i), points2, move=False)
        ij = (i, j) if distij < distji else (j, i)
        cls.move_line_ends_to_closest_node(nx_graph, ij, points2, move=True)
        ax2.plot(points2[:, 1], -points2[:, 0])

    @classmethod
    def move_line_ends_to_closest_node(cls, nx_graph, ij, points, move=False):
        pts = [points[0], points[-1]]
        node_pts = [nx_graph.nodes[ij[0]]["o"], nx_graph.nodes[ij[1]]["o"]]
        delta_dist = None
        if move:
            # print(f"line end {points[0]} moved to {node_pts[0]}")
            points[0] = node_pts[0]
            # print(f"line end {points[-1]} moved to {node_pts[1]}")
            points[-1] = node_pts[1]
        else:
            delta_dist = math.dist(pts[0], node_pts[0]) + math.dist(pts[1], node_pts[1])

        return delta_dist

    @classmethod
    def add_bbox_rect(cls, axis, bbox, linewidth=1, edgecolor="red", facecolor="none"):
        """
        adds rectangle to axis subplot
        :param axis: axis from matplotlib subplots
        :param bbox: BBox from pyamiimage or its ranges
        :param linewidth: linewidth of plotted rect (1)
        :param edgecolor: stroke color of line ("red")
        :param facecolor: fill of rect ("none")
        :return:
        """
        assert type(bbox) is BBox, f"bbox should be BBox, found {type(bbox)}"
        xyr = bbox.xy_ranges
        rect = patches.Rectangle((xyr[0][0], xyr[1][0]), bbox.get_width(), bbox.get_height(),
                                 linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
        axis.add_patch(rect)

    @classmethod
    def plot_axis(cls, img_array, axis, islands, title=None):
        if title:
            axis.set_title(title)
        for island in islands:
            bbox = island.get_or_create_bbox()
            AmiGraph.add_bbox_rect(axis, bbox)
        axis.imshow(img_array)

    @classmethod
    def plot_text_box_boxes(cls, img_array, ax, text_boxes1):
        for text_box in text_boxes1:
            assert type(text_box) is TextBox, f"should be TextBox found {type(text_box)}"
            assert type(text_box.bbox) is BBox, f"expected BBox found {type(text_box.bbox)}"
            AmiGraph.add_bbox_rect(ax, text_box.bbox)
        ax.imshow(img_array)

    def get_nx_edge_list_for_node(self, node_id):
        """.

        :param node_id:
        :return: list (not view) of edges as (node_id, node_id1) (node_id, node_id2) ...
        """
        nx_edges = list(self.nx_graph.edges(node_id))
        return nx_edges

    def get_angle_to_x(self, edge):

        n0 = self.nx_graph.nodes[edge[0]]
        n1 = self.nx_graph.nodes[edge[1]]
        print(f"x0 {n0} x1 {n1}")
        pass

    def get_direct_length(self, nx_edge):
        """
        
        :param nx_edge: a tuple of node_ids
        :return: 
        """
        assert len(nx_edge) == 2
        xy0 = self.get_or_create_centroid_xy(nx_edge[0])
        xy1 = self.get_or_create_centroid_xy(nx_edge[1])
        # centroid0 = AmiNode(node_id=nx_edge[0], nx_graph=nx_graph).get_or_create_centroid_xy()
        # centroid1 = AmiNode(node_id=nx_edge[1], nx_graph=nx_graph).get_or_create_centroid_xy()
        length = math.sqrt((xy0[0] - xy1[0]) ** 2 + (xy0[1] - xy1[1]) ** 2)
        return length

    def get_nx_edge_lengths_list_for_node(self, node_id):
        nx_edges = list(self.nx_graph.edges(node_id))
        lengths = [self.get_direct_length(nx_edge) for nx_edge in nx_edges]
        return lengths

    @classmethod
    def calculate_angles_to_edges(cls, nx_graph, edges):
        for edge in edges:
            angle = AmiEdge.get_angle_to_x(nx_graph, edge)

    def get_or_create_centroid_xy(self, node_id):
        """
        gets centroid from nx_graph.nodes[node_id]
        :return:
        """
        self.centroid_xy = AmiUtil.get_xy_from_sknw_centroid(
            self.nx_graph.nodes[node_id][AmiNode.CENTROID])
        return self.centroid_xy




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
    PTS = "pts"

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
    ["o"] for centrois (AmiNode.CENTROID)
    ["pts"] for multiple points (AmiNode.POINTS)
    ALL COORDINATES COMMUNICATED BY/TO USER ARE X,Y
    (SKNW uses y,x coordinates)
    """
    CENTROID = "o"
    NODE_PTS = "pts"

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
        self.edge_ids = None
        self.ami_graph = ami_graph
        self.coords_xy = None
        self.bbox = None
        self.edges = None

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
                xy = AmiUtil.get_xy_from_sknw_centroid(yx)
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
                coord[0] = int(coord[0])
                coord[1] = int(coord[1])
                self.bbox.add_coordinate(coord)
        return self.bbox

    def plot_island(self):
        """
        Plots a given component
        :return:
        """
        # start_node_index = list(component)[0]  # take first node
        # start_node = self.nodes[start_node_index]
        # start_pixel = start_node[self.NODE_PTS][0]  # may be a list of xy for a complex node always pick first.
        start_pixel = self.coords_xy[0]
        flooder = FloodFill()
        flooder.flood_fill(self.binary, start_pixel)
        if self.interactive:
            flooder.plot_used_pixels()

    def create_edges(self):
        assert self.ami_graph.nx_graph is not None
        nx_graph = self.ami_graph.nx_graph
        self.edges = []
        for node_id in self.node_ids:
            edges = nx_graph.edges(node_id)
            for e in edges:
                if e[0] < e[1]:
                    self.edges.append(e)
