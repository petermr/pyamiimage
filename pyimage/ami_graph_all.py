"""supports/wraps nx_graphs from NetworkX"""
# library
import numpy as np
import networkx as nx
import copy
from networkx.algorithms import tree
from skimage import io
import sknw  # must pip install sknw
import logging
from pathlib import PurePath
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

logger = logging.getLogger(__name__)

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
        """create from nodes and edges"""
        if nx_graph is None:
            raise Exception(f"nx_graph cannot be None")
        self.nx_graph = nx_graph

        self.ami_edges = None
        self.ami_nodes = None
        self.ami_island_list = None
        self.nd_skeleton = nd_skeleton
        self.ami_edge_dict = None
        self.generate_nodes = generate_nodes
        self.node_dict = None
        self.centroid_xy = None

        self.read_nx_graph(nx_graph)
        assert self.nx_graph is not None, f"ami_graph.nx_graph should not be None"
        return

    @classmethod
    def create_ami_graph_from_arbitrary_image_file(cls, file, interactive=False):
        assert file.exists()
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(file, interactive=interactive)
        return AmiGraph(nx_graph)

    def get_nx_graph(self):
        """try to avoid circular imports or attribute not found"""
        return self.nx_graph

    def create_ami_node(self, node_id):
        """create from node_id, use this rather than AmiNode() as it's easy to omit graph
        :param node_id: node_id (should exist in nx_graph but not checked yet
        :return: AmiNode (None if node_id is None)
        """
        if node_id is None:
            return None
        ami_node = AmiNode(node_id, ami_graph=self)
        return ami_node

    def create_ami_edge(self, start_id, end_id, branch_id=None):
        """create edge from node_ids, prefer this to AmiEdge constructor
        note: user must check which is start and end
        currently does nt check validity od node_ids
        :param start_id: start of edge
        :param end_id: end of edge
        :return: None if start_id or end_id is None
        """
        if start_id is None or end_id is None:
            return None
        ami_edge = AmiEdge(ami_graph=self, start_id=start_id, end_id=end_id, branch_id=branch_id)

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
            if key in self.node_dict and fail_on_duplicate:
                raise AmiGraphError(f"cannot add same node twice {key}")
            if type(raw_node) is dict:
                self.node_dict[key] = copy.deepcopy(raw_node)
            else:
                self.node_dict[key] = "node"  # store just the key at present
        else:
            self.logger.warning("node cannot be None")

    def read_edges(self, edges):
        # self.ami_edges = edges
        if len(self.node_dict.keys()) == 0 and self.generate_nodes:
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

    def _ingest_graph_info(self):
        if self.nx_graph is None:
            self.logger.warning("Null graph")
            return
        nx_island_list = list(nx.connected_components(self.nx_graph))
        if nx_island_list is None or len(nx_island_list) == 0:
            self.logger.warning("No islands")
            return

        AmiGraph.assert_nx_island_info(nx_island_list)
        nx_edgelist = self.get_edge_list_ids_through_maximum_spanning_edges()
        AmiGraph.debug_edges_and_nodes(nx_edgelist, debug_count=7)
        nodes = self.nx_graph.nodes
        self.node_dict = {i: (nodes[node]["o"][0], nodes[node]["o"][1]) for i, node in enumerate(nodes)}

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
            print("points", pts_)
        edgelist_pts_ = nx_edgelist[0][2]['pts']
        for step in edgelist_pts_[:debug_count]:
            print("step", step)
            pass

    def get_edge_list_ids_through_maximum_spanning_edges(self):
        """

        :return: list of edges as ids
        """
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
        # this may be the critical data structure and the others are convenience
        self.nx_graph = nx_graph
        self.read_nx_edges()
        self.read_nx_nodes()
        ingest = False
        if ingest:
            self._ingest_graph_info()
        return

    def read_nx_nodes(self):
        """read nx_graph and create AmiNodes"""
        assert self.nx_graph is not None, "must set self.nx_graph"
        max_digits = 5
        self.ami_nodes = []
        node_ids = self.nx_graph.nodes()
        for node_id in node_ids:
            assert len(str(node_id)) < max_digits, f"node_id is {node_id}"
            ami_node = AmiNode(ami_graph=self, node_id=node_id, nx_graph=self.nx_graph)
            ami_node.set_centroid_yx(self.nx_graph.nodes[node_id][AmiNode.CENTROID])
            self.ami_nodes.append(ami_node)

    def read_nx_edges(self):
        """read nx_graph and create AmiEdges
        """
        self.ami_edges = []
        for (start_id, end_id) in self.nx_graph.edges():
            ami_edges = self.get_ami_edges(start_id, end_id)
            for ami_edge in ami_edges:
                self.ami_edges.append(ami_edge)

    def get_ami_edges(self, start_id, end_id):
        """return edges between given nodes
        takes account of multigraph multiple edges
        """
        ami_edges = []
        if self.nx_graph.is_multigraph:
            edge_ids = self.get_edge_ids_for_node_ids(start_id, end_id)
            edge_count = len(edge_ids)
            for edge_id in range(edge_count):
                ami_edge = AmiEdge(self, start_id, end_id, edge_id)
                ami_edges.append(ami_edge)
        else:
            ami_edge = AmiEdge(self, start_id, end_id)
            ami_edges.append(ami_edge)
        return ami_edges

    def get_edge_ids_for_node_ids(self, start_id, end_id):
        """get edge_ids for pair of node_ids
        NOTE: multigrapahs may have several edges
        also the raw graph (not a Digraph) usually has pairs edges [s, e] and [e,s ]
        up to the caller to make sure that each edge is correctly extracted
        :param start_id:
        :param end_id:
        :return: edges as pairs of ids"""
        edge_ids = self.nx_graph[start_id][end_id]
        return edge_ids

    def get_or_create_ami_islands(self, maxdim=None, mindim=None, minmaxdim=None, maxmindim=None, reset=True):
        """
        NOT YET TESTED
        AmiIslands are nx_graph 'components' with added functionality

        :param reset: recalculate island list and reset self.ami_island_list ; default = True
        :param maxdim: maximum maximum dimension (BOTH dimensions <= maxdim)
        :param mindim: minimum minimum dimension (BOTH dimensions >= mindim
        :param minmaxdim: maximum maximum dimension (at least ONE >= minmaxdim)
        :param maxmindim: maximum minimum dimension (at least ONE <= maxmindim
        :return: list of AmiIslands

        if no optional args returns all islands
        Dimensions:
        Assume islands of width, height:
        A[1, 6] B[1,10], C[4,3], D[20, 30] E[1, 20] F[1,2]
        maxdim = 5 selects C, F
        mindim = 3 selects C D
        maxmindim = 2 selects A B E F (i.e thin)
        minmaxdim = 5 selects A B D E (i.e. not tiny)

        They can be combined:
        mindim = 3 and maxdim = 4 selects C

        """
        if (self.ami_island_list is None or reset) and self.nx_graph is not None:
            self.ami_island_list = [self.create_ami_island(comp) for comp in
                                    nx.algorithms.components.connected_components(self.nx_graph)]
        if mindim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list if isl.get_or_create_bbox().min_dimension() >= mindim]
        if maxdim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list if isl.get_or_create_bbox().max_dimension() <= maxdim]
        if minmaxdim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list if isl.get_or_create_bbox().max_dimension() >= minmaxdim]
        if maxmindim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list if isl.get_or_create_bbox().min_dimension() <= maxmindim]
        return self.ami_island_list

    @classmethod
    def create_nx_graph_from_arbitrary_image_file(cls, path, interactive=False):
        assert path.exists() and not path.is_dir(), f"{path} should be existing file"
        assert isinstance(path, PurePath)                              

        image1 = io.imread(str(path))
        AmiUtil.check_type_and_existence(image1, np.ndarray)
        gray_image = AmiImage.create_grayscale_from_image(image1)
        assert AmiImage.check_grayscale(gray_image)

        if interactive:
            io.imshow(gray_image)
            io.show()
        skeleton_array = AmiImage.invert_binarize_skeletonize(gray_image)
        if interactive:
            io.imshow(skeleton_array)
            io.show()
        assert np.max(skeleton_array) == 255, f"max skeleton should be 255 found {np.max(skeleton_array)}"

        nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)
        return nx_graph

    @classmethod
    def create_nx_graph_from_skeleton(cls, skeleton_image):
        """
        DO NOT INLINE
        multi: allows for multiple paths between two nodes (almost certainly required)
        iso: finds isolated points. It would be valuable for dotted lines
        ring: finds rings without nodes (e.g. boxes)
        full: makes all edges connect to centre of extended node

        :param skeleton_image:
        :return:
        """
        AmiUtil.check_type_and_existence(skeleton_image, np.ndarray)

        nx_graph = sknw.build_sknw(skeleton_image, multi=True, iso=True, ring=True, full=True)
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

        ami_island = AmiIsland(ami_graph=self, node_ids=node_ids)
        return ami_island

    # -------- AmiGraph/AmiNode routines

    def get_angles_round_node(self, node_id):
        """
        gets ordered lists of angles
        may not be efficient but for small numnbers of connections can be tolerated
        :param node_id:
        :return: ref_node, central_node, {node: val}
        """
        assert node_id is not None

        node_ids = list(self.nx_graph.neighbors(node_id))
        assert node_ids is not None
        assert node_ids[0] is not None, f"node_ids {node_ids}"

        # xys = {node_id: AmiNode(ami_graph=self, node_id=node_id).centroid_xy for node_id in node_ids}
        # print(xys.keys())

        # angle_dict = {node1: self.get_angle(node_ids[0], node_id, node1) for node1 in node_ids[1:]}
        angle_dict = {}
        for node1 in node_ids:
            angle = self.get_angle_between_nodes(node_ids[0], node_id, node1)
            angle_dict[node1] = angle
        return node_ids[0], node_id, angle_dict

    def get_angle_between_nodes(self, node_id0, node_id1, node_id2):
        """
        gets angle 0-1-2
        :param node_id0:
        :param node_id1:
        :param node_id2:
        :return: radians
        """
        assert node_id0 is not None
        assert node_id1 is not None
        assert node_id2 is not None

        # TODO messy - recreating AmiNodes is very expensive
        xy0 = AmiUtil.to_float_array(AmiNode(node_id0, ami_graph=self).centroid_xy)
        xy1 = AmiUtil.to_float_array(AmiNode(node_id1, ami_graph=self).centroid_xy)
        xy2 = AmiUtil.to_float_array(AmiNode(node_id2, ami_graph=self).centroid_xy)
        return AmiUtil.get_angle(xy0, xy1, xy2)

    # -------- AmiEdge routines

    def find_longest_edge(self, node_id):
        edges = self.nx_graph.edges(node_id)

        max_edge = None
        max_length = -1.0
        for edge in edges:
            ami_edge = AmiEdge(self, edge[0], edge[1], branch_id=0)  # actual edge_id doesn't matter for distance
            length = ami_edge.get_cartesian_length()
            assert length is not None
            if length > max_length:
                max_edge = edge
                max_length = length
        return max_edge, max_length

    # -------- segmentation and plotting

    def pre_plot_edges(self, plot_target):
        """
        recognizes different edge structures for simple and multigraph and prepares to plot
        :param plot_target:
        :return:
        """
        multi = type(self.nx_graph) is nx.classes.MultiGraph
        for (s, e) in self.nx_graph.edges():
            if multi:
                nedges = len(list(self.nx_graph[s][e]))
                for edge_id in range(nedges):
                    pts = self.nx_graph[s][e][edge_id]['pts']
                    ami_edge = AmiEdge(self, s, e, branch_id=edge_id)
                    ami_edge.plot_edge(pts, plot_target, edge_id=edge_id)
            else:
                pts = self.nx_graph[s][e]['pts']
                ami_edge = AmiEdge(self, s, e)
                ami_edge.plot_edge(pts, plot_target)

    def pre_plot_nodes(self, node_color="red", plot_ids=False):
        """prepares to plot matplotlib nodes for
        :param node_color:
        :param plot_ids:
        """
        nodes = self.nx_graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')
        if plot_ids:
            # probably a more pythomic approach
            for i in nodes:
                plt.text(ps[i, 1], ps[i, 0], str(i))

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
        assert self.nx_graph.is_multigraph(), "should be multigraph"
        nx_edges = list(self.nx_graph.edges(node_id, keys=True))
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
        return AmiUtil.get_dist(xy0, xy1)

    def get_nx_edge_lengths_by_edge_list_for_node(self, node_id):
        length_by_nx_edge = {}
        nx_edges = list(self.nx_graph.edges(node_id))
        # lengths = [self.get_direct_length(nx_edge)
        for nx_edge in nx_edges:
            length = self.get_direct_length(nx_edge)
            length_by_nx_edge[nx_edge] = length
        return length_by_nx_edge

    def get_interedge_tuple_angle(self, nx_edge0, nx_edge1):
        """
        angle between 2 nx_edges meeting at a common point
        first component of each edge is the common node
        :param nx_edge0:
        :param nx_edge1:
        :return:
        """
        assert (type(nx_edge0) is tuple), f"nx_edge0 is {type(nx_edge0)}"
        assert (type(nx_edge1) is tuple), f"nx_edge1 is {type(nx_edge1)}"
        assert len(nx_edge0) == 3, f"edge should be 3 integers {nx_edge0}"
        assert len(nx_edge1) == 3, f"edge should be 3 integers {nx_edge1}"
        assert nx_edge0[0] == nx_edge1[0], f"edges should have a common node {nx_edge0} {nx_edge1}"
        xy0 = AmiUtil.float_list(self.get_or_create_centroid_xy(nx_edge0[1]))
        xyc = AmiUtil.float_list(self.get_or_create_centroid_xy(nx_edge0[0]))
        xy1 = AmiUtil.float_list(self.get_or_create_centroid_xy(nx_edge1[1]))

        angle = AmiUtil.get_angle(xy0, xyc, xy1)
        return angle

    def get_or_create_centroid_xy(self, node_id):
        """
        gets centroid from nx_graph.nodes[node_id]
        :return:
        """
        self.centroid_xy = AmiUtil.get_xy_from_sknw_centroid(
            self.nx_graph.nodes[node_id][AmiNode.CENTROID])
        return self.centroid_xy

    @classmethod
    def get_node_ids_from_graph_with_degree(cls, nx_graph, degree):
        """
        iterates over graph.nodes to find those with given degree
        graph may be a subgraph
        :param nx_graph: might be a subgraph
        :param degree: connnectivity
        :return: list of node_ids (may be empty)
        """
        assert type(nx_graph) is nx.Graph or type(nx_graph) is nx.MultiGraph, f"not a graph {type(nx_graph)} "
        return [node_id for node_id in nx_graph.nodes if nx_graph.degree(node_id) == degree]

    def get_node_ids_with_degree(self, degree):
        """
        iterates over graph.nodes to find those with given degree
        graph may be a subgraph
        :param degree:
        :return: list of node_ids (may be empty)
        """
        return AmiGraph.get_node_ids_from_graph_with_degree(self.nx_graph, degree)

    def create_ami_nodes_from_ids(self, node_ids):
        """
        creates a list of AmiNodes from a list of node_ids

        :param node_ids:
        :return: list of AmiNodes
        """
        return [AmiNode(ami_graph=self, node_id=node_id) for node_id in node_ids]

    def extract_aligned_node_lists(self, node_ids, pixel_error):
        """Lists of horizontal, vertical, and other node-node edges

        checks x- and y- coords of connected nodes to propose connections parallel to axes
        :param node_ids: list of node_ids to analyse
        :param pixel_error: max deviation from orthogonality in pixels
        :return: list of horizontal, vectrical and other node-node connections
        """
        vertical_lines = []
        horizontal_lines = []
        non_hv_lines = []
        for node_id in node_ids:
            ami_node = AmiNode(node_id=node_id, ami_graph=self)
            node_xy = ami_node.centroid_xy
            for neighbour_id in ami_node.get_neighbour_ids():
                neighbour_xy = AmiNode(node_id=neighbour_id, ami_graph=self).centroid_xy
                # only add each line once
                if node_id < neighbour_id:
                    # line = SVGLine(node_xy, neighbour_xy)
                    line = [node_xy, neighbour_xy]
                    if abs(neighbour_xy[0] - node_xy[0]) <= pixel_error:
                        vertical_lines.append(line)
                    elif abs(neighbour_xy[1] - node_xy[1]) <= pixel_error:
                        horizontal_lines.append(line)
                    else:
                        non_hv_lines.append(line)
        return horizontal_lines, vertical_lines, non_hv_lines,

    def get_unique_edges_and_multibranches(self, node_ids):
        """gets all unique edges including multibranches
        only includes edges where start_id < end_id
        :param node_ids: node_ids
        :return: unique_edges, multibranches
        """
        unique_edges = []
        multibranches = []
        for central_node_id in node_ids:
            ami_node = self.create_ami_node(central_node_id)
            for neighbour_id in ami_node.get_neighbour_ids():
                ami_edges = self.get_ami_edges(central_node_id, neighbour_id)
                # uniquify by  adding multibranch edges if start_id < end_id
                if len(ami_edges) > 1:
                    for ami_edge in ami_edges:
                        if ami_edge.start_id < ami_edge.end_id:
                            multibranches.append(ami_edge)
                for ami_edge in ami_edges:
                    if central_node_id < neighbour_id:
                        unique_edges.append(ami_edge)
        return multibranches, unique_edges

# ================================================

    @classmethod
    def assert_nodes_of_degree(cls, ami_graph, degree, node_id_list):
        """
        assert that nodes of given degree equal expected
        :param ami_graph:
        :param node_id_list: expected node_ids
        :param degree: of nodes
        :return:
        """
        assert ami_graph.get_node_ids_of_degree(degree) == node_id_list, \
            f"expected nodes of degree {degree} +> {node_id_list}, found {ami_graph.get_node_ids_of_degree(degree)}"



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

    def __init__(self, ami_graph, start_id, end_id, branch_id=None):
        """Do not use this, but ami_graph.create_edge instead
        """
        self.ami_graph = ami_graph
        self.start_id = start_id
        self.end_id = end_id
        self.points_xy = None
        self.bbox = None
        if ami_graph.nx_graph.is_multigraph():
            if branch_id is None:
                raise ValueError("branch_id for {self.start} {self.end} should be set")
            self.branch_id = branch_id
        self.nx_graph = ami_graph.nx_graph
        self.get_points()

    def get_tuple(self):
        """
        get edge as (node_id, node_id2, branch_id)

        :return: (self.start, self.end, self.branch_id)
        """
        return (self.start_id, self.end_id, self.branch_id)

    def get_id(self):
        """
        unique id based on start, end and branch

        :return: <start>_<end>_<branch>
        """
        return f"{self.start_id}_{self.end_id}_{self.branch_id}"

    def get_points(self):
        assert self.start_id is not None
        assert self.end_id is not None
        edges = self.nx_graph[self.start_id][self.end_id]
        assert edges is not None
        if self.nx_graph.is_multigraph():
            assert self.branch_id is not None
            try:
                edge = edges[self.branch_id]
                points = edge[self.PTS]
            except KeyError as e:
                logger.error(f"{__name__} key error {self.branch_id} {e}")
                raise e
        else:
            points = edges[self.PTS]
        if points is not None:
            # print(f"points: {len(points)} {points[:3]} ... {points[-3:]}")
            self.read_nx_edge_points_yx(points)
        return points

    # class AmiEdge:

    def get_cartesian_length(self):
        assert self.ami_graph is not None
        xy0 = AmiNode(ami_graph=self.ami_graph, node_id=self.start_id).centroid_xy
        xy1 = AmiNode(ami_graph=self.ami_graph, node_id=self.end_id).centroid_xy
        dist = AmiUtil.get_dist(xy0, xy1)
        assert dist is not None
        return dist

    def read_nx_edge_points_yx(self, points_array_yx):
        """
        convert from nx_points (held as yarray, xarray) to array(x, y)
        :param points_array_yx:
        :return:
        """
        assert type(points_array_yx) is np.ndarray, \
            f"points must be numpy array from sknw, found {type(points_array_yx)}"
        # points are in separate columns (y, x)
        # TODO reshape this better
        assert points_array_yx is not None and points_array_yx.ndim == 2 and points_array_yx.shape[1] == 2
        self.points_xy = []
        for point in points_array_yx:
            self.points_xy.append([point[1], point[0]])

    # class AmiEdge:

    def __repr__(self):
        s = ""
        if self.points_xy is not None:
            ll = int(self.pixel_length() / 2)
            s = f"ami_edge {self.start_id}...{self.end_id} ({self.pixel_length()}) " \
                f"{self.points_xy[:1]}___{self.points_xy[ll - 0:ll + 1]}___{self.points_xy[-1:]}"
        return s

    def pixel_length(self):
        """
        returns len(self.points_xy)
        :return:
        """
        return len(self.points_xy)

    def remote_node_id(self, node_id):
        """
        gets node_id at other end of edge
        :param node_id:
        :return: id of node at other end (None if node_id is None or not in edge)
        """
        if node_id is None:
            return None
        elif node_id == self.start_id:
            return self.end_id
        elif node_id == self.end_id:
            return self.start_id
        else:
            return None

    # class AmiEdge:

    def get_or_create_bbox(self):
        if self.bbox is None and self.points_xy is not None:
            self.bbox = BBox()
            for point in self.points_xy:
                self.bbox.add_coordinate(point)

        logger.debug(f"bbox {self.bbox}")
        return self.bbox

    def plot_edge(self, pts, plot_region, edge_id=None, boxcolor=None):
        """
        include bbox
        :param edge_id:
        :param pts: points in nx_graph format
        :param plot_region:
        :param boxcolor: if not None plot edge box in this colour
        :return:
        """
        colors = ["green", "blue", "magenta", "cyan"]
        # print(f"edge_id {edge_id} pts: {pts[:5]}")

        if boxcolor is not None:
            bbox = self.get_or_create_bbox()
            AmiGraph.add_bbox_rect(plot_region, bbox, linewidth=1, edgecolor=boxcolor, facecolor="none")
        edgecolor = colors[0] if edge_id is None else colors[edge_id % len(colors)]
        plt.plot(pts[:, 1], pts[:, 0], edgecolor)

    # class AmiEdge:

    @classmethod
    def plot_all_lines(cls, nx_graph, lines, tolerance):
        """
        plots edges as lines
        compare with above, maybe merge
        :param nx_graph:
        :param lines: to plot - where from?
        :param tolerance:
        :return:
        """
        assert type(lines) is list, f"lines should be list {lines}"
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        for line in lines:
            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
            for i, j in line:
                assert type(i) is int, f"i should be int {type(i)}"
                assert type(j) is int, f"j should be int {type(j)}"
                AmiEdge.douglas_peucker_plot_line(nx_graph, i, j, tolerance, ax1, ax2)
            plt.show()

    @classmethod
    def douglas_peucker_plot_line(cls, nx_graph, i, j, tolerance, ax1, ax2):
        points = nx_graph[i][j][AmiEdge.PTS]

        # original wiggly line
        # x and y are reversed in sknw
        ax1.plot(points[:, 1], -points[:, 0])  # negative since down the page
        points2 = approximate_polygon(points, tolerance=tolerance)

        # the line is not directed so find which end fits which node is best
        distij = cls.move_line_ends_to_closest_node(nx_graph, (i, j), points2, move=False)
        distji = cls.move_line_ends_to_closest_node(nx_graph, (j, i), points2, move=False)
        ij = (i, j) if distij < distji else (j, i)
        AmiEdge.move_line_ends_to_closest_node(nx_graph, ij, points2, move=True)
        ax2.plot(points2[:, 1], -points2[:, 0])

    # class AmiEdge:

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


"""a wrapper for an sknw/nx node, still being developed"""


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
    NEXT_ANG = "next_angle"
    PIXLEN = "pixlen"
    REMOTE = "remote"

    def __init__(self, node_id, ami_graph=None, nx_graph=None):
        """

        :param node_id: mandatory
        :param ami_graph: will use ami_graph.nx_graph
        :param nx_graph: else will use nx_graph
        """

        if node_id is None:
            raise ValueError("AmiNode must have node_id")
        if len(str(node_id)) > 4:
            print(f"ami_graph {ami_graph}, nx_graph {nx_graph}")
            raise Exception(f"id should be simple {node_id}")

        self.ami_graph = ami_graph
        self.nx_graph = nx_graph

        if nx_graph is None and self.ami_graph is not None:
            self.nx_graph = self.ami_graph.nx_graph
        assert self.nx_graph is not None
        assert node_id is not None

        self.centroid_xy = None if self.nx_graph is None\
            else AmiUtil.get_xy_from_sknw_centroid(self.nx_graph.nodes[node_id][self.CENTROID])
        if self.centroid_xy is None:
            raise ValueError(f"Null centroid {node_id}")
        assert self.centroid_xy is not None

        self.coords_xy = None
        self.node_id = node_id
        self.edges = None  # is this used?
        self.node_dict = None  # is this used?
        self.edge_dict = None
        self.ami_edges = None

# class AmiNode:

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

    def get_neighbour_ids(self):
        """
        returns list of neighbours of current node
        :return: list of neighbours as node_ids
        """
        return list(self.nx_graph.neighbors(self.node_id))

# class AmiNode:

    def __repr__(self):
        s = str(self.coords_xy) + "\n" + str(self.centroid_xy)
        return s

    def __str__(self):
        s = f"{self.node_id} centroid {self.centroid_xy}"
        return s

    def get_or_create_ami_edges(self):
        """
        get list of AmiEdges
        :return: list of AmiEdges (empty list if none)
        """
        edge_list = self.ami_graph.get_nx_edge_list_for_node(self.node_id)
        self.ami_edges = [AmiEdge(self.ami_graph, edge[0], edge[1], branch_id=edge[2]) for edge in edge_list]
        return self.ami_edges

    def create_edge_property_dict(self):
        """
        Retrun a dictionary of the neighbouring edges and their properties
        :return: dictionary of each edge by id, properties include pixel length and
        angle to next edge
        """
        self.get_or_create_ami_edges()
        self.edge_dict = {}
        assert self.node_id is not None
        for i, ami_edge in enumerate(self.ami_edges):
            self._add_edge_to_dict(ami_edge, i)
        logger.debug(f"dict: {self.edge_dict}")
        return self.edge_dict

    def _add_edge_to_dict(self, ami_edge, i):
        logger.debug(f"edge {i}: {ami_edge}")
        remote_node_id = ami_edge.remote_node_id(self.node_id)
        assert remote_node_id is not None
        pixel_len = ami_edge.pixel_length()
        self.edge_dict[ami_edge.get_id()] = {self.PIXLEN: pixel_len}
        # get interedge angles
        self._add_next_edge(ami_edge, remote_node_id, i)

    def _add_next_edge(self, ami_edge, remote_node_id, i):
        """
        adds following edge and angle to it
        (noop if only one edge)
        :param ami_edge:
        :param remote_node_id:
        :param i:
        :return:
        """
        if len(self.ami_edges) > 1:
            next_edge = self.ami_edges[(i + 1) % len(self.ami_edges)]
            next_node_id = next_edge.remote_node_id(self.node_id)
            assert next_node_id is not None
            logger.debug("angle ", (remote_node_id, self.node_id, next_node_id))
            angle = self.ami_graph.get_angle_between_nodes(remote_node_id, self.node_id, next_node_id)
            self.edge_dict[ami_edge.get_id()][self.NEXT_ANG] = angle
            self.edge_dict[ami_edge.get_id()][self.REMOTE] = remote_node_id

    @classmethod
    def get_xy_for_node_id(cls, nx_graph, node_id):
        return nx_graph.nodes[node_id][AmiNode.CENTROID]


# =====


"""AmiIsland is a set of node_ids that NetwworkX has listed as a "component"""

"""
==========================================================================
==============================ISLAND======================================
==========================================================================
"""


class AmiIsland:
    """
    An isolated set of nodes and edges (nx calls them components)

    Not sure whether this should subclass AmiGraph or whether we should repeat / functions
    """
    def __init__(self, ami_graph, node_ids=None, make_edges=True):
        """creates island from nx_graph components and pupulates with nodes and edges

        :param ami_graph: to create from (mandatory)
        :param node_ids: node ids
        :param make_edges: uses nx_graph edges to create list of edge ids

        """
        self.id = None
        self.node_ids = node_ids
        self.edge_ids = None
        self.ami_graph = ami_graph

        self.island_nx_graph = self.create_island_sub_graph()
        assert self.island_nx_graph is not None
        self.coords_xy = None
        self.bbox = None
        self.edges = None
        # self.degree_dict = None
        self.node_degree_dict = None
        if make_edges:
            self.create_edges()

    def create_island_sub_graph(self, deep_copy=False):
        """
        from NetworkX https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html
        essentially a deep copy of the graph components
        its island_nx_graph can be passed to class methods in AmiGraph
        :return: self.island_nx_graph
        """
        if deep_copy:
            nx_graph = self.ami_graph.nx_graph
            largest_wcc = self.node_ids
            subgraph = nx_graph.__class__()
            subgraph.add_nodes_from((n, nx_graph.nodes[n]) for n in self.node_ids)
            if subgraph.is_multigraph():
                subgraph.add_edges_from((n, nbr, key, d)
                                        for n, nbrs in nx_graph.adj.items() if n in largest_wcc
                                        for nbr, keydict in nbrs.items() if nbr in largest_wcc
                                        for key, d in keydict.items())
            else:
                subgraph.add_edges_from((n, nbr, d)
                                        for n, nbrs in nx_graph.adj.items() if n in largest_wcc
                                        for nbr, d in nbrs.items() if nbr in largest_wcc)
            subgraph.graph.update(nx_graph.graph)
            self.island_nx_graph = subgraph
        else:
            # shallow subgraph (shares attributes with main nx_graph
            assert self.ami_graph is not None, f"Null ami_graph"
            assert self.ami_graph.nx_graph is not None, f"Null nx_graph"
            self.island_nx_graph = self.ami_graph.nx_graph.subgraph(self.node_ids)
            assert self.island_nx_graph is not None

        return self.island_nx_graph

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
            node_coords_xy = self.get_or_create_coords()
            self.bbox = BBox()
            for coord in node_coords_xy:
                coord[0] = int(coord[0])
                coord[1] = int(coord[1])
                self.bbox.add_coordinate(coord)

            for node_id in self.node_ids:
                edge_list = self.ami_graph.get_nx_edge_list_for_node(node_id)
                logger.debug(f"iterating over {edge_list}")
                for edge in edge_list:
                    ami_edge = AmiEdge(self.ami_graph, edge[0], edge[1], branch_id=edge[2])
                    bbox = ami_edge.get_or_create_bbox()
                    self.bbox = self.bbox.union(bbox)

        logger.debug(f"final {self.bbox}")
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
        """creates self.edges for an island from its node ids and the graph edge generator

        :return: list of graph edges
        """
        logger.warning(f"{__name__} use subgraph instead")
        assert self.ami_graph.nx_graph is not None
        nx_graph = self.ami_graph.nx_graph
        self.edges = []
        for node_id in self.node_ids:
            edges = nx_graph.edges(node_id)
            for e in edges:
                if e[0] < e[1]:
                    self.edges.append(e)
        return self.edges

    def get_node_ids_of_degree(self, node_count):
        """
        gets node_ids in island with given count
        delegates to  AmiGraph.get_node_ids_from_graph_with_degree
        :param node_count:
        :return:
        """
        return AmiGraph.get_node_ids_from_graph_with_degree(self.island_nx_graph, node_count)

    @classmethod
    def get_islands_with_max_dimension_greater_than(cls, max_dim, islands):
        """
        get islands where at least one of width and height >= max_dim
        :param max_dim:
        :param islands:
        :return:
        """
        return [island for island in islands if island.get_or_create_bbox().max_dimension() >= max_dim]

    @classmethod
    def get_islands_with_max_min_dimension(cls, min_dim, islands):
        """
        get islands where at least one of width and height <= min_dim
        :param min_dim:
        :param islands:
        :return:
        """
        return [island for island in islands if island.get_or_create_bbox().min_dimension() <= min_dim]

    def create_node_degree_dict(self):
        """
        create diction of form <degree>: <nodes of degree>
        up to degree == 5 (even 5 is unlikely after thinning)
        :return:
        """
        self.node_degree_dict = {}
        for degree in range(6):
            nodes = self.get_node_ids_of_degree(degree)
            self.node_degree_dict[degree] = nodes
        return self.node_degree_dict

    def create_edge_property_dikt(self, node_id):
        """
        create dictionary for eges connected to island
        :param node_id:
        :return:
        """
        ami_node = AmiNode(node_id, ami_graph=self.ami_graph)
        edge_dict = ami_node.create_edge_property_dict()
        return edge_dict
