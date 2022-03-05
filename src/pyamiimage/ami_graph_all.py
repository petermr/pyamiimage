"""supports/wraps nx_graphs from NetworkX"""
import copy
import logging
import math
from pathlib import PurePath

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
# library
import numpy as np
import sknw  # must pip install sknw
from networkx.algorithms import tree
from skimage import io
from skimage.measure import approximate_polygon

# local
from ..pyimage.ami_image import AmiImage
from ..pyimage.ami_plot import AmiLine
from ..pyimage.ami_util import AmiUtil
from ..pyimage.svg import BBox
from ..pyimage.text_box import TextBox

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
     and which can be used to compute other graph functionality (e.g. edges on nodes). 
     Here we wrap its functionality in Ami* classes. This is because 
     (a) it's hard for
     newcomers like me to remember all the syntax (which is almost C-like "dict-of-dicts-of-dict-of-dicts)
     or has many Views (rather than functions)
     (b) There are 4 different types of graphs with different syntaxes. This code started as 
     simple graphs nx_graph[i][j][properties] and then moved to multigraph nx_graph[i][j][branch][properties] 
     
     This is meant to help. If it doesn't, I'm sorry! and you can revert to native nx_graph. Here's a sample:
      >>>
      class MultiAdjacencyView(AdjacencyView):
     An MultiAdjacencyView is a Read-only Map of Maps of Maps of Maps.

    It is a View into a dict-of-dict-of-dict-of-dict data structure.
    The inner level of dict is read-write. But the
    outer levels are read-only.

    See Also
    ========
    AtlasView: View into dict-of-dict
    AdjacencyView: View into dict-of-dict-of-dict
    
    <<<
    That's what AmiGraph tries to hide. I hope it works.
    

     
     The components are:
     * AmiGraph (wraps nx_graph and derived quantities such as lists of nodes and edges)
     * AmiNode (wraps nx_graph.nodes[i]). nx_nodes are by default ints and should be kept as such
     * AmiEdge (wraps nx_graph.edges - a pair of ints)
     * AmiIsland (a discrete "component" of the graph)
     
     Our default is Multigraphs as images can contain paths which create loops and join two nodes in to or more
     different ways. This means that edges require three indexes, start, end and branch. This means checking that 
     this has been introduced universally.
     
     (I'm hoping that Simple graphs can also be switched on/off)
     
     Currently working with the idea that every edge has 
     (i, j) for simple graph
     (i, j, branch) for multigraph (requires keys=True)
          
     
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

        # sets up all nodes and edges
        self.read_nx_graph(nx_graph)
        assert self.nx_graph is not None, f"ami_graph.nx_graph should not be None"
        return

    @classmethod
    def create_ami_graph_from_arbitrary_image_file(cls, file, interactive=False):
        assert file.exists()
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(file, interactive=interactive)
        return AmiGraph(nx_graph)

    # def get_nx_graph(self):
    #     """try to avoid circular imports or attribute not found"""
    #     return self.nx_graph
    #
    def get_or_create_ami_node(self, node_id):
        """create from node_id or rerieve from node_dict
        use this method rather than AmiNode() constructor
        Stores AmiNodes in self.node_dict
        :param node_id: node_id (should exist in nx_graph but not checked yet
        :return: AmiNode (None if node_id is None)
        """
        if node_id is None:
            return None
        if self.node_dict is None:
            self.node_dict = dict()
        if node_id not in self.node_dict:
            ami_node = AmiNode(node_id, ami_graph=self, _private=True)
            self.node_dict[node_id] = ami_node
        else:
            ami_node = self.node_dict[node_id]
        return ami_node

    # AmiGraph

    def get_or_create_ami_edge_from_nx_edge(self, nx_edge):
        """Wrapper for get_or_create_ami_edge_from_ids() to create ami_edge
        :param nx_edge: 
        :return: ami_edge (or None)
        """
        if nx_edge is None:
            return None
        branch_id = nx_edge[2] if self.nx_graph.is_multigraph() else None
        ami_edge = self.get_or_create_ami_edge_from_ids(nx_edge[0], nx_edge[1], branch_id=branch_id)
        return ami_edge

    def get_or_create_ami_edge_from_ids(self, node_id1, node_id2, branch_id=None):
        """create or lookup AmiEdge from node_ids, or retrievs from edge_dict
        prefer this to AmiEdge constructor
        key is (sorted(node_id1, node_id2)),
        looks up AmiNode to check validity of node_ids
        If there are multiple branches between two nodes they must have different branch_ids
        (It's up to the user to manage this). Adding an edge without a branch_id will replace
        the current one)

        :param node_id1: start of edge
        :param node_id2: end of edge
        :param branch_id: id there are multiple branches; up to the user to manage these
        :return: None if start_id or end_id is None
        """
        if type(node_id1) is not int or type(node_id2) is not int:
            raise ValueError(f"node_ids must be ints , found: {node_id1}, {node_id2}")
        if self.nx_graph.is_multigraph:
            if type(branch_id) is not int:
                raise ValueError(f"branch_id for multigraph must be int, found {type(branch_id)}")
            key = (node_id1, node_id2, branch_id) if node_id1 < node_id2 \
                else (node_id2, node_id1, branch_id)
        else:
            key = (node_id1, node_id2) if node_id1 < node_id2 else (node_id2, node_id1)

        if self.ami_edge_dict is None:
            self.ami_edge_dict = dict()
        # new edge?
        if key not in self.ami_edge_dict:
            ami_edge = self.create_and_index_new_edge(key, node_id1, node_id2, branch_id)
        # existing edge/s
        else:
            ami_edge = self.ami_edge_dict[key]
        return ami_edge

    def create_and_index_new_edge(self, key, node_id1, node_id2, branch_id):
        """create new Edge and index it in edge_dict"""
        ami_edge = AmiEdge(self, node_id1, node_id2, branch_id=branch_id, _private=True)
        self.ami_edge_dict[key] = ami_edge
        return ami_edge

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

    # def read_edges(self, edges):
    #     # self.ami_edges = edges
    #     if len(self.node_dict.keys()) == 0 and self.generate_nodes:
    #         self.generate_nodes_from_edges()
    #     for i, edge in enumerate(edges):
    #         idx = "e" + str(i)
    #         self.add_edge(edge, idx)
    #
    # def add_edge(self, raw_edge, idx, fail_on_duplicate=True):
    #     if fail_on_duplicate and idx in self.ami_edge_dict.keys():
    #         raise ValueError("duplicate edge")
    #
    #     if raw_edge is None:
    #         raise AmiGraphError("cannot add edge=None")
    #     edge1 = ("n" + str(raw_edge[0]), "n" + str(raw_edge[1]))
    #     self.ami_edge_dict[idx] = edge1
    #
    # AmiGraph

    # def generate_nodes_from_edges(self):
    #     if self.ami_edges is not None:
    #         for edge in self.ami_edges:
    #             self.add_raw_node(edge[0])
    #             self.add_raw_node(edge[1])
    #
    # @classmethod
    # def create_ami_graph_from_skeleton(cls, nd_skeleton):
    #     """Uses Sknw to create a graph object within a new AmiGraph"""
    #     # currently only called in a test
    #     nx_graph = sknw.build_sknw(nd_skeleton)
    #     ami_graph = AmiGraph(nx_graph, nd_skeleton=nd_skeleton)
    #     return ami_graph

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
        self.node_dict = {i: (nodes[node][AmiNode.CENTROID][0], nodes[node][AmiNode.CENTROID][1])
                          for i, node in enumerate(nodes)}

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

    # AmiGraph

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
        the nx_graph may be the fundamental data structure
        :param nx_graph:
        :return:
        """
        # this may be the critical data structure and the others are convenience
        self.nx_graph = nx_graph
        self.get_or_create_all_ami_edges()
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
        if self.node_dict is None:
            self.node_dict = dict()
        for node_id in node_ids:
            assert len(str(node_id)) < max_digits, f"node_id is {node_id}"
            ami_node = self.get_or_create_ami_node(node_id)
            self.node_dict[node_id] = ami_node
            ami_node.set_centroid_yx(self.nx_graph.nodes[node_id][AmiNode.CENTROID])
            self.ami_nodes.append(ami_node)

    # AmiGraph

    def get_or_create_all_ami_edges(self):
        """read nx_graph and create AmiEdges
        do not confuse with self.get_ami_edges_for_start_end

        :return: self.ami_edges
        """
        if not self.ami_edges:
            self.ami_edges = []
            if self.nx_graph.is_multigraph:
                for (start_id, end_id) in self.nx_graph.edges():
                    # this gives us a list of ami_edges for node pair
                    ami_edges = self.get_ami_edge_list_for_start_end(start_id, end_id)
                    for ami_edge in ami_edges:
                        self.ami_edges.append(ami_edge)
            else:
                for (start_id, end_id) in self.nx_graph.edges():
                    ami_edge = self.get_or_create_ami_edge_from_ids(start_id, end_id)
                    self.ami_edges.append(ami_edge)

            # print(f"ami_edge_dict {self.ami_edge_dict.keys()} ")
            #       f"=> {self.ami_edge_dict} ")
        return self.ami_edges

    def get_ami_edge_list_for_start_end(self, start_id, end_id):
        """return edges between given nodes
        takes account of multigraph multiple edges
        :param start_id: start id
        :param end_id: may or may not be identical to start_id
        """
        if not self.nx_graph.is_multigraph:
            raise ValueError(f"get_ami_edge_list_for_start_end requires a multigraph ")

        ami_edges = []
        branch_ids = self.get_branch_ids_for_start_end(start_id, end_id)
        for branch_id in branch_ids:
            ami_edge = self.get_or_create_ami_edge_for_start_end_branch(start_id, end_id, branch_id)
            ami_edges.append(ami_edge)

        return ami_edges

    def get_branch_ids_for_start_end(self, start_id, end_id):
        """this iterates over the keys of the edge[i][j]
        which are AtlasViews
        These views need calling as functions

        An explanation from NetworkX:

        An AtlasView is a Read-only Mapping of Mappings.

        It is a View into a dict-of-dict data structure.
        The inner level of dict is read-write. But the
        outer level is read-only.

        This is not the best way of finding keys but I think it works

        :param start_id: start id numerica
        :param end_id: end_id (for loops/circles this may be equal to start_id
        """
        branch_ids = []
        # count each branch once only
        if AmiEdge.is_normalized_edge(start_id, end_id):
            branch_dict = self.nx_graph[start_id][end_id]
            for key in branch_dict.keys():
                branch_ids.append(key)
        return branch_ids

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
            self.ami_island_list = [isl for isl in self.ami_island_list
                                    if isl.get_or_create_bbox().min_dimension() >= mindim]
        if maxdim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list
                                    if isl.get_or_create_bbox().max_dimension() <= maxdim]
        if minmaxdim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list
                                    if isl.get_or_create_bbox().max_dimension() >= minmaxdim]
        if maxmindim is not None:
            self.ami_island_list = [isl for isl in self.ami_island_list
                                    if isl.get_or_create_bbox().min_dimension() <= maxmindim]
        return self.ami_island_list

    # AmiGraph

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

    # AmiGraph

    # def extract_coords_for_node(self, idx):
    #     """
    #     gets coords for a single node with given id
    #     :param idx: normally an int
    #     :return:
    #     """
    #     node_data = self.nx_graph.nodes[idx]
    #     centroid = node_data[AmiNode.CENTROID]
    #     centroid = (centroid[1], centroid[0])  # swap y,x as sknw seems to have this unusual order
    #     return centroid
    #

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

    # AmiGraph

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
        xy0 = AmiUtil.to_float_array(self.get_or_create_ami_node(node_id0).centroid_xy)
        xy1 = AmiUtil.to_float_array(self.get_or_create_ami_node(node_id1).centroid_xy)
        xy2 = AmiUtil.to_float_array(self.get_or_create_ami_node(node_id2).centroid_xy)
        return AmiUtil.get_angle(xy0, xy1, xy2)

    # -------- AmiEdge routines

    def find_longest_edge(self, node_id):
        edges = self.nx_graph.edges(node_id)

        max_edge = None
        max_length = -1.0
        for edge in edges:
            ami_edge = self.get_or_create_ami_edge_from_ids(edge[0], edge[1], branch_id=0)
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
        if self.nx_graph.is_multigraph:
            for (start_id, end_id) in self.nx_graph.edges():
                branch_ids = self.get_branch_ids_for_start_end(start_id, end_id)
                for branch_id in branch_ids:
                    print(f" {__name__} branch_ids / id {branch_ids} {branch_id}")
                    pts = self.get_points_on_line(start_id, end_id, branch_id=branch_id)
                    ami_edge = self.get_or_create_ami_edge_from_ids(start_id, end_id, branch_id=branch_id)
                    ami_edge.plot_edge(pts, plot_target, edge_id=branch_id)
        else:
            for (start_id, end_id) in self.nx_graph.edges():
                pts = self.get_points_on_line(start_id, end_id)
                ami_edge = self.get_or_create_ami_edge_from_ids(start_id, end_id)
                ami_edge.plot_edge(pts, plot_target)

    def get_points_on_line(self, start_id, end_id, branch_id=None):
        print(f"s/e/b {start_id} {end_id} {branch_id}")
        if self.nx_graph.is_multigraph and type(branch_id) is int:
            try:
                points = self.nx_graph[start_id][end_id][branch_id][AmiEdge.PTS]
            except KeyError as e:
                logger.error(f"cannot get points {e} {start_id} {end_id} {branch_id}")
                raise e
        else:
            points = self.nx_graph[start_id][end_id][AmiEdge.PTS]
        return points

    def pre_plot_nodes(self, plot_ids=False):
        """prepares to plot matplotlib nodes for
        :param plot_ids:
        """
        nodes = self.nx_graph.nodes()
        ps = np.array([nodes[i][AmiNode.CENTROID] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')
        if plot_ids:
            # probably a more pythomic approach
            for i in nodes:
                plt.text(ps[i, 1], ps[i, 0], str(i))

    # AmiGraph

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
        # keys gives ghe branch_ids
        nx_edges = list(self.nx_graph.edges(node_id, keys=True))

        return nx_edges

    # def get_angle_to_x(self, edge):
    #
    #     n0 = self.nx_graph.nodes[edge[0]]
    #     n1 = self.nx_graph.nodes[edge[1]]
    #     print(f"x0 {n0} x1 {n1}")
    #

    # AmiGraph

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
        # find the common node and the other ends
        # this is not pretty, but works (nodes = end1, centre, end2)
        if nx_edge0[0] == nx_edge1[0]:
            nodes = nx_edge0[1], nx_edge0[0], nx_edge1[1]
        elif nx_edge0[1] == nx_edge1[1]:
            nodes = nx_edge0[0], nx_edge0[1], nx_edge1[0]
        elif nx_edge0[1] == nx_edge1[0]:
            nodes = nx_edge0[0], nx_edge0[1], nx_edge1[1]
        elif nx_edge0[0] == nx_edge1[1]:
            nodes = nx_edge0[1], nx_edge0[0], nx_edge1[0]
        else:
            raise ValueError(f"edges should have a common node {nx_edge0} {nx_edge1}")
        xy0 = AmiUtil.float_list(self.get_or_create_centroid_xy(nodes[0]))
        xyc = AmiUtil.float_list(self.get_or_create_centroid_xy(nodes[1]))
        xy1 = AmiUtil.float_list(self.get_or_create_centroid_xy(nodes[2]))

        angle = AmiUtil.get_angle(xy0, xyc, xy1)
        return angle

    # AmiGraph

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

    # def get_node_ids_with_degree(self, degree):
    #     """
    #     iterates over graph.nodes to find those with given degree
    #     graph may be a subgraph
    #     :param degree:
    #     :return: list of node_ids (may be empty)
    #     """
    #     return AmiGraph.get_node_ids_from_graph_with_degree(self.nx_graph, degree)
    #
    def create_ami_nodes_from_ids(self, node_ids):
        """
        creates a list of AmiNodes from a list of node_ids

        :param node_ids:
        :return: list of AmiNodes
        """
        return [self.get_or_create_ami_node(node_id) for node_id in node_ids]

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
        # TODO need to iterate over edges. Get (start, end, branch)
        for node_id in node_ids:
            ami_node = self.get_or_create_ami_node(node_id)
            node_xy = ami_node.centroid_xy
            for neighbour_id in ami_node.get_neighbour_ids():
                # only add each line once
                if node_id < neighbour_id:
                    # ami_edge = self.get_or_create_ami_edge(node_id, neighbour_id)
                    neighbour_xy = self.get_or_create_ami_node(neighbour_id).centroid_xy
                    line = AmiLine([node_xy, neighbour_xy])
                    # line.set_ami_edge(ami_edge)
                    if abs(neighbour_xy[0] - node_xy[0]) <= pixel_error:
                        vertical_lines.append(line)
                    elif abs(neighbour_xy[1] - node_xy[1]) <= pixel_error:
                        horizontal_lines.append(line)
                    else:
                        non_hv_lines.append(line)
        return horizontal_lines, vertical_lines, non_hv_lines

    # AmiGraph

    def get_unique_ami_edges_and_multibranches(self, node_ids):
        """gets all unique edges including multibranches
        only includes edges where start_id < end_id
        :param node_ids: node_ids
        :return: unique_edges, multibranches
        """
        # TODO have to sort out multiple branches to neighbours
        unique_ami_edges = []
        multibranches = []
        for central_node_id in node_ids:
            ami_node = self.get_or_create_ami_node(central_node_id)
            for neighbour_id in ami_node.get_neighbour_ids():
                ami_edges = self.get_ami_edge_list_for_start_end(central_node_id, neighbour_id)
                # uniquify by  adding edges if start_id < end_id
                if len(ami_edges) > 1:
                    for ami_edge in ami_edges:
                        if ami_edge.has_start_lt_end():
                            multibranches.append(ami_edge)
                for ami_edge in ami_edges:
                    if ami_edge.has_start_lt_end():
                        unique_ami_edges.append(ami_edge)
        return unique_ami_edges, multibranches

    def get_or_create_ami_edge_for_start_end_branch(self, start_id, end_id, branch_id):
        """gets edge from ids
        :param start_id:
        :param end_id:
        :param branch_id:
        """
        if self.nx_graph.is_multigraph:
            edge_id = (start_id, end_id, branch_id)
            self.get_or_create_ami_edge_dict()
            if edge_id in self.ami_edge_dict:
                ami_edge = self.ami_edge_dict[edge_id]
            else:
                ami_edge = self.get_or_create_ami_edge_from_ids(start_id, end_id, branch_id=branch_id)
                self.ami_edge_dict[edge_id] = ami_edge
        else:
            # assume simple undirected graph - not fully tested
            edge_id = (start_id, end_id)
            ami_edge = self.get_or_create_ami_edge_from_ids(start_id, end_id)
            self.ami_edge_dict[edge_id] = ami_edge
        return ami_edge

    def get_or_create_ami_edge_dict(self):
        """ensures ami_graph.edge_dict exists
        :return: self.ami_edge_dict
        """
        if self.ami_edge_dict is None:
            self.ami_edge_dict = dict()
        return self.ami_edge_dict

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

    def __init__(self, ami_graph, start_id, end_id, branch_id=None, _private=False):
        """ Only for initial development
        Do not use this, but various ami_graph.create_edge instead
        """
        if not _private:
            raise ValueError(f"Do not call AmiEdge() directly, use AmiGraph.get_or_create_edge*()")
        self.ami_graph = ami_graph
        self.start_id = start_id
        self.end_id = end_id
        self.points_xy = None
        self.bbox = None
        if ami_graph.nx_graph.is_multigraph():
            if branch_id is None:
                branch_id = 0
                # raise ValueError("branch_id for {self.start} {self.end} should be set")
            self.branch_id = branch_id
        self.nx_graph = ami_graph.nx_graph
        self.line_points = None
        self.tolerance = 1
        self.segments = None
        self.get_points()

    def __eq__(self, other):
        """equality is based on node_ids and branch_id alone
        later we want to add this object to the nx_graph which will guarantee uniqueness
        :param other: another edge
        """
        if not isinstance(other, type(self)):
            return False
        if self.branch_id != other.branch_id:
            return False
        if self.start_id == other.start_id and self.end_id == other.end_id:
            return True
        # if self.start_id == other.end_id and self.end_id == other.start_id:
        #     return True
        return False

    # class AmiEdge:

    def __hash__(self):
        """uses node_id"""
        return hash(self.start_id) + hash(self.end_id)

    def get_tuple(self):
        """
        get edge as (node_id, node_id2, branch_id)

        :return: (self.start, self.end, self.branch_id)
        """
        return self.start_id, self.end_id, self.branch_id

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
            self.read_nx_edge_points_yx_into_self_points_xy(points)
        return points

    # class AmiEdge:

    def get_cartesian_length(self):
        assert self.ami_graph is not None
        start_node = self.ami_graph.get_or_create_ami_node(self.start_id)
        xy0 = start_node.centroid_xy
        end_node = self.ami_graph.get_or_create_ami_node(self.end_id)
        xy1 = end_node.centroid_xy
        dist = AmiUtil.get_dist(xy0, xy1)
        assert dist is not None
        return dist

    def read_nx_edge_points_yx_into_self_points_xy(self, points_array_yx):
        """
        convert from nx_points (held as yarray, xarray) to array(x, y)
        :param points_array_yx:
        :return:
        """
        assert type(points_array_yx) is np.ndarray, \
            f"points must be numpy array from sknw, found {type(points_array_yx)}"
        # points are in separate columns (y, x)
        # TODO reshape this better - use np.flip
        assert points_array_yx is not None and points_array_yx.ndim == 2 and points_array_yx.shape[1] == 2
        self.points_xy = []
        for point in points_array_yx:
            self.points_xy.append([int(point[1]), int(point[0])])

    # class AmiEdge:

    def __repr__(self):
        s = ""
        if self.points_xy is not None:
            ll = int(self.pixel_length() / 2)
            s = f"ami_edge {self.start_id}...{self.end_id} ({self.pixel_length()}) " \
                f"{self.points_xy[:1]}___{self.points_xy[ll - 0:ll + 1]}___{self.points_xy[-1:]}"
        return s

    def has_start_lt_end(self):
        """True if start_id is numerically less than end_id
        Used to prune duplicate edges (s, e) and (e,s)
        """
        return self.start_id < self.end_id

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

    def create_line_segments(self, tolerance=1):
        """create AmiLine segments from sknw points
        :param tolerance: totlerance in pixels
        :return: array of lines
        """

        points = self.nx_graph[self.start_id][self.end_id][self.branch_id][AmiEdge.PTS]
        points2 = approximate_polygon(points, tolerance=tolerance)
        return points2

    def find_single_line(self, tol=1) -> AmiLine:
        """segments the edge into straight lines (AmiLine)

        If segmentation gives a single line (of any orientation) returns it
        else None
        :param tol: max deviation of points from segments , def = 1
        :return: AmiLine or None.
        """
        segments = self.get_segments(tol)
        return segments[0] if len(segments) == 1 else None

    @classmethod
    def get_single_lines(cls, edges) -> list:
        """extracts single line from any edges which have one
        :return: list of AmiLines from edges which have exactly one"""
        ami_lines = []
        for ami_edge in edges:
            ami_line = ami_edge.find_single_line()
            if ami_line is not None:
                ami_lines.append(ami_line)
        return ami_lines

    def get_axial_lines(self, tolerance=1) -> list:
        """segments the edge into straight lines parallel to axes (AmiLine)

        If All segments are aligned with axes, returns that list else None
        :param tolerance: max deviation of points from segments
        :return: list of AmiLines or None.
        """
        segments = self.get_segments(tolerance=tolerance)  # maybe cache this
        if len(segments) > 1:
            logger.debug(f"segments {len(segments)} ... {self}")
            corners = self._get_axial_corners(segments, tolerance=tolerance)
            if len(corners) == len(segments) - 1:
                return segments

        return None

    @classmethod
    def get_axial_polylines(cls, edges, tolerance=1) -> list:
        """extracts axial polylines from any edges which consist of 2 or more axial lines
        :return: list of polylines (lists of AmiLines) from edges which have 2 or more"""
        axial_polylines = []
        for ami_edge in edges:
            ami_lines = ami_edge.get_axial_lines(tolerance=tolerance)
            if ami_lines is not None and len(ami_lines) > 1:
                axial_polylines.append(ami_lines)
        return axial_polylines

    # class AmiEdge:

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

    def get_segments(self, tolerance=1):
        """extracts line segments by skimage.measure.approximate_polygon() (Douglas-Peucker)

        :param tolerance: max deviation of curve from line segments
        :return: list of line segments as AmiLines
        """
        points_array = np.array(self.get_coords())

        self._get_or_create_segment_points(points_array, tolerance)
        self.segments = []
        for i, _ in enumerate(self.line_points[:-1]):
            pt0 = self.line_points[i]
            pt1 = self.line_points[i + 1]
            ami_line = AmiLine([pt0, pt1])
            self.segments.append(ami_line)
        return self.segments

    def _get_or_create_segment_points(self, points_array, tolerance):
        """calculate segments with approximate_polygon()
        recalculate if tolerance changes"""
        if self.line_points is None or tolerance != self.tolerance:
            self.line_points = approximate_polygon(points_array, tolerance)
            self.tolerance = tolerance
        return self.line_points

    def get_single_segment(self, segments=None, tolerance=1):
        """get edge as a single segment
        :return: a single segment if D-P finds it within self.tolerance, else None
        """
        segments = self.get_segments(tolerance=tolerance) if segments is None else segments
        return segments[0] if len(segments) == 1 else None

    def is_horizontal(self, tolerance=1):
        segment = self.get_single_segment(tolerance=tolerance)
        return segment is not None and segment.is_horizontal(tolerance=tolerance)

    def is_vertical(self, tolerance=1):
        segment = self.get_single_segment(tolerance=tolerance)
        return segment is not None and segment.is_vertical(tolerance=tolerance)

    def _get_axial_corners(self, segments, tolerance):
        """
        finds Hor-Vert and Vert-Hor corners in segments
        :param tolerance:
        :return: list of corners (last_segment, next_segment)
        """
        last_segment = None
        corners = []
        for segment in segments:
            if last_segment:
                lvert = last_segment.is_vertical(tolerance=tolerance)
                thoriz = segment.is_horizontal(tolerance=tolerance)
                lhoriz = last_segment.is_horizontal(tolerance=tolerance)
                tvert = segment.is_vertical(tolerance=tolerance)
                if (lvert and thoriz) or (tvert and lhoriz):
                    corners.append((last_segment, segment))
            last_segment = segment
        return corners

    @classmethod
    def get_vertical_edges(cls, ami_edges, tolerance=1):
        return list(filter(lambda ami_edge: ami_edge.is_vertical(tolerance=tolerance), ami_edges))

    @classmethod
    def get_horizontal_edges(cls, ami_edges, tolerance=1):
        return list(filter(lambda ami_edge: ami_edge.is_horizontal(tolerance=tolerance), ami_edges))

    # =========================================

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
        """this may be replaced by create_ami_lines()"""
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
        node_pts = [nx_graph.nodes[ij[0]][AmiNode.CENTROID], nx_graph.nodes[ij[1]][AmiNode.CENTROID]]
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
    def create_normalized_edge_id_tuple(cls, node_id1, node_id2, branch_id):
        """ensures that node_id1 <= node_id2
        This counts each edge only once"""
        if branch_id is None:
            return (node_id1, node_id2) if node_id1 <= node_id2 else (node_id2, node_id1)
        return (node_id1, node_id2, branch_id) if node_id1 <= node_id2 else (node_id2, node_id1, branch_id)

    @classmethod
    def is_normalized_edge(cls, start_id, end_id):
        """requires not None and start_id <= end_id
        :param start_id:
        :param end_id:
        """
        return start_id is not None and end_id is not None and start_id <= end_id

    def get_coords(self):
        return self.points_xy

    # def douglas_peucker2(self):
    #     """https://towardsdatascience.com/simplify-polylines-with-the-douglas-peucker-algorithm-ac8ed487a4a1
    #     Experimental - may not keep
    #     """
    #     import matplotlib.animation as animation
    #     import numpy as np
    #
    #     def rdp(points, epsilon):
    #         # get the start and end points
    #         start = np.tile(np.expand_dims(points[0], axis=0), (points.shape[0], 1))
    #         end = np.tile(np.expand_dims(points[-1], axis=0), (points.shape[0], 1))
    #
    #         # find distance from other_points to line formed by start and end
    #         dist_point_to_line = np.abs(np.cross(end - start, points - start, axis=-1)) / np.linalg.norm(end - start,
    #                                                                                                      axis=-1)
    #         # get the index of the points with the largest distance
    #         max_idx = np.argmax(dist_point_to_line)
    #         max_value = dist_point_to_line[max_idx]
    #
    #         result = []
    #         if max_value > epsilon:
    #             partial_results_left = rdp(points[:max_idx + 1], epsilon)
    #             result += [list(i) for i in partial_results_left if list(i) not in result]
    #             partial_results_right = rdp(points[max_idx:], epsilon)
    #             result += [list(i) for i in partial_results_right if list(i) not in result]
    #         else:
    #             result += [points[0], points[-1]]
    #
    #         return result
    #
    #     if __name__ == "__main__":
    #         min_x = 0
    #         max_x = 5
    #
    #         xs = np.linspace(min_x, max_x, num=200)
    #         ys = np.exp(-xs) * np.cos(2 * np.pi * xs)
    #         sample_points = np.concatenate([
    #             np.expand_dims(xs, axis=-1),
    #             np.expand_dims(ys, axis=-1)
    #         ], axis=-1)
    #
    #         # First set up the figure, the axis, and the plot element we want to animate
    #         fig = plt.figure()
    #         ax = plt.axes(xlim=(min_x, max_x), ylim=(-1, 1))
    #         plt.xlabel("x")
    #         plt.ylabel("y")
    #         text_values = ax.text(
    #             0.70,
    #             0.15,
    #             "",
    #             transform=ax.transAxes,
    #             fontsize=12,
    #             verticalalignment='top',
    #             bbox=dict(boxstyle='round',
    #                       facecolor='wheat',
    #                       alpha=0.2)
    #         )
    #         original_line, = ax.plot(xs, ys, lw=2, label=r"$y = e^{-x}cos(2 \pi x)$")
    #         simplified_line, = ax.plot([], [], lw=2, label="simplified", marker='o', color='r')
    #
    #         # initialization function: plot the background of each frame
    #         def init():
    #             simplified_line.set_data(xs, ys)
    #             return original_line, simplified_line, text_values
    #
    #         # animation function.  This is called sequentially
    #         def animate(i):
    #             epsilon = 0 + (i * 0.1)
    #             simplified = np.array(rdp(sample_points, epsilon))
    #             print(f"i: {i}, episilon: {'%.1f' % epsilon}, n: {simplified.shape[0]}")
    #             simplified_line.set_data(simplified[:, 0], simplified[:, 1])
    #             text_values.set_text(fr"$\epsilon$: {'%.1f' % epsilon}, $n$: {simplified.shape[0]}")
    #             return original_line, simplified_line, text_values
    #
    #         # call the animator.  blit=True means only re-draw the parts that have changed.
    #         anim = animation.FuncAnimation(
    #             fig,
    #             animate,
    #             init_func=init,
    #             frames=21,
    #             interval=1000,
    #             repeat=True
    #         )
    #         plt.legend()
    #         plt.show()


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

    def __init__(self, node_id, ami_graph=None, nx_graph=None, _private=False):
        """

        :param node_id: mandatory
        :param ami_graph: will use ami_graph.nx_graph
        :param nx_graph: else will use nx_graph
        """
        if not _private:
            raise ValueError("Do not call AmiNode directly; ise ami_graph.get_or_create_node*() factories")
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

        self.centroid_xy = None if self.nx_graph is None \
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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.node_id != other.node_id:
            return False
        return True

    # class AmiNode:

    def __hash__(self):
        """uses node_id"""
        return hash(self.node_id)

    # def read_nx_node(self, node_dict):
    #     """read dict for node, contains coordinates
    #     typically: 'o': array([ 82., 844.]), 'pts': array([[ 82, 844]], dtype=int16)}
    #     dict ket
    #     """
    #     self.node_dict = copy.deepcopy(node_dict)

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
        self.ami_edges = [self.ami_graph.get_or_create_ami_edge_from_ids(edge[0], edge[1], branch_id=edge[2])
                          for edge in edge_list]
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
        self.nx_edges = None
        # self.degree_dict = None
        self.node_degree_dict = None
        self.ami_edges = None
        if make_edges:
            self.create_nx_edges()

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
            "\n"
        return s

    def get_or_create_coords(self):
        coords = []
        assert self.ami_graph is not None, "must have AmiGraph"
        if self.coords_xy is None:
            for node_id in self.node_ids:
                yx = self.ami_graph.nx_graph.nodes[node_id][AmiNode.CENTROID]
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
                    ami_edge = self.ami_graph.get_or_create_ami_edge_from_ids(edge[0], edge[1], branch_id=edge[2])
                    bbox = ami_edge.get_or_create_bbox()
                    self.bbox = self.bbox.union(bbox)

        logger.debug(f"final {self.bbox}")
        return self.bbox

    def create_nx_edges(self):
        """creates self.nx_edges for an island from its node ids and the graph edge generator

        :return: list of graph edges
        """
        logger.warning(f"{__name__} Probably should use subgraph instead")
        assert self.ami_graph.nx_graph is not None
        nx_graph = self.ami_graph.nx_graph
        self.nx_edges = []
        for node_id in self.node_ids:
            edges = nx_graph.edges(node_id, keys=self.ami_graph.nx_graph.is_multigraph())
            for e in edges:
                # normalize edges
                if AmiEdge.is_normalized_edge(e[0], e[1]):
                    self.nx_edges.append(e)
        return self.nx_edges

    def get_node_ids_of_degree(self, node_count):
        """
        gets node_ids in island with given count
        delegates to  AmiGraph.get_node_ids_from_graph_with_degree
        :param node_count:
        :return:
        """
        return AmiGraph.get_node_ids_from_graph_with_degree(self.island_nx_graph, node_count)

    def get_or_create_ami_edges(self):
        self.create_nx_edges()
        self.ami_edges = []
        for edge in self.nx_edges:
            ami_edge = self.ami_graph.get_or_create_ami_edge_from_nx_edge(edge)
            self.ami_edges.append(ami_edge)
        return self.ami_edges

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
        ami_node = self.ami_graph.get_or_create_ami_node(node_id)
        edge_dict = ami_node.create_edge_property_dict()
        return edge_dict
