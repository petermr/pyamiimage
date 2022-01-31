"""skeleton image used by SKNW and then by AmiGraph

This class had a lot of mess and has been refactored"""

import logging
from pathlib import Path
import numpy as np
import networkx as nx
import sknw  # must pip install sknw
import os
import matplotlib.pyplot as plt
from skimage import io
import logging
# local
from pyamiimage.pyimage.ami_image import AmiImage
from ..pyimage.ami_util import AmiUtil

from ..pyimage.bbox import BBox
from ..pyimage.flood_fill import FloodFill
from ..pyimage.ami_graph_all import AmiGraph

logger = logging.getLogger(__name__)

class AmiSkeleton:

    """manages workflow from file to plot.
    creates:
    * binary
    * skeleton
    * sknw nodes and edges
    * networkx graph (often called nx_graph)
    * plots


    May need rationalizatiom with AmiGraph
    """
    # NODE_PTS = "pts"
    # CENTROID = "o"

    logger = logging.getLogger("ami_skeleton")

    def __init__(self, plot_plot=False, title=None):
        self.skeleton_image = None
        self.binary = None
        self.nx_graph = None
        self.edge_xy_list = []
        self.node_xy = []
        self.nodes = []
        self.image = None
        self.path = None
        self.new_binary = None
        self.interactive = False
        self.title = title
        self.plot_plot = plot_plot
        self.islands = None
        self.bboxes = None
        self.thresh = None
        #
        self.ami_graph = None
        self.node_dict = {}
        self.edge_dict = {}

    @classmethod
    def binarize_skeletonize_sknw_nx_graph_plot_TEST(cls, path, plot_plot=True):
        """
        Creates skeleton and nx_graph and plots it

        :param path:
        :param plot_plot:
        :return: AmiSkeleton
        """
        assert path is not None
        path = Path(path)
        skeleton_image = AmiImage.create_white_skeleton_from_file(path)
        # build graph from skeleton
        nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_image)
        if plot_plot:
            self.plot_nx_graph_NX(nx_graph)
        return skeleton_image

    def create_nx_graph_via_skeleton_sknw_NX_GRAPH(self, path):
        """
        Creates a nx_graph
        does it need a path?
        :param path:
        :return: AmiSkeleton
        """
        logger.warning("maybe obsolete")
        assert path is not None
        path = Path(path)
        self.skeleton_image = AmiImage.create_white_skeleton_from_file(path)
        io.imshow(self.skeleton_image)
        io.show()
        assert self.skeleton_image is not None
        # build graph from skeleton
        nx_graph = AmiGraph.create_nx_graph_from_skeleton(self.skeleton_image)
        return nx_graph

    def plot_nx_graph_NX(self, nx_graph, title="skeleton"):
        """
        :param nx_graph:
        :param title:
        :return: None
        """
        """
        requires that nx_graph has been created
graph.node[id]['pts'] : Numpy(x, n), coordinates of nodes points
graph.node[id]['o']: Numpy(n), centried of the node
graph.edge(id1, id2)['pts']: Numpy(x, n), sequence of the edge point
graph.edge(id1, id2)['weight']: float, length of this edge        """

        assert nx_graph is not None
        AmiSkeleton.get_coords_for_nodes_and_edges_from_nx_graph_GRAPH(nx_graph)
        self.plot_edges_nodes_and_title_GRAPH(title)
        return None

    def plot_edges_nodes_and_title_GRAPH(self, title, plot_plot=True):
        """
        Requires nodes and edges to have been created
        :param title:
        :param plot_plot:
        :return:
        """
        for edge_xy in self.edge_xy_list:
            plt.plot(edge_xy[:, 1], np.negative(edge_xy[:, 0]), 'green')
        # draw node by small circle (".")
        plt.plot(self.node_xy[:, 1], np.negative(self.node_xy[:, 0]), 'r.')
        # title and show
        plt.title(title)
        if plot_plot:
            plt.show()

        path = Path(Path(__file__).parent.parent, "temp/figs")
        if not path.exists():
            path.mkdir()
        fig = Path(path, f"{title}.png")
        if fig.exists():
            os.remove(fig)
        plt.savefig(fig, format="png")
        if self.interactive:
            plt.show()

    @classmethod
    def get_coords_for_nodes_and_edges_from_nx_graph_GRAPH(cls, nx_graph):
        """
        creates nodes and edges from networkx graph
        :return: Node
        """
        assert nx_graph is not None
        nodes = nx_graph.nodes()
        node_xy = np.array([nodes[i][AmiSkeleton.CENTROID] for i in nodes])
        # edges by pts (s(tart),e(nd)) appear to be the nodes on each edge
        edge_xy_list = []
        for (s, e) in nx_graph.edges():
            edge_xy = nx_graph[s][e][AmiSkeleton.NODE_PTS]
            edge_xy_list.append(edge_xy)
        return node_xy, edge_xy_list

    def extract_bbox_for_nodes_ISLAND(self, ami_island):
        from pyimage import AmiIsland
        """
        gets bounding box for a list of nodes in

        requires nodes to have been created
        :param ami_island:
        :return: bounding box ((xmin, xmax), (ymin, ymax))
        """
        assert ami_island is not None
        assert type(ami_island) is AmiIsland, f"expected {AmiIsland} found {type(ami_island)}"
        node_xy = self.extract_coords_for_nodes_ISLAND(ami_island)
        # print ("node_xy...", node_xy)
        xx = node_xy[:, 0]
        yy = node_xy[:, 1]
        xmin = int(np.min(xx))
        xmax = int(np.max(xx))
        ymin = int(np.min(yy))
        ymax = int(np.max(yy))
        bbox = BBox(((xmin, xmax), (ymin, ymax)))
        return bbox

    def extract_coords_for_nodes_ISLAND(self, ami_island):
        """
        gets coordinates for a set of nx_graph nodes
        *** NOTE it seems the sknw output has y,x rather than x,y ***

        :param ami_island: normally ints but I suppose could be other
        :return: node_xy as [npoints, 2] ndarray
        """
        assert ami_island is not None
        assert type(ami_island) is AmiIsland, f"expected {AmiIsland} found {type(ami_island)}"
        npoints = len(ami_island)
        node_xy = np.empty([0, 2], dtype=float)
        for isd in ami_island:
            centroid = self.extract_coords_for_node_NX_GRAPH_CLS(isd)
            node_xy = np.append(node_xy, centroid)
        node_xy = np.reshape(node_xy, (npoints, 2))
        return node_xy

    def create_bbox_for_island_ISLAND(self, island):
        bbox0 = self.extract_bbox_for_nodes_ISLAND(island)
        bbox = BBox(bbox0)
        return bbox

    def read_image_plot_component(self, component_index, image):
        """
        Convenience method to read imag, get components and plot given one
        :param component_index:
        :param image:
        :return:
        """
        nx_graph = self.create_nx_graph_via_skeleton_sknw_NX_GRAPH(image)
        # self.get_coords_for_nodes_and_edges_from_nx_graph_GRAPH(nx_graph)
        # ami_graph = AmiGraph.
        # TODO needs AmiGraph adding
        islands = self.get_ami_islands_from_nx_graph_GRAPH()
        island = islands[component_index]
        self.plot_island_ISLAND(island)

    def plot_island_ISLAND(self, component):
        """
        Plots a given component
        :param component:
        :return:
        """
        start_node_index = list(component)[0]  # take first node
        start_node = self.nodes[start_node_index]
        start_pixel = start_node[self.NODE_PTS][0]  # may be a list of xy for a complex node always pick first
        flooder = FloodFill()
        pixels = flooder.flood_fill(self.binary, start_pixel)
        if self.interactive:
            flooder.plot_used_pixels()

    def create_and_plot_all_components_TEST(self, path, min_size=None):
        """

        :param path:
        :param min_size:
        :return:
        """
        if min_size is None:
            min_size = [30, 30]
        self.create_nx_graph_via_skeleton_sknw_NX_GRAPH(path)
        nodes_xy, edges_xy = self.get_coords_for_nodes_and_edges_from_nx_graph_GRAPH(self.nx_graph)
        components = self.get_ami_islands_from_nx_graph_GRAPH()
        assert self.nx_graph is not None
        self.islands = self.get_ami_islands_from_nx_graph_GRAPH()
        bboxes = self.islands
        for component, bbox in zip(components, bboxes):
            w, h = AmiSkeleton.get_width_height_BBOX(bbox)
            if min_size[0] < w or min_size[1] < h:
                self.plot_island_ISLAND(component)

    def get_ami_islands_from_image_OBSOLETE(self, image):
        """
        read image, calculate islands

        :param image:
        :return: list of islands in arbitrary order
        """
        nx_graph = self.create_nx_graph_via_skeleton_sknw_NX_GRAPH(image)
        self.get_coords_for_nodes_and_edges_from_nx_graph_GRAPH(nx_graph)
        return self.get_ami_islands_from_nx_graph_GRAPH()
