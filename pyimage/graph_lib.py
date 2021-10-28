import numpy as np
import networkx as nx
import copy
from networkx.algorithms import tree
from skimage import morphology, io, color
from skan.pre import threshold
import sknw  # must pip install sknw
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from lxml.etree import Element

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
    NODE_PTS = "pts"
    CENTROID = "o"

    logger = logging.getLogger("ami_skeleton")

    def __init__(self):
        self.skeleton = None
        self.binary = None
        self.nx_graph = None
        self.edge_xy_list = []
        self.node_xy = []
        self.nodes = []
        self.image = None
        self.path = None

    def create_grayscale_from_file(self, path):
        """
        Reads an image from path and creates a grayscale (w. skimage)
        May throw image exceptions (not trapped)
        :param path: 
        :return: single channel grayscale
        """
        assert path is not None
        self.path = path
        self.image = color.rgb2gray(io.imread(path))
        return self.image

    def create_white_skeleton_from_file(self, path):
        """
        the image may be inverted so the highlights are white

        :param path: path with image
        :return: AmiSkeleton
        """
        # image = io.imread(file)
        assert path is not None
        path = Path(path)
        self.image = self.create_grayscale_from_file(path)
        self.skeleton = self.create_white_skeleton_from_image(self.image)
        return self.skeleton

    def create_white_skeleton_from_image(self, image):
        """
        create AmiSkeleton based on white components of image

        :param image:
        :return: AmiSkeleton
        """
        assert image is not None
        self.create_white_binary_from_image(image)
        self.skeleton = morphology.skeletonize(self.binary)
        return self.skeleton

    def create_white_binary_from_image(self, image):
        """
        Create a thresholded, binary image from a grayscale

        :param image: grayscale image
        :return: binary with white pixels as signal
        """
        self.binary, self.thresh = self.create_thresholded_image_and_value(image)
        # print("thresh", self.thresh)
        self.binary = np.invert(self.binary)

    def create_thresholded_image_and_value(self, image):
        """
        Thresholded image and (attempt) to get threshold
        The thresholded image is OK but the threshold value may not yet work

        :param image: grayscale
        :return: thresholded image, threshold value (latter may not work)
        """
        # print("image", np.amin(image), np.amax(image))
        t_image = threshold(image)
        # print("t_image", np.amin(t_image), np.amax(t_image))
        tt = np.where(t_image > 0)
        # print("tt", np.amin(tt), np.amax(tt))
        return t_image, tt

    def binarize_skeletonize_sknw_nx_graph_plot(self, path):
        """
        Creates skeleton and nx_graph and plots it
        :param path:
        :return: AmiSkeleton
        """
        assert path is not None
        path = Path(path)
        self.skeleton = self.create_white_skeleton_from_file(path)
        # build graph from skeleton
        self.nx_graph = sknw.build_sknw(self.skeleton)
        self.plot_nx_graph()
        return self.skeleton

    def create_nx_graph_via_skeleton_sknw(self, path):
        """
        Creates a nx_graph
        :param path:
        :return: AmiSkeleton
        """
        assert path is not None
        path = Path(path)
        self.skeleton = self.create_white_skeleton_from_file(path)
        # build graph from skeleton
        self.nx_graph = sknw.build_sknw(self.skeleton)
        return self.nx_graph

    def plot_nx_graph(self, title="skeleton"):
        """

        :param title:
        :return: None
        """
        """
        requires that nx_graph has been created
graph.node[id]['pts'] : Numpy(x, n), coordinates of nodes points
graph.node[id]['o']: Numpy(n), centried of the node
graph.edge(id1, id2)['pts']: Numpy(x, n), sequence of the edge point
graph.edge(id1, id2)['weight']: float, length of this edge        """

        assert self.nx_graph is not None
        self.get_nodes_and_edges_from_nx_graph()
        self.plot_edges_nodes_and_title(title)
        return None

    def plot_edges_nodes_and_title(self, title):
        """
        Requires nodes and edges to have been created
        :param title:
        :return:
        """
        for edge_xy in self.edge_xy_list:
            plt.plot(edge_xy[:, 1], np.negative(edge_xy[:, 0]), 'green')
        # draw node by small circle (".")
        plt.plot(self.node_xy[:, 1], np.negative(self.node_xy[:, 0]), 'r.')
        # title and show
        plt.title(title)
        plt.show()

    def get_nodes_and_edges_from_nx_graph(self):
        """
        creates nodes and edges from graph
        :return: Node
        """
        assert self.nx_graph is not None
        self.nodes = self.nx_graph.nodes()
        self.node_xy = np.array([self.nodes[i]['o'] for i in self.nodes])
        # draw edges by pts (s(tart),e(nd)) appear to be the nodes on each edge
        self.edge_xy_list = []
        for (s, e) in self.nx_graph.edges():
            edge_xy = self.nx_graph[s][e]['pts']
            self.edge_xy_list.append(edge_xy)
        return None

    def extract_bbox_for_nodes(self, node_ids):
        """
        gets bounding box for a list of nodes in

        requires nodes to have been created
        :param node_ids:
        :return: bounding box ((xmin, xmax), (ymin, ymax))
        """
        assert node_ids is not None
        node_xy = self.extract_coords_for_nodes(node_ids)
        xx = node_xy[:, 0]
        yy = node_xy[:, 1]
        xmin = int(np.min(xx))
        xmax = int(np.max(xx))
        ymin = int(np.min(yy))
        ymax = int(np.max(yy))
        bbox = ((xmin, xmax), (ymin, ymax))
        return bbox

    def extract_coords_for_nodes(self, node_ids):
        """
        gets coordinates for a set of nx_graph nodes
        :param node_ids:
        :return: node_xy as [npoints, 2] ndarray
        """
        assert node_ids is not None
        npoints = len(node_ids)
        node_xy = np.empty([0, 2], dtype=float)
        for id in node_ids:
            centroid = self.nx_graph.nodes[id][AmiSkeleton.CENTROID]
            node_xy = np.append(node_xy, centroid)
        node_xy = np.reshape(node_xy, (npoints, 2))
        return node_xy

    def create_bboxes_for_connected_components(self):
        """

        :param nx_graph:
        :return: list of bboxes
        """

        assert self.nx_graph is not None
        connected_components = self.get_connected_components()
        bboxes = []
        for component in connected_components:
            bboxes.append(self.extract_bbox_for_nodes(component))
        return bboxes

    def get_connected_components(self):
        """
        Get the pixel-disjoint "islands"

        :param nx_graph:
        :return:
        """
        assert self.nx_graph is not None
        connected_components = list(nx.algorithms.components.connected_components(self.nx_graph))
        return connected_components

    def flood_fill(self, component, color=0x00ff0000):
        """
        Fills the component with the color
        :param component:
        :param color: default is "#ff0"
        :return: new_image
        """
        assert component is not None
        assert self.nx_graph is not None
        assert self.nodes is not None
        assert len(self.nodes) > 0
        assert self.binary is not None
        start = 0
        node = self.nodes[start]
        xy = node["pts"][start]
        # xy = [xy[1], xy[0]]
        # print("xy+", xy)
        print(self.binary)
        img = self.binary
        # xy = (1,1)
        print ("shape", self.image.shape)
        # new_image = morphology.flood_fill(self.binary, xy, color)
        color = self.binary[xy[0], xy[1]]
        new_image = self.flood_fill_binary(self.binary, xy, color )
        print("new", new_image)
        return new_image

    def flood_fill_binary(self, binary_image, seed_xy, color):
        assert seed_xy is not None
        assert binary_image is not None
        self.binary_image = binary_image
        neighbours = self.get_neighbours(self.binary_image, seed_xy)
        print("neigh", neighbours)
        for i in range(3):
            for j in range(3):
                seed_xy = (seed_xy[0] + i, seed_xy[1] + j)
                neighbours = self.get_neighbours(self.binary_image, seed_xy)
                print("neigh", neighbours)
        return None

    @classmethod
    def get_neighbours(cls, image, xy):
        i = xy[0]
        j = xy[1]
        neighbours = image[max(i - 1, 0):min(i + 2, image.shape[0]), max(j - 1, 0):min(j + 2, image.shape[1])]
        return neighbours

    def parse_hocr_title(self, title):
        """
         title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7"

        :param title:
        :param kw:
        :return:
        """
        if title is None:
            return None
        parts = title.split("; ")
        title_dict = {}
        for part in parts:
            pp = part.split()
            kw = pp[0]
            if kw == "bbox":
                val = ((pp[1], pp[3]), (pp[2], pp[4]))
            else:
                val = pp[1:]
            title_dict[kw] = val
            # print(f"kw {kw} val {val}")
            # print(f"title_dict {title_dict}")
        return title_dict

    def create_svg_text_box_from_hocr(self, bbox, txt):

        g = Element("g")
        g.attrib["xmlns"] = "http://www.w3.org/2000/svg"

        rect = Element("rect")
        rect.attrib["xmlns"] = "http://www.w3.org/2000/svg"
        rect.attrib["x"] = bbox[0][0]
        rect.attrib["width"] = str(int(bbox[0][1]) - int(bbox[0][0]))
        height = int(bbox[1][1]) - int(bbox[1][0])
        rect.attrib["y"] = str(int(bbox[1][0]) - height)  # kludge for offset of inverted text
        rect.attrib["height"] = str(height)
        rect.attrib["stroke-width"] = "1.0"
        rect.attrib["stroke"] = "red"
        rect.attrib["fill"] = "none"
        g.append(rect)

        text = Element("text")
        text.attrib["xmlns"] = "http://www.w3.org/2000/svg"
        text.attrib["x"] = bbox[0][0]
        text.attrib["y"] = bbox[1][0]
        text.attrib["font-size"] = str(0.9 * height)
        text.attrib["stroke"] = "blue"
        text.attrib["font-family"] = "sans-serif"
        text.text = txt

        g.append(text)

        return g

class AmiGraph:
    """holds AmiNodes and AmiEdges
    may also hold subgraphs
    """

    logger = logging.getLogger("ami_graph")

    def __init__(self, generate_nodes=True):
        """create fro nodes and edges"""
        self.ami_node_dict = {}
        self.ami_edge_dict = {}
        self.generate_nodes = generate_nodes
        self.nx_graph = None
        self.edges = None
        self.island_list = None
        self.node_dict = None

    def read_nodes(self, nodes):
        """create a list of AmiNodes """
        if nodes is not None:
            for node in nodes:
                self.add_raw_node(node)

    def add_raw_node(self, raw_node, fail_on_duplicate=False):
        """add a raw node either a string or string-indexed dict
        if already a dict, deepcopy it
        if a primitive make a node_dict and start it with raw_node as id
        :raw_node: node to add, must have key
        :fail_on_duplicate: if true fail if key already exists
        """
        if raw_node is not None:
            ami_node = AmiNode()
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
        self.edges = edges
        if len(self.ami_node_dict.keys()) == 0 and self.generate_nodes:
            self.generate_nodes_from_edges()
            print("after node generation", str(self))
        for i, edge in enumerate(self.edges):
            idx = "e" + str(i)
            self.add_edge(edge, idx)

    def add_edge(self, raw_edge, idx, fail_on_duplicate=True):
        if raw_edge is None:
            raise AmiGraphError("cannot add edge=None")
        # node0 =
        edge1 = ("n" + str(raw_edge[0]), "n" + str(raw_edge[1]))
        self.ami_edge_dict[idx] = edge1

    def generate_nodes_from_edges(self):
        if self.edges is not None:
            for edge in self.edges:
                self.add_raw_node(edge[0])
                self.add_raw_node(edge[1])

    @classmethod
    def create_ami_graph(cls, skeleton_image):
        """Uses Sknw to create a graph object within a new AmiGraph"""
        ami_graph = AmiGraph()
        ami_graph.nx_graph, nodes, edges = sknw.build_sknw(skeleton_image)
        ami_graph.read_nodes(nodes)
        ami_graph.read_edges(edges)
        return ami_graph

    def get_graph_info(self):
        if self.nx_graph is None:
            self.logger.warning("Null graph")
            return
        print("graph", self.nx_graph)
        self.island_list = list(nx.connected_components(self.nx_graph))
        print("islands", self.island_list)
        mst = tree.maximum_spanning_edges(self.nx_graph, algorithm="kruskal", data=True)
        # mst = tree.minimum_spanning_tree(graph, algorithm="kruskal")
        nx_edgelist = list(mst)
        for edge in nx_edgelist:
            print(edge[0], edge[1], "pts in edge", len(edge[2]['pts']))
        for step in nx_edgelist[0][2]['pts']:
            print("step", step)
        nodes = self.nx_graph.nodes
        self.node_dict = {i: (nodes[node]["o"][0], nodes[node]["o"][1]) for i, node in enumerate(nodes)}

    @classmethod
    def set_bbox_pixels_to_color(cls, bbox, image, color=255):
        """sets all pixels in box to uniform color

        :param bbox:
        :param image:
        :return: modified image
        """
        xx = bbox[0]
        yy = bbox[1]
        image[xx[0]:xx[1], yy[0]:yy[1]] = color
        return image


    def __str__(self):
        s = "nodes: " + str(self.ami_node_dict) + \
            "\n edges: " + str(self.ami_edge_dict)
        return s


class AmiNode:
    def __init__(self):
        self.node_dict = {}


class AmiEdge:
    def __init__(self):
        pass


class AmiIsland:
    """A connected group of pixels"""
    def __init__(self):
        self.nodes = None


class AmiGraphError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0]])

    # sknw.example1()
    sknw.example2horse()  # works
    # sknw.example3() # needs flipping White to black
    # sknw.example4() # needs flipping White to black
