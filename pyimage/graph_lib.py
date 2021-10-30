import numpy as np
import networkx as nx
import copy
from networkx.algorithms import tree
from skimage import morphology, io, color
from skan.pre import threshold
import sknw  # must pip install sknw
import logging
from pathlib import Path
from collections import deque
from lxml.etree import Element, QName
from lxml import etree
import matplotlib.pyplot as plt


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

    E_G = 'g'
    E_RECT = 'rect'
    E_SVG = 'svg'
    E_TEXT = "text"

    A_BBOX = "bbox"
    A_FILL = "fill"
    A_FONT_SIZE = "font-size"
    A_FONT_FAMILY = "font-family"
    A_HEIGHT = "height"
    A_STROKE = "stroke"
    A_STROKE_WIDTH = "stroke-width"
    A_TITLE = "title"
    A_WIDTH = "width"
    A_XLINK = 'xlink'
    A_X = "x"
    A_Y = "y"

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
        self.new_binary = None

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
        self.binary = np.invert(self.binary)

    def create_thresholded_image_and_value(self, image):
        """
        Thresholded image and (attempt) to get threshold
        The thresholded image is OK but the threshold value may not yet work

        :param image: grayscale
        :return: thresholded image, threshold value (latter may not work)
        """

        t_image = threshold(image)
        tt = np.where(t_image > 0) # above threshold
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
        graph_nodes = self.nx_graph.nodes()
        self.nodes = graph_nodes
        self.node_xy = np.array([self.nodes[i][self.CENTROID] for i in self.nodes])
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

        :return:
        """

        self.get_nodes_and_edges_from_nx_graph()
        assert self.nx_graph is not None
        connected_components = list(nx.algorithms.components.connected_components(self.nx_graph))
        return connected_components

    def read_image_plot_component(self, component_index, image):
        """
        Convenience method to read imag, get components and plot given one
        :param component_index:
        :param image:
        :return:
        """
        components = self.get_connected_components_from_image(image)
        component = components[component_index]
        self.plot_component(component)

    def plot_component(self, component):
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
        flooder.plot_used_pixels()

    def create_and_plot_all_components(self, path, min_size=None):
        if min_size is None:
            min_size = [30,30]
        self.create_nx_graph_via_skeleton_sknw(path)
        self.get_nodes_and_edges_from_nx_graph()
        components = self.get_connected_components()
        bboxes = self.create_bboxes_for_connected_components()
        for component, bbox in zip(components, bboxes):
            w, h = AmiSkeleton.get_width_height(bbox)
            if min_size[0] < w or min_size[1] < h:
                self.plot_component(component)

    @classmethod
    def get_width_height(cls, bbox):
        """

        :param bbox: tuple of tuples ((x0,x1), (y0,y1))
        :return: (width, height) tuple
        """
        """
        needs to have its own class
        """
        width = bbox[0][1] - bbox[0][0]
        height = bbox[1][1] - bbox[1][0]
        return (width, height)

    def get_connected_components_from_image(self, image):
        """
        read image, calculate components

        :param image:
        :return: list of components in arbitrary order
        """
        self.create_nx_graph_via_skeleton_sknw(image)
        self.get_nodes_and_edges_from_nx_graph()
        components = self.get_connected_components()
        return components

    def parse_hocr_title(self, title):
        """
         title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7"

        :param title:
        :return:
        """
        if title is None:
            return None
        parts = title.split("; ")
        title_dict = {}
        for part in parts:
            vals = part.split()
            kw = vals[0]
            if kw == self.A_BBOX:
                val = ((vals[1], vals[3]), (vals[2], vals[4]))
            else:
                val = vals[1:]
            title_dict[kw] = val
        return title_dict

    def create_svg_from_hocr(self, hocr_html):
        """

        :param hocr_html:
        :return:
        """
        html = etree.parse(hocr_html)
        word_spans = html.findall("//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
        E_SVG = 'svg'
        svg = Element(QName(XMLNamespaces.svg, self.E_SVG), nsmap={
            self.E_SVG: XMLNamespaces.svg,
            self.A_XLINK: XMLNamespaces.xlink,
        })
        for word_span in word_spans:
            title = word_span.attrib[self.A_TITLE]
            title_dict = self.parse_hocr_title(title)
            bbox = title_dict[self.A_BBOX]
            text = word_span.text
            g = self.create_svg_text_box_from_hocr(bbox, text)
            svg.append(g)
        bb = etree.tostring(svg, encoding='utf-8', method='xml')
        s = bb.decode("utf-8")
        path_svg = Path(Path(__file__).parent.parent, "temp", "textbox.svg")
        with open(path_svg, "w", encoding="UTF-8") as f:
            f.write(s)
            print(f"Wrote textboxes to {path_svg}")

    def create_svg_text_box_from_hocr(self, bbox, txt):

        g = Element(QName(XMLNamespaces.svg, self.E_G))
        height = int(bbox[1][1]) - int(bbox[1][0])

        rect = Element(QName(XMLNamespaces.svg, self.E_RECT))
        rect.attrib[self.A_X] = bbox[0][0]
        rect.attrib[self.A_WIDTH] = str(int(bbox[0][1]) - int(bbox[0][0]))
        rect.attrib[self.A_Y] = str(int(bbox[1][0]) - height)  # kludge for offset of inverted text
        rect.attrib[self.A_HEIGHT] = str(height)
        rect.attrib[self.A_STROKE_WIDTH] = "1.0"
        rect.attrib[self.A_STROKE] = "red"
        rect.attrib[self.A_FILL] = "none"
        g.append(rect)

        text = Element(QName(XMLNamespaces.svg, self.E_TEXT))
        text.attrib[self.A_X] = bbox[0][0]
        text.attrib[self.A_Y] = bbox[1][0]
        text.attrib[self.A_FONT_SIZE] = str(0.9 * height)
        text.attrib[self.A_STROKE] = "blue"
        text.attrib[self.A_FONT_FAMILY] = "sans-serif"

        text.text = txt

        g.append(text)

        return g

class XMLNamespaces:
    svg = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"


class FloodFill:
    """creates a list of flood_filling pixels given a seed"""

    def __init__(self):
        self.start_pixel = None
        self.binary = None
        self.filling_pixels = None

    def flood_fill(self, binary_image, start_pixel):
        """

        :param binary_image: Not altered
        :return: (filled image, set of filling pixels)
        """
        # self.binary = self.binary.astype(int)

        self.start_pixel = start_pixel
        self.binary = binary_image
        self.filling_pixels = self.get_filling_pixels()
        return self.filling_pixels

    def get_filling_pixels(self):
        new_image = np.copy(self.binary)
        xy = self.start_pixel
        xy_deque = deque()
        xy_deque.append(xy)
        filling_pixels = set()
        while xy_deque:
            xy = xy_deque.popleft()
            self.binary[xy[0], xy[1]] = 0 # unset pixel
            neighbours_list = self.get_neighbours(xy)
            for neighbour in neighbours_list:
                neighbour_xy = (neighbour[0], neighbour[1]) # is this necessary??
                if not neighbour_xy in filling_pixels:
                    filling_pixels.add(neighbour_xy)
                    xy_deque.append(neighbour_xy)
                else:
                    pass
        return filling_pixels

    def get_neighbours(self, xy):
        i = xy[0]
        j = xy[1]
        w = 3
        h = 3
        neighbours = []
        # I am sure there's a more pythonic way
        for i in range(w):
            ii = xy[0] + i - 1
            if ii < 0 or ii >= self.binary.shape[0]:
                continue
            for j in range(h):
                jj = xy[1] + j - 1
                if jj >= 0 or jj < self.binary.shape[1]:
                    if self.binary[ii][jj] == 1:
                        neighbours.append((ii, jj))
        return neighbours

    def plot_used_pixels(self):
        used_image = self.create_image_of_filled_pixels()
        fig, ax = plt.subplots()
        ax.imshow(used_image)
        plt.show()

    def create_image_of_filled_pixels(self):
        used_image = np.zeros(self.binary.shape, dtype=bool)
        for pixel in self.filling_pixels:
            used_image[pixel[0], pixel[1]] = 1
        return used_image


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
