import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage import data
from pathlib import Path
from skimage.filters import threshold_otsu
import copy
from networkx.algorithms import tree
from skimage import morphology, io
from skan.pre import threshold
from pmrsknw import PmrSknw
import matplotlib.pyplot as plt


class AmiSkeleton:
    """manages workflow from file to plot.
    creates:
    * binary
    * skeleton
    * sknw nodes and edges
    * networkx graph
    * plots


    May need rationalizatiom with AmiGraph
    """
    NODE_PTS = "pts"
    CENTROID = "o"

    def __init__(self):
        self.skeleton = None
        self.binary = None
        self.nx_graph = None
        self.edge_xy_list = []
        self.node_xy = []
        self.nodes = []

    def create_white_skeleton(self, file):
        image = io.imread(file)
        self.skeleton = self.create_white_skeleton_from_image(image)
        return self.skeleton

    def create_white_skeleton_from_image(self, image):
        self.binary = threshold(image)
        self.binary = np.invert(self.binary)
        self.skeleton = morphology.skeletonize(self.binary)
        return self.skeleton

    def binarize_skeletonize_sknw_nx_graph_plot(self, text):
        self.skeleton = self.create_white_skeleton(text)
        # build graph from skeleton
        self.nx_graph = sknw.build_sknw(self.skeleton)
        self.plot_nx_graph(self.nx_graph)

    def create_nx_graph_via_skeleton_sknw(self, file):
        self.skeleton = self.create_white_skeleton(file)
        # build graph from skeleton
        self.nx_graph = sknw.build_sknw(self.skeleton)
        return self.nx_graph

    def plot_nx_graph(self, nx_graph, title="skeleton"):
        """
graph.node[id]['pts'] : Numpy(x, n), coordinates of nodes points
graph.node[id]['o']: Numpy(n), centried of the node
graph.edge(id1, id2)['pts']: Numpy(x, n), sequence of the edge point
graph.edge(id1, id2)['weight']: float, length of this edge        """

        self.get_nodes_and_edges_from_nx_graph(nx_graph)
        self.plot_edges_nodes_and_title(nx_graph, title)
        return

    def plot_edges_nodes_and_title(self, nx_graph, title):
        for edge_xy in self.edge_xy_list:
            plt.plot(edge_xy[:, 1], np.negative(edge_xy[:, 0]), 'green')
        # draw node by small circle (".")
        plt.plot(self.node_xy[:, 1], np.negative(self.node_xy[:, 0]), 'r.')
        # title and show
        plt.title(title)
        plt.show()

    def get_nodes_and_edges_from_nx_graph(self, nx_graph):
        self.nodes = nx_graph.nodes()
        self.node_xy = np.array([self.nodes[i]['o'] for i in self.nodes])
        # draw edges by pts (s(tart),e(nd)) appear to be the nodes on each edge
        self.edge_xy_list = []
        for (s, e) in nx_graph.edges():
            edge_xy = nx_graph[s][e]['pts']
            self.edge_xy_list.append(edge_xy)


class AmiGraph():
    """holds AmiNodes and AmiEdges
    may also hold subgraphs
    """

    def __init__(self, generate_nodes=True):
        """create fro nodes and edges"""
        self.ami_node_dict = {}
        self.ami_edge_dict = {}
        self.generate_nodes = generate_nodes
        self.nx_graph = None

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
            self.logger.warn("node cannot be None")

    def read_edges(self, edges):
        self.edges = edges
        if len(self.ami_node_dict.keys()) == 0 and self.generate_nodes:
            self.generate_nodes_from_edges()
            print("after node generation", str(self))
        for i, edge in enumerate(self.edges):
            id = "e" + str(i)
            self.add_edge(edge, id)

    def add_edge(self, raw_edge, id, fail_on_duplicate=True):
        if raw_edge is None:
            raise AmiGraphError("cannot add edge=None")
        # node0 =
        edge1 = ("n" + str(raw_edge[0]), "n" + str(raw_edge[1]))
        self.ami_edge_dict[id] = edge1

    def generate_nodes_from_edges(self):
        if self.edges is not None:
            for edge in self.edges:
                self.add_raw_node(edge[0])
                self.add_raw_node(edge[1])

    @classmethod
    def create_ami_graph(self, skeleton_image):
        """Uses Sknw to create a graph object within a new AmiGraph"""
        ami_graph = AmiGraph()
        ami_graph.nx_graph, nodes, edges = PmrSknw().build_sknw(skeleton_image)
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

    def __str__(self):
        s = "nodes: " + str(self.ami_node_dict) + \
            "\n edges: " + str(self.ami_edge_dict)
        return s


class AmiNode():
    def __init__(self):
        self.node_dict = {}


class AmiEdge():
    def __init__(self):
        pass


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

    sknw = PmrSknw()
    # sknw.example1()
    sknw.example2horse()  # works
    # sknw.example3() # needs flipping White to black
    # sknw.example4() # needs flipping White to black
