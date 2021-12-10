"""
tests AmiGraph, AmiNode, AmiEdge, AmiIsland
"""

from pathlib import PosixPath
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
import sknw
import unittest
from skimage import io

from pyimage.ami_skeleton import AmiSkeleton
from pyimage.ami_graph_all import AmiNode, AmiIsland, AmiGraph
from test.resources import Resources
from pyimage.ami_image import AmiImage
from pyimage.util import Util
from pyimage.bbox import BBox


class TestAmiGraph:

    def setup_method(self, method):
        arrows1 = io.imread(Resources.BIOSYNTH1_ARROWS)
        assert arrows1.shape == (315, 1512)
        pass


    @unittest.skip("background")
    def test_sknw_example(self):
        """
        From the SKNW docs
        not really a test, more a debug
        :return:
        """
        """
        from https://github.com/Image-Py/sknw
        Skeleton Network
build net work from nd skeleton image

graph = sknw.build_sknw(ske， multi=False)
ske: should be a nd skeleton image
multi: if True，a multigraph is retured, which allows more than one edge between 
two nodes and self-self edge. default is False.

return: is a networkx Graph object

graph detail:
graph.nodes[id]['pts'] : Numpy(x, n), coordinates of nodes points
graph.nodes[id]['o']: Numpy(n), centried of the node
graph.edges(id1, id2)['pts']: Numpy(x, n), sequence of the edge point
graph.edges(id1, id2)['weight']: float, length of this edge

if it's a multigraph, you must add a index after two node id to get the edge, 
like: graph.edge(id1, id2)[0].

build Graph by Skeleton, then plot as a vector Graph in matplotlib.

from skimage.morphology import skeletonize
from skimage import data
import sknw

# open and skeletonize
img = data.horse()
ske = skeletonize(~img).astype(np.uint16)  # the tilde (~) inverts the binary image

# build graph from skeleton
graph = sknw.build_sknw(ske)
plt.imshow(img, cmap='gray')

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

# title and show
plt.title('Build Graph')
plt.show()"""

        show_plot = False
        show_plot = True
        # open and skeletonize
        multi = False  # edge/node access needs an array for True
        img = data.horse()
        ske = skeletonize(~img).astype(np.uint16)

        graph = sknw.build_sknw(ske, multi=multi)
        assert graph.number_of_nodes() == 22 and graph.number_of_edges() == 22
        # theres a cycle 9, 13, 14

        # draw image
        plt.imshow(img, cmap='gray')

        assert str(graph.nodes[0].keys()) == "dict_keys(['pts', 'o'])",\
            "nodes have 'pts' and 'o "
        # this is a 2-array with ["pts", "weights'] Not sure what th structure is
        assert f"{list(graph[0][2])[:1]}" == "['pts']"
        assert len(graph[0][2]['pts']) == 18
        assert str(graph[0][2].keys()) == "dict_keys(['pts', 'weight'])"

        print(f"\n=========nodes=========")
        prnt = True
        nprint = 3  # print first nprint
        for i, (s, e) in enumerate(graph.edges()):
            if i < nprint and prnt:
                print(f"{s} {e} {graph[s][e].keys()}")
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        print(f"=========neighbours=========")
        nodes = graph.nodes()
        # print neighbours
        for node in graph.nodes():
            print(f"neighbours {node} {list(graph.neighbors(node))}")

        # coordinates are arranged y-array, x-array
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

        # title and show
        plt.title('Build Graph')
        if show_plot:
            plt.show()

        print("================")
        print(f"edge02 {list(graph.edges[(0, 2)])}")
        print("=======0=========")
        edges_ = list(graph.edges[(0, 2)])[0]
        print(f"edge02b {type(edges_)} {edges_}")
        print("=======2=========")
        print(f"edge02c {graph.edges[(0, 2)]}")
        print("=======3=========")

        # this is a tree so only got one component
        assert nx.algorithms.components.number_connected_components(graph) == 1
        connected_components = list(nx.algorithms.components.connected_components(graph))
        assert connected_components == [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}]
        assert connected_components[0] == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}

    @unittest.skip("exploration")
    def test_sknw_5_islands(self):
        """
        This checks all the fields that sknw returns
        :return:
        """
        # island5_skel = AmiGraph.create_ami_graph(Resources.ISLANDS_5_SKEL)
        # print (f"island5_skel = {island5_skel}")
        # assert type(island5_skel) is str, f"type {type(island5_skel)} {island5_skel} should be {str}"
        skel_path = Resources.BIOSYNTH1_ARROWS
        Util.check_type_and_existence(skel_path, PosixPath)

        skeleton_array = AmiImage.create_white_skeleton_from_file(skel_path)
        Util.check_type_and_existence(skeleton_array, np.ndarray)

        nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)
        Util.check_type_and_existence(nx_graph, nx.classes.graph.Graph)

        Util.check_type_and_existence(nx_graph.nodes, nx.classes.reportviews.NodeView)
        assert list(nx_graph.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                        18, 19, 20, 21, 22, 23, 24, 25, 26]
        Util.check_type_and_existence(nx_graph.edges, nx.classes.reportviews.EdgeView)
        assert list(nx_graph.edges) == [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6),
                                        (8, 19), (9, 19), (10, 12), (11, 13), (12, 13), (12, 18),
                                        (13, 14), (13, 15), (16, 18), (17, 18), (18, 20), (19, 26),
                                        (21, 24), (22, 24), (23, 24), (24, 25)]

        node1ps = nx_graph.nodes[1]["pts"]
        # print(f"node1ps {node1ps}")
        node1ps0 = node1ps[0]
        # print("node1ps0", node1ps0)
        assert str(node1ps) == "[[ 83 680]]"

        if False:
            pts_ = nx_graph.edges[(1, 2)]["pts"]
            print(f"pts_ {pts_}")
            assert str(pts_) == "[[ 83, 680]]"

    def test_islands(self):
        """
        Create island_node_id_sets using sknw/NetworkX and check basic properties
        :return:
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)


        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        print("components", connected_components)
        assert nx.algorithms.components.number_connected_components(nx_graph) == 4
        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        assert type(connected_components) is list, f"type of connected components should be list"
        assert connected_components == [{0, 1, 2, 3, 4, 5, 6, 7},
                                        {8, 9, 26, 19},
                                        {10, 11, 12, 13, 14, 15, 16, 17, 18, 20},
                                        {21, 22, 23, 24, 25}]


        assert type(connected_components[0]) is set and len(connected_components[0]) == 8, \
            f"components should be sets and first len == 8"
        assert type(list(connected_components[0])[0]) is int, f"members should be int"

        ami_graph = AmiGraph(nx_graph)
        ami_graph.read_nx_graph(nx_graph)
        island_node_id_sets = ami_graph.get_or_create_ami_islands()
        assert len(island_node_id_sets) == 4
        assert type(island_node_id_sets[0]) is AmiIsland
        assert island_node_id_sets[0].node_ids == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_set(self):
        """
        basic syntax
        :return:
        """
        sss = set({1,2})
        assert type(sss) is set, f"sss should be set, found {type(sss)})"

    def test_nodes(self):
        """
        Tests
        :return:
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        ami_graph = AmiGraph(nx_graph)
        ami_graph.read_nx_graph(nx_graph)
        nodex = AmiNode(nx_graph=nx_graph, node_id=(list(nx_graph.nodes)[0]))
        xy = nodex.get_or_create_centroid_xy()
        assert xy == [844.0, 82.0]

    def test_bboxes(self):
        """
        Create bounding boxes for islands
        :return:
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        ami_graph = AmiGraph(nx_graph)
        islands = ami_graph.get_or_create_ami_islands()
        assert len(islands) == 4, "arrows"
        bbox_list = []
        for island in islands:
            bbox = island.get_or_create_bbox()
            assert type(bbox) is BBox
            bbox_list.append(bbox)
        assert len(bbox_list) == 4
        # this is horrible and fragile, need __eq__ for bbox
        assert str(bbox_list[0]) == "[[661, 863], [82, 102]]", f"bbox_list[0] is {bbox_list[0]}"
