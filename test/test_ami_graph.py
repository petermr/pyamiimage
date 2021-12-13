"""
tests AmiGraph, AmiNode, AmiEdge, AmiIsland
"""

from pathlib import PosixPath
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
from skimage import filters
import sknw
import unittest
from skimage import io, morphology
from skimage.measure import approximate_polygon, subdivide_polygon
import logging

from pyimage.ami_graph_all import AmiNode, AmiIsland, AmiGraph, AmiEdge
from test.resources import Resources
from pyimage.ami_image import AmiImage
from pyimage.util import Util
from pyimage.bbox import BBox

logger = logging.getLogger(__name__)
interactive = True
interactive = False


class TestAmiGraph:

    def setup_method(self, method):
        self.arrows1 = io.imread(Resources.BIOSYNTH1_ARROWS)
        assert self.arrows1.shape == (315, 1512)
        self.arrows1 = np.where(self.arrows1 < 127, 0, 255)
        self.nx_graph_arrows1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        # self.arrows1 = filters.threshold_mean(self.arrows1)
        # assert self.arrows1.shape == (1,2,3)

        prisma = io.imread(Resources.PRISMA)
        assert prisma.shape == (667, 977, 4)
        self.nx_graph_prisma = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.PRISMA)

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

        # open and skeletonize
        multi = False  # edge/node access needs an array for True
        img = data.horse()
        ske = skeletonize(~img).astype(np.uint16)

        graph = sknw.build_sknw(ske, multi=multi)
        assert graph.number_of_nodes() == 22 and graph.number_of_edges() == 22
        # theres a cycle 9, 13, 14

        # draw image
        if interactive:
            plt.imshow(img, cmap='gray')

        assert str(graph.nodes[0].keys()) == "dict_keys(['pts', 'o'])",\
            "nodes have 'pts' and 'o "
        # this is a 2-array with ["pts", "weights'] Not sure what th structure is
        # this is an edges generator, so may need wrapping in a list
        edges_gen = graph[0][2]  # plural because there may be multiple edges between nodes
        assert f"{list(edges_gen)[:1]}" == "['pts']"  # pts are connected points in an edge
        assert len(edges_gen['pts']) == 18
        assert str(edges_gen.keys()) == "dict_keys(['pts', 'weight'])"

        print(f"\n=========nodes=========")
        prnt = True
        nprint = 3  # print first nprint
        for i, (s, e) in enumerate(graph.edges()):
            edge = graph[s][e]
            if i < nprint and prnt:
                print(f"{s} {e} {edge.keys()}")
            ps = edge[AmiEdge.PTS]
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
        if interactive:
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

        node1ps = nx_graph.nodes[1][AmiNode.PTS]
        # print(f"node1ps {node1ps}")
        node1ps0 = node1ps[0]
        # print("node1ps0", node1ps0)
        assert str(node1ps) == "[[ 83 680]]"

        assert str(nx_graph.edges[(1, 2)][AmiEdge.PTS]) == "[[ 83, 680]]"

    def test_segmented_edges(self):
        """
        analyse 4 arrows and convert to lines


        Note:
        nx_graph[i][j]["pts"] seens to be the same as nx_graph[j][i]["pts"]
        Note: double backslash is an escape, not meaningful
        :return:
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)

        """
        {0, 1, 2, 3, 4, 5, 6, 7},  # double arrow
            0         6
          /            \\
    3----2-------------4----5
          \\            /
           7          1
         [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6),
        """
        print("\n2 0", nx_graph[2][0]["pts"][:2:-2])
        print("\n2 7", nx_graph[2][7])
        print("\n2 3", nx_graph[2][3])

        print("\n2 4", nx_graph[2][4])

        print("\n4 5", nx_graph[4][5])
        print("\n4 6", nx_graph[4][6])
        print("\n1 4", nx_graph[1][4])

        points0_2 = nx_graph[2][0]["pts"]

        """
        {8, 9, 26, 19},            # y-shaped arrow-less
         (8, 19), (9, 19), (19, 26),
        8        26
        |         |
        ----19-----
             |
            26
        """
        """
        {10, 11, 12, 13, 14, 15, 16, 17, 18, 20}, # bifurcated arrow
         (10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15), (16, 18), (17, 18), (18, 20),
        """
        """
        {21, 22, 23, 24, 25}]      # simple arrow
        (21, 24), (22, 24), (23, 24), (24, 25)]
                  
                  21
                  |
                  |
                  |
          22      |      23
             \\    |    /
                \\ |  /
                  24
                  |
                  |
                  25
        """
        # print("21 24", nx_graph[21][24])
        # print("24 25", nx_graph[24][25])
        # print("22 24", nx_graph[22][24])
        # print("23 24", nx_graph[23][24])

        return

    def test_arrows(self):
        """
        looks for arrowheads, three types
        * point with shaft and two edges going "backwards" symmetrically
        * point with shaft, 2 edges backward and short one forward (result of thinning filled triangle)
        * half arrow. point with one edge backwards (e.g. in chemical equilibrium
        :return:
        """

    def test_islands(self):
        """
        Create island_node_id_sets using sknw/NetworkX and check basic properties
        :return:
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)

        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        assert nx.algorithms.components.number_connected_components(nx_graph) == 4
        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        assert type(connected_components) is list, f"type of connected components should be list"
        assert connected_components == [
            {0, 1, 2, 3, 4, 5, 6, 7},
            {8, 9, 26, 19},
            {10, 11, 12, 13, 14, 15, 16, 17, 18, 20},
            {21, 22, 23, 24, 25}
        ]

        assert type(connected_components[0]) is set and len(connected_components[0]) == 8, \
            f"components should be sets and first len == 8"
        assert type(list(connected_components[0])[0]) is int, f"members should be int"

        ami_graph = AmiGraph(nx_graph)
        ami_graph.read_nx_graph(nx_graph)
        island_node_id_sets = ami_graph.get_or_create_ami_islands()
        assert len(island_node_id_sets) == 4
        assert type(island_node_id_sets[0]) is AmiIsland
        assert island_node_id_sets[0].node_ids == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_nodes(self):
        """
        Tests
        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_file(Resources.BIOSYNTH1_ARROWS)
        nodex = AmiNode(nx_graph=ami_graph.nx_graph, node_id=(list(ami_graph.nx_graph.nodes)[0]))
        node_id = 0
        nodex = AmiNode(ami_graph=ami_graph, node_id=node_id)

        xy = nodex.get_or_create_centroid_xy()
        assert xy == [844.0, 82.0]

    def test_edges(self):
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

    def test_line_segments(self):
        """
        split edges into segments (Douglas-Paucker) - Python tutorial
        :return:
        """

        hand = np.array([[1.64516129, 1.16145833],
                         [1.64516129, 1.59375],
                         [1.35080645, 1.921875],
                         [1.375, 2.18229167],
                         [1.68548387, 1.9375],
                         [1.60887097, 2.55208333],
                         [1.68548387, 2.69791667],
                         [1.76209677, 2.56770833],
                         [1.83064516, 1.97395833],
                         [1.89516129, 2.75],
                         [1.9516129, 2.84895833],
                         [2.01209677, 2.76041667],
                         [1.99193548, 1.99479167],
                         [2.11290323, 2.63020833],
                         [2.2016129, 2.734375],
                         [2.25403226, 2.60416667],
                         [2.14919355, 1.953125],
                         [2.30645161, 2.36979167],
                         [2.39112903, 2.36979167],
                         [2.41532258, 2.1875],
                         [2.1733871, 1.703125],
                         [2.07782258, 1.16666667]])

        # subdivide polygon using 2nd degree B-Splines (green)
        new_hand = hand.copy()
        ncycle = 5  # doubles the number of points/splines each cycle
        for _ in range(ncycle):
            new_hand = subdivide_polygon(new_hand, degree=2, preserve_ends=True)

        # approximate subdivided polygon with Douglas-Peucker algorithm (orange line)
        tolerance = 0.02
        appr_hand = approximate_polygon(new_hand, tolerance=tolerance)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))

        ax1.plot(hand[:, 0], hand[:, 1])
        ax1.plot(appr_hand[:, 0], appr_hand[:, 1])

        points = np.array([
            [1., 1.],
            [1.1, 2.],
            [0.9, 3.],
            [2., 2.9],
            [3., 3.],

        ])
        ax2.plot(points[:, 0], points[:, 1])
        tolerance = 0.02
        tolerance = 0.5
        points2 = approximate_polygon(points, tolerance=tolerance)
        ax2.plot(points2[:, 0], points2[:, 1])
        if interactive:
            plt.show()

    @unittest.skipUnless(interactive, "ignorte plotting in routine tests")
    def test_plot_line(self):
        """straightens lines by Douglas Peucker and plots"""
        nx_graph = self.nx_graph_arrows1
        tolerance = 2
        lines = [
            [(21, 24), (22, 24), (23, 24), (24, 25)],
            [(10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15), (16, 18), (17, 18), (18, 20)],
            [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6)],
            [(8, 19), (9, 19), (19, 26)]
            ]
        AmiGraph.plot_all_lines(nx_graph, lines, tolerance)

    def test_plot_lines_with_nodes(self):
        """adds nodes straightens lines by Douglas Peucker and plots"""
        nx_graph = self.nx_graph_arrows1
        tolerance = 2
        lines = [
            [(21, 24), (22, 24), (23, 24), (24, 25)],
            [(10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15), (16, 18), (17, 18), (18, 20)],
            [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6)],
            [(8, 19), (9, 19), (19, 26)]
            ]
        nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 26, 19, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25}
        # TODO make this tidying routine universal
        if interactive:
            # TODO split into line segmentattion and plotting
            logger.warning("skipping line segmentation test")
            AmiGraph.plot_all_lines(nx_graph, lines, tolerance, nodes=nodes)

    def test_prisma(self):
        """extract primitives from partial prisma diagram"""
        assert Resources.PRISMA.exists()
        ami_graph = AmiGraph(self.nx_graph_prisma)
        islands = ami_graph.get_or_create_ami_islands()
        assert len(islands) == 349
        big_islands = []
        for island in islands:
            bbox = island.get_or_create_bbox()
            if bbox.get_height() > 40:
                big_islands.append(island)
        assert len(big_islands) == 10

    def test_extract_raw_image(self):
        """extract the raw pixels (not the skeletonm) underlying the extracted lines
        """
        nx_graph = self.nx_graph_arrows1
        islands = AmiGraph(nx_graph).get_ami_islands_from_nx_graph()
        arrows1 = self.arrows1
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(9, 6))

        ax1.set_title("raw image")
        ax1.imshow(arrows1)

        arrows_inv = np.invert(arrows1)
        AmiGraph.plot_axis(arrows_inv, ax2, islands, title="inverted")

        arrows_dil = morphology.dilation(arrows_inv, morphology.disk(3))
        AmiGraph.plot_axis(arrows_dil, ax3, islands, title="dilated")

        arrows_erod = morphology.erosion(arrows_inv, morphology.disk(3))
        AmiGraph.plot_axis(arrows_erod, ax4, islands, title="eroded")

        fig.tight_layout()
        plt.show()


