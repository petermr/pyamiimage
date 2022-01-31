"""
tests AmiGraph, AmiNode, AmiEdge, AmiIsland
"""

import logging
import unittest
from collections import Counter
# library
from pathlib import PurePath

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import sknw
from skimage import data
from skimage import morphology
from skimage.measure import approximate_polygon, subdivide_polygon
from skimage.morphology import skeletonize

# local
# from ..pyimage.ami_edge_manager import AmiEdgeAnalyzer, X, Y
from ..pyimage.ami_graph_all import AmiNode, AmiIsland, AmiGraph, AmiEdge
from ..pyimage.ami_image import AmiImage
from ..pyimage.ami_plot import AmiEdgeTool, AmiLine
from ..pyimage.ami_util import AmiUtil
from ..pyimage.bbox import BBox
from ..pyimage.text_box import TextBox, TextUtil
from ..test.resources import Resources

logger = logging.getLogger(__name__)

interactive = False


class TestAmiGraph:

    def setup_class(self):
        """
        resources are created once only in self.resources.create_ami_graph_objects()
        Make sure you don't corrupt them
        we may need to add a copy() method
        :return:
        """
        self.resources = Resources()
        self.resources.create_ami_graph_objects()

    def setup_method(self, method):

        self.arrows1 = self.resources.arrows1_image
        self.nx_graph_arrows1 = self.resources.nx_graph_arrows1

        self.biosynth1_binary = self.resources.biosynth1_binary
        self.biosynth1_elem = self.resources.biosynth1_elem

        self.nx_graph_biosynth3 = self.resources.biosynth3_dto.nx_graph

        self.nx_graph_prisma = self.resources.nx_graph_prisma

        self.battery1_image = self.resources.battery1_image
        self.nx_graph_battery1 = self.resources.nx_graph_battery1

        return self

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

        assert str(graph.nodes[0].keys()) == "dict_keys([AmiEdge.PTS, 'o'])", \
            "nodes have 'pts' and 'o "
        # this is a 2-array with ["pts", "weights'] Not sure what th structure is
        # this is an edges generator, so may need wrapping in a list
        edges_gen = graph[0][2]  # plural because there may be multiple edges between nodes
        assert f"{list(edges_gen)[:1]}" == "[AmiEdge.PTS]"  # pts are connected points in an edge
        assert len(edges_gen[AmiEdge.PTS]) == 18
        assert str(edges_gen.keys()) == "dict_keys([AmiEdge.PTS, 'weight'])"

        print(f"\n=========nodes=========")
        prnt = True
        nprint = 3  # print first nprint
        for i, (s, e) in enumerate(graph.edges()):
            edge = graph[s][e]
            if i < nprint and prnt:
                print(f"{s} {e} {edge.keys()}")
            points = edge[AmiEdge.PTS]
            plt.plot(points[:, 1], points[:, 0], 'green')

        # draw node by o
        print(f"=========neighbours=========")
        nodes = graph.nodes()
        # print neighbours
        for node in graph.nodes():
            print(f"neighbours {node} {list(graph.neighbors(node))}")

        # coordinates are arranged y-array, x-array
        points = np.array([nodes[i][AmiNode.CENTROID] for i in nodes])
        plt.plot(points[:, 1], points[:, 0], 'r.')

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

    # @unittest.skip("exploration")
    def test_sknw_5_islands(self):
        """
        This checks all the fields that sknw returns
        also explares many primitives in nx.graph

        :return:
        """
        skel_path = Resources.BIOSYNTH1_ARROWS
        assert isinstance(skel_path, PurePath)

        skeleton_array = AmiImage.create_white_skeleton_from_file(skel_path)
        AmiUtil.check_type_and_existence(skeleton_array, np.ndarray)

        # nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        AmiUtil.check_type_and_existence(nx_graph, nx.classes.multigraph.MultiGraph)

        assert list(nx_graph.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                        18, 19, 20, 21, 22, 23, 24, 25, 26]
        # this fails because it's a NodeView
        # assert nx_graph.nodes() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        #                                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        print(f"nodes {type(nx_graph.nodes)} {nx_graph.nodes()}")

        # print (f"edges {nx_graph.edges()}")
        assert list(nx_graph.edges) == [
            (0, 2, 0), (1, 4, 0), (2, 4, 0), (2, 3, 0), (2, 7, 0), (4, 5, 0), (4, 6, 0), (8, 19, 0),
            (9, 19, 0), (10, 12, 0), (11, 13, 0), (12, 13, 0), (12, 18, 0), (13, 14, 0), (13, 15, 0),
            (16, 18, 0), (17, 18, 0), (18, 20, 0), (19, 26, 0), (21, 24, 0), (22, 24, 0), (23, 24, 0),
            (24, 25, 0)], f"edges {list(nx_graph.edges)}"

        # np.testing.assert_array_equal(x, y, msg)
        # AmiUtil.check_type_and_existence(nx_graph.nodes, nx.classes.reportviews.NodeView)
        #
        # AmiUtil.check_type_and_existence(nx_graph.edges, nx.classes.reportviews.MultiEdgeView)

        node1ps = nx_graph.nodes[1][AmiNode.CENTROID]
        assert str(node1ps) == "[ 83 680]"

        edge_yx = nx_graph.edges[(0, 2, 0)][AmiEdge.PTS]
        edge_xy = np.flip(edge_yx, 1)
        expected = np.array(
            [[844, 82], [845, 83], [846, 84], [847, 84], [848, 85], [849, 85], [850, 86], [851, 86],
             [852, 87], [853, 87], [854, 87], [855, 88], [856, 89], [857, 91]], dtype=np.int16)
        # print(f"expected {type(expected)} {expected}")
        # can't get this to work
        # assert np.testing.assert_array_equal(edge_xy, expected) #, err_msg=f"found {edge_xy}")
        # but this does
        assert np.array_equal(edge_xy, expected), f"found {edge_xy}"

    def test_ami_edges(self):
        """wrappers for nx_graph
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        print("------------------")
        ami_edges = ami_graph.get_or_create_all_ami_edges()
        print("==================")
        assert len(ami_edges) == 23, f"found {len(ami_edges)}"
        print(f"len edges {len(ami_edges)}")
        # print (f"ami_edge 0 {ami_edges[0]}")

    # @unittest.skip("needs multigraph adding NYI")
    def test_segmented_edges(self):
        """
        analyse 4 arrows and convert to lines


        Note:
        nx_graph[i][j]["pts"] seens to be the same as nx_graph[j][i]["pts"]
        NO! there are 2 edges in opposite directions
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

        print("\n2 0", nx_graph[2][0][0][AmiEdge.PTS][:2:-2])

        points0_2 = nx_graph[2][0][0][AmiEdge.PTS]
        expected_numpy = np.array([[82, 844],
                                   [83, 845],
                                   [84, 846],
                                   [84, 847],
                                   [85, 848],
                                   [85, 849],
                                   [86, 850],
                                   [86, 851],
                                   [87, 852],
                                   [87, 853],
                                   [87, 854],
                                   [88, 855],
                                   [89, 856],
                                   [91, 857]])
        assert type(expected_numpy[0]) is np.ndarray
        assert type(expected_numpy[0][0]) is np.int32
        assert np.array_equal(points0_2, expected_numpy), f"found {points0_2}"

        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        ami_edges = ami_graph.get_or_create_all_ami_edges()
        print(f"len(edges) {len(ami_edges)}")
        for ami_edge in ami_edges:
            for tol in range(1, 10, 3):
                segments = ami_edge.create_line_segments(tolerance=tol)
                print(f"tol {tol} edge {ami_edge.start_id, ami_edge.end_id} segments {len(segments)} -> {segments}")

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

        return

    def test_arrows(self):
        """
        looks for arrowheads, three types
        * point with shaft and two edges going "backwards" symmetrically
        * point with shaft, 2 edges backward and short one forward (result of thinning filled triangle)
        * half arrow. point with one edge backwards (e.g. in chemical equilibrium
        :return:
        """
        AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)

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

    def test_node_centroid(self):
        """
        Tests
        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        xy = ami_graph.get_or_create_centroid_xy(0)
        assert xy == [844.0, 82.0]

    def test_get_nx_edge_list_for_node(self):
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        edges = ami_graph.get_nx_edge_list_for_node(24)
        assert edges == [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)], \
            f"found {edges} expected {[(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)]}"

    def test_get_nx_edge_lengths_for_node(self):
        ami_graph = self.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        lengths = ami_graph.get_nx_edge_lengths_by_edge_list_for_node(24)
        print(f"lengths {lengths}")
        lengthsx = [length for length in lengths.values()]
        assert [0.1 + 0.2, 0.2 + 0.4] == pytest.approx([0.3, 0.6])
        aaa = [30.00, 8.944, 9.848, 12.041]
        expect = pytest.approx(aaa, 0.001)
        assert lengthsx == expect, \
            f"found {lengthsx} expected {expect}"

    def test_get_nodes_with_degree(self):
        """
        uses get_nodes_with_degree on each node to create lists
        :return:
        """
        ami_graph = self.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        self.assert_degrees(ami_graph, 4, [2, 4, 13, 18, 24])
        self.assert_degrees(ami_graph, 3, [12, 19])
        self.assert_degrees(ami_graph, 2, [])
        self.assert_degrees(ami_graph, 1, [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 25, 26])

    def test_distal_node(self):
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        edge = ami_graph.get_nx_edge_list_for_node(1)[0]
        ami_edge0 = ami_graph.get_or_create_ami_edge_from_ids(edge[0], edge[1], edge[2])
        assert ami_edge0.start_id == 1
        assert ami_edge0.remote_node_id(1) == 4
        assert ami_edge0.remote_node_id(4) == 1
        assert ami_edge0.remote_node_id(None) is None
        assert ami_edge0.remote_node_id(3) is None

    def test_get_neighbours(self):
        ami_graph = self.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        assert [2] == ami_graph.get_or_create_ami_node(0).get_neighbour_ids()
        assert [4] == ami_graph.get_or_create_ami_node(1).get_neighbour_ids()
        assert [0, 4, 3, 7] == ami_graph.get_or_create_ami_node(2).get_neighbour_ids()
        assert [10, 13, 18] == ami_graph.get_or_create_ami_node(12).get_neighbour_ids()

    def create_ami_graph_from_arbitrary_image_file(self, path):
        """
        creates nx_graph and wraps it in AmiGraph object
        :return: AmiGraph
        """
        nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(path)
        ami_graph = AmiGraph(nx_graph=nx_graph)
        return ami_graph

    @unittest.skip("NYI")
    def test_get_angles_of_edges_node(self):
        ami_graph = self.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        edges = ami_graph.get_nx_edge_list_for_node(24)
        angles = AmiNode.calculate_angles_to_edges(edges)

    def test_bboxes(self):
        """
        Create bounding boxes for islands
        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
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
        interactive = False
        ignore_hand = True
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))

        if not ignore_hand:
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
        tolerance = 0.5
        points2 = approximate_polygon(points, tolerance=tolerance)
        ax2.plot(points2[:, 0], points2[:, 1])
        if interactive:
            plt.show()

    # @unittest.skip("not needed as skimage works")
    # def test_rdp_line_segments(self):
    #     """rdp method may interface better than the approximate_polygon one"""
    #     arr = np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2)
    #     mask = rdp(arr, algo="iter", return_mask=True)
    #     print(f"mask {mask}")
    #     print(f"arr {arr[mask]}")
    #     ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
    #     islands = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)
    #     assert len(islands) == 1, f"expected single islands {len(islands)}"
    #     small_plot = islands[0]
    #     node_ids = small_plot.node_ids
    #     assert len(node_ids) == 36, f"nodes in small graph {len(node_ids)}"
    #     assert node_ids == {258, 132, 389, 136, 24, 288, 546, 36, 40, 424, 46, 177, 569, 63, 448, 193,
    #                         323, 590, 591, 592, 593, 594, 595, 596, 86, 605, 606, 607, 608, 609, 354,
    #                         101, 103, 487, 114, 510}, f"found {node_ids}"
    #     assert small_plot.nx_edges == [
    #         (258, 323, 0), (132, 177, 0), (389, 448, 0), (136, 177, 0), (136, 193, 0), (24, 36, 0), (24, 36, 1),
    #         (24, 40, 0),
    #         (288, 323, 0), (288, 354, 0), (546, 569, 0), (546, 595, 0), (36, 46, 0), (40, 46, 0), (40, 63, 0),
    #         (424, 448, 0),
    #         (424, 487, 0), (46, 86, 0), (177, 258, 0), (569, 590, 0), (63, 86, 0), (63, 101, 0), (448, 510, 0),
    #         (193, 258, 0),
    #         (193, 288, 0), (323, 389, 0), (590, 591, 0), (590, 608, 0), (591, 592, 0), (591, 605, 0), (592, 593, 0),
    #         (592, 606, 0),
    #         (593, 594, 0), (593, 596, 0), (594, 595, 0), (594, 607, 0), (595, 609, 0), (86, 103, 0), (354, 389, 0),
    #         (354, 424, 0),
    #         (101, 103, 0), (101, 114, 0), (103, 132, 0), (487, 510, 0), (487, 546, 0), (114, 132, 0), (114, 136, 0),
    #         (510, 569, 0)], \
    #         f"found {small_plot.nx_edges}"
    #
    #     ami_edges = small_plot.get_or_create_ami_edges()
    #     print(f"ami_edges {ami_edges}")
    #     ami_edges = AmiEdge.get_ami_edge_start_end

    @unittest.skipUnless(interactive, "ignore plotting in routine tests")
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
        AmiEdge.plot_all_lines(nx_graph, lines, tolerance)

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
            AmiEdge.plot_all_lines(nx_graph, lines, tolerance, nodes=nodes)

    def test_prisma(self):
        """extract primitives from partial prisma diagram"""
        assert Resources.PRISMA.exists()
        ami_graph = AmiGraph(self.nx_graph_prisma)
        islands = ami_graph.get_or_create_ami_islands()
        assert len(islands) == 382
        big_islands = []
        for island in islands:
            bbox = island.get_or_create_bbox()
            if bbox.get_height() > 40:
                big_islands.append(island)
        assert len(big_islands) == 6

    # def test_extract_raw_image(self):
    #     """extract the raw pixels (not the skeletonm) underlying the extracted lines
    #     plot boxes
    #     erode and dilate
    #     """
    #     TestAmiGraph.display_erode_dilate(self.arrows1, self.nx_graph_arrows1)

    def test_extract_raw_image(self):
        """extract the raw pixels (not the skeletonm) underlying the extracted lines
        plot boxes
        erode and dilate
        """
        self.display_erode_dilate(self.arrows1, self.nx_graph_arrows1, erode=True, dilate=True)

    def test_erode_battery(self):
        """extract the raw pixels (not the skeletonm) underlying the extracted lines
        plot boxes
        erode and dilate
        """
        nx_graph = self.nx_graph_battery1
        TestAmiGraph.display_erode_dilate(self.battery1_image, nx_graph)

    @unittest.skip("Moved to AmiOCR, NYI")
    def test_find_bboxes_with_text(self):
        """find text boxes and remove those with more than one character
        so the remaining lines can be analyses
        """
        text_boxes = TextBox.find_text_boxes(self.biosynth1_elem)
        assert len(text_boxes) == 38
        text_boxes1 = []
        for text_box in text_boxes:
            if TextUtil.is_text_from_tesseract(text_box.text):
                assert type(text_box) is TextBox, f"cannot add {type(text_box)} as TextBox"
                text_boxes1.append(text_box)
        assert len(text_boxes1) > 0, "require non_zero count of text_boxes"
        logger.info(f"{__name__} plotting {len(text_boxes1)} text_boxes of type {type(text_boxes1[0])}")

        fig, ax = plt.subplots()
        AmiGraph.plot_text_box_boxes(self.biosynth1_binary, ax, text_boxes1)

        fig.tight_layout()
        if interactive:
            plt.show()

    def test_get_nx_edge_lengths_list_for_node(self):
        """
        asserts lengths of edges to node
        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        edge_length_by_nx_edge = ami_graph.get_nx_edge_lengths_by_edge_list_for_node(24)
        assert {'a': 2.000001} == pytest.approx({'a': 2})
        assert {'a': 2.01} == pytest.approx({'a': 2}, 0.1)
        expected1 = pytest.approx(
            {(24, 21): 30.004166377354995, (24, 22): 9.394147114027968, (24, 23): 9.394147114027968,
             (24, 25): 12.010412149464313}, 0.001)
        assert {(24, 21): 30.00, (24, 22): 9.39, (24, 23): 9.39, (24, 25): 12.01} == expected1

    def test_battery1_elements(self):
        """
        Create island_node_id_sets using sknw/NetworkX and check basic properties
        :return:
        """
        # TODO package commands into AmiGraph
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BATTERY1, interactive=False)
        nx_graph = ami_graph.nx_graph
        assert len(nx_graph.nodes) == 647  # multi, iso, ring full
        # assert len(nx_graph.nodes) == 569

        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        # assert nx.algorithms.components.number_connected_components(nx_graph) == 212  #
        assert nx.algorithms.components.number_connected_components(nx_graph) == 290  # multyi iso ring full
        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        assert type(connected_components) is list, f"type of connected components should be list"

        assert type(connected_components[0]) is set and len(connected_components[0]) == 4, \
            f"components should be sets and first len == 8"
        assert type(list(connected_components[0])[0]) is int, f"members should be int"

        ami_graph = AmiGraph(nx_graph)
        ami_graph.read_nx_graph(nx_graph)
        island_node_id_sets = ami_graph.get_or_create_ami_islands()
        #        assert len(island_node_id_sets) == 212  # multi iso ring full
        assert len(island_node_id_sets) == 290

        assert type(island_node_id_sets[0]) is AmiIsland
        # assert island_node_id_sets[0].node_ids == {0, 9, 4, 5}
        assert island_node_id_sets[0].node_ids == {0, 10, 5, 6}

        islands = ami_graph.get_or_create_ami_islands()
        for island in islands:
            bbox = island.get_or_create_bbox()
            w = bbox.get_width()
            h = bbox.get_height()
            if h > 100 or w > 100:
                pass

        # image = io.imread(Resources.BATTERY1)

    def test_battery1square(self):
        """
        tests rings using a single square
        :return:
        """
        # TODO package commands into AmiGraph
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BATTERY1BSQUARE, interactive=False)
        nx_graph = ami_graph.nx_graph
        assert type(nx_graph) is nx.MultiGraph
        assert len(nx_graph.nodes) == 3  # multi, iso, ring full  (square has an artificial node)

        ami_graph = AmiGraph(nx_graph)
        ami_graph.read_nx_graph(nx_graph)
        island_node_id_sets = ami_graph.get_or_create_ami_islands()
        assert len(island_node_id_sets) == 2

        assert type(island_node_id_sets[0]) is AmiIsland
        assert island_node_id_sets[0].node_ids == {0, 1}

        islands = ami_graph.get_or_create_ami_islands()
        for island in islands:
            bbox = island.get_or_create_bbox()
            w = bbox.get_width()
            h = bbox.get_height()
            # print(f"{__name__}{bbox}")

        """acces edges 
        EITHER list(nx_graph.edges(0, 1))[0] (the 3rd index is for mUltigraph
        OR nx_graph[s][e][0] and then list
        I think...
        """

        print("edges", nx_graph.edges)
        print("=======   -----   =======")
        print("edges_list", list(nx_graph.edges), len(list(nx_graph.edges)))
        print("=======  xxxx  =======")
        print("edges[0]", list(nx_graph.edges)[0], type(list(nx_graph.edges)[0]), len(list(nx_graph.edges)[0]))
        print("=======  yyyy  =======")
        print("edges(0, 1)[0]", "tuple->", list(nx_graph.edges(0, 1))[0], type(list(nx_graph.edges(0, 1))[0]))  # start
        print("=======  edges  =======")
        for (s, e) in nx_graph.edges():
            nx_graph[s][e][0]["nxg"] = "foo"
            ps = nx_graph[s][e][0][AmiEdge.PTS]
            print("ps", type(ps))
            print("keys", nx_graph[s][e][0].keys(), nx_graph[s][e][0]["nxg"])

            print("points: ", len(ps), ps[:, 1], ps[:, 0])

    def test_create_ami_nodes_from_ids(self):
        """wrap node_ids in AmiNodes"""
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        node_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        ami_node_list = ami_graph.create_ami_nodes_from_ids(node_ids)
        assert 8 == len(ami_node_list)
        assert type(ami_node_list[0]) is AmiNode

    # utils ----------------

    def test_primitives(self):
        """
        analyses a figure with ~15 simple diagrams and returns their analyses
        :return:
        """
        interactive = False
        island_names = [
            "thin_zig",
            "diag_sq",
            "2ring",
            "thick_zag",
            "line",
            "chord",
            "phi",
            "cross",
            "window",
            "racket",
            "turnip",
            "bigo",
            "mido",
            "babyo",
            "rect",
            "inrect"
        ]
        colors = ["green", "blue", "purple", "cyan"]
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.PRIMITIVES, interactive=False)
        nx_graph = ami_graph.nx_graph
        assert type(nx_graph) is nx.MultiGraph
        assert len(nx_graph.nodes) == 42  # multi, iso, ring full  (square has an artificial node)
        print("\nPrimitives: ", nx_graph)
        islands = ami_graph.get_or_create_ami_islands()
        assert len(islands) == 16

        # TODO create as tests
        for i, island in enumerate(islands):
            cyc = None
            # island.get_or_create_bbox()
            print("\n", "******* island:", i, island_names[i],
                  island.get_or_create_bbox(),
                  island.node_ids)
            try:
                cyc = nx.algorithms.cycles.find_cycle(island.island_nx_graph)
                print(f"cycles {cyc}")
            except nx.exception.NetworkXNoCycle:
                # only way of trapping acyclic graph
                pass

        # draw image
        AmiImage.pre_plot_image(Resources.PRIMITIVES, erode_rad=2)

        # draw edges by pts
        ami_graph.pre_plot_edges(plt.gca())

        # draw node by o, len
        ami_graph.pre_plot_nodes(plot_ids=True)

        # plt.imshow(img, cmap='gray')
        if interactive:
            plt.show()

    def assert_degrees(self, ami_graph, degree, result_nodes):
        """
        tests degree of connectivity of nodes in graph
        uses ami_graph.get_nodes_with_degree

        :param ami_graph:
        :param degree:
        :param result_nodes:
        :return:
        """
        nodes = AmiGraph.get_node_ids_from_graph_with_degree(ami_graph.nx_graph, degree)
        assert nodes == result_nodes, f"nodes of degree {degree} should be {result_nodes}"

    # ----- edges -----

    def test_multigraph_edges(self):
        """
        To annotate multiple edges , currently with keys
        :return:
        """
        nx_graph = nx.MultiGraph()
        nx_graph.add_edge(0, 1, foo=3)
        nx_graph.add_edge(0, 2, weight=5)
        nx_graph.add_edge(0, 2, weight=10)
        nx_graph.add_edge(2, 0, weight=15)
        assert len(nx_graph.nodes) == 3
        assert list(nx_graph.edges(0)) == [(0, 1), (0, 2), (0, 2), (0, 2)]
        assert list(nx_graph.edges(1)) == [(1, 0)]
        assert list(nx_graph.edges(2)) == [(2, 0), (2, 0), (2, 0)]

        edge_list = list(nx_graph.edges.data())
        assert edge_list[0] == (0, 1, {'foo': 3})
        assert edge_list[1] == (0, 2, {'weight': 5})
        assert edge_list[2] == (0, 2, {'weight': 10})
        assert edge_list[3] == (0, 2, {'weight': 15})
        assert list(nx_graph.edges.data('weight', default=1)) == \
               [(0, 1, 1), (0, 2, 5), (0, 2, 10), (0, 2, 15)]
        assert list(nx_graph.edges.data(keys=True)) == \
               [(0, 1, 0, {'foo': 3}), (0, 2, 0, {'weight': 5}), (0, 2, 1, {'weight': 10}), (0, 2, 2, {'weight': 15})]
        edges = list(nx_graph.edges.data(keys=True))
        assert len(edges) == 4
        assert edges[0] == (0, 1, 0, {'foo': 3})
        key = 2
        assert edges[0][key] == 0
        assert edges[1][key] == 0
        assert edges[2][key] == 1
        assert edges[3][key] == 2

    def test_whole_image(self):
        assert self.nx_graph_biosynth3 is not None
        biosynth3_ami_graph = AmiGraph(nx_graph=self.nx_graph_biosynth3)
        islands = biosynth3_ami_graph.get_or_create_ami_islands()
        assert len(islands) == 436
        islands_big = [island for island in islands if island.get_or_create_bbox().min_dimension() > 20]
        assert len(islands_big) == 5

    def test_whole_image1(self):
        """
        analyse min-maximum and max-min of islands
        :return:
        """
        assert self.nx_graph_biosynth3 is not None
        biosynth3_ami_graph = AmiGraph(nx_graph=self.nx_graph_biosynth3)
        islands = biosynth3_ami_graph.get_or_create_ami_islands()
        assert len(islands) == 436
        counts_by_maxdim = {0: 436, 1: 408, 2: 408, 3: 401, 4: 396, 5: 389, 6: 376, 7: 368, 10: 368, 15: 211, 22: 19,
                            30: 6, 50: 5, 80: 1}
        for max_dim in counts_by_maxdim:
            islands_big = AmiIsland.get_islands_with_max_dimension_greater_than(max_dim, islands)
            assert len(islands_big) == counts_by_maxdim[max_dim]
        counts_by_mindim = {0: 42, 1: 51, 2: 79, 3: 102, 4: 115, 5: 149, 6: 169, 7: 182, 8: 236, 9: 323, 10: 352,
                            15: 423, 22: 432, 30: 435, 50: 435, 100: 435}
        for min_dim in counts_by_mindim:
            islands_big = AmiIsland.get_islands_with_max_min_dimension(min_dim, islands)
            assert len(islands_big) == counts_by_mindim[min_dim]

    def test_island_sizes(self):
        """uses mindim, maxdim, to filter in/out islands. etc.
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        # all islands
        islands = ami_graph.get_or_create_ami_islands()
        assert len(islands) == 227, f"expected total islands {len(islands)}"
        # two largest
        islands = ami_graph.get_or_create_ami_islands(mindim=50)
        assert len(islands) == 2, f"expected large islands {len(islands)}"
        # very small ones (periods, dots, etc.)
        islands = ami_graph.get_or_create_ami_islands(maxdim=2)
        assert len(islands) == 16, f"very small islands (periods, dots, decimals) {len(islands)}"
        # all thin , including dots
        islands = ami_graph.get_or_create_ami_islands(maxmindim=3)
        assert len(islands) == 49, f"very thin islands (sans-serif l, i, etc) {len(islands)}"
        # long thin , excluding dots
        islands = ami_graph.get_or_create_ami_islands(minmaxdim=19)
        assert len(islands) == 22, f"thin long islands (sans-serif l, i, etc) {len(islands)}"
        # long thin , excluding dots
        islands = ami_graph.get_or_create_ami_islands(minmaxdim=24)
        assert len(islands) == 9, f"very thin long islands (sans-serif l, i, etc) {len(islands)}"
        # islands with no thin dimension
        islands = ami_graph.get_or_create_ami_islands(mindim=15)
        assert len(islands) == 21, f"chunky islands (H M O etc and the 2 boxes) {len(islands)}"

    def test_graph_plots(self):
        """uses mindim, maxdim, to filter in/out islands. etc.
        then finds horizontal, vertical and other cnnectiosn between nodes
        """
        # NOTE these tests seem correct but pixel-fragile

        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        # second largest island is a boxed plot
        # all islands
        islands = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)
        assert len(islands) == 1, f"expected single islands {len(islands)}"
        small_plot = islands[0]
        node_ids = small_plot.node_ids
        assert len(node_ids) == 36, f"nodes in small graph {len(node_ids)}"
        assert node_ids == {258, 132, 389, 136, 24, 288, 546, 36, 40, 424, 46, 177, 569, 63, 448, 193,
                            323, 590, 591, 592, 593, 594, 595, 596, 86, 605, 606, 607, 608, 609, 354,
                            101, 103, 487, 114, 510}, f"found {node_ids}"
        assert small_plot.nx_edges == [
            (258, 323, 0), (132, 177, 0), (389, 448, 0), (136, 177, 0), (136, 193, 0), (24, 36, 0), (24, 36, 1),
            (24, 40, 0),
            (288, 323, 0), (288, 354, 0), (546, 569, 0), (546, 595, 0), (36, 46, 0), (40, 46, 0), (40, 63, 0),
            (424, 448, 0),
            (424, 487, 0), (46, 86, 0), (177, 258, 0), (569, 590, 0), (63, 86, 0), (63, 101, 0), (448, 510, 0),
            (193, 258, 0),
            (193, 288, 0), (323, 389, 0), (590, 591, 0), (590, 608, 0), (591, 592, 0), (591, 605, 0), (592, 593, 0),
            (592, 606, 0),
            (593, 594, 0), (593, 596, 0), (594, 595, 0), (594, 607, 0), (595, 609, 0), (86, 103, 0), (354, 389, 0),
            (354, 424, 0),
            (101, 103, 0), (101, 114, 0), (103, 132, 0), (487, 510, 0), (487, 546, 0), (114, 132, 0), (114, 136, 0),
            (510, 569, 0)], \
            f"found {small_plot.nx_edges}"

        pixel_error = 2
        # I have struggled to find the right term. "axial" suggests the actual axis
        horizontal_lines, vertical_lines, non_hv_lines = \
            ami_graph.extract_aligned_node_lists(node_ids, pixel_error)

        horiz_line_coord_pairs = str([
            [[66, 758], [107, 759]],
            [[107, 759], [149, 759]],
            [[149, 759], [191, 759]],
            [[191, 759], [232, 759]],
            [[232, 759], [274, 759]]
        ])
        assert str(horizontal_lines) == horiz_line_coord_pairs, f"horizontal lines should be {horiz_line_coord_pairs}"
        vert_line_coord_pairs = str([
            [[66, 462], [66, 520]],
            [[66, 355], [66, 408]],
            [[66, 577], [66, 632]],
            [[295, 372], [295, 427]],
            [[295, 96], [295, 151]],
            [[295, 483], [295, 542]],
            [[66, 131], [66, 185]],
            [[295, 151], [295, 206]],
            [[295, 598], [295, 655]],
            [[66, 185], [66, 241]],
            [[66, 408], [66, 462]],
            [[66, 752], [66, 758]],
            [[295, 206], [295, 267]],
            [[66, 632], [66, 692]],
            [[295, 427], [295, 483]],
            [[66, 520], [66, 577]],
            # [[66, 758], [65, 770]],
            [[107, 759], [107, 764]],
            [[149, 759], [149, 764]],
            [[191, 759], [191, 762]],
            [[232, 759], [232, 764]],
            [[274, 759], [274, 770]],
            [[66, 241], [66, 300]],
            [[295, 542], [295, 598]],
            [[295, 267], [295, 324]],
            [[66, 300], [66, 355]],
            [[295, 655], [295, 715]],
            [[295, 324], [295, 372]],
            [[66, 692], [66, 752]]
        ])
        assert str(vertical_lines) == vert_line_coord_pairs, f"vertical lines should be {vert_line_coord_pairs}"
        non_horvert_node_ids = str([
            [[295, 372], [66, 408]], [[295, 96], [66, 131]], [[295, 483], [66, 520]], [[295, 715], [66, 752]],
            [[295, 715], [274, 759]], [[295, 151], [66, 185]], [[295, 598], [66, 632]], [[295, 206], [66, 241]],
            [[295, 427], [66, 462]], [[66, 758], [65, 770]], [[295, 542], [66, 577]], [[295, 267], [66, 300]],
            [[295, 655], [66, 692]],
            [[295, 324], [66, 355]]])
        assert str(non_hv_lines) == non_horvert_node_ids, f"non-axial lines should be {non_hv_lines}"

    def test_enumerate_unique_edges(self):
        """separates 3- connected nodes into separate lines """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island_0 = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        triply_connected_ids = AmiGraph.get_node_ids_from_graph_with_degree(small_plot_island_0.island_nx_graph, 3)
        assert len(triply_connected_ids) == 30, f"found {len(triply_connected_ids)}"
        unique_ami_edges, multibranches = ami_graph.get_unique_ami_edges_and_multibranches(triply_connected_ids)
        assert len(unique_ami_edges) == 48, f"should be {len(unique_ami_edges)}"
        assert len(multibranches) == 2, f"should be {len(multibranches)} multibranches"

    def test_analyze_topology(self):
        """merges short horizontal and verstical lines from sknw
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        node_ids = small_plot_island.node_ids
        ami_edges, multibranches = ami_graph.get_unique_ami_edges_and_multibranches(node_ids)
        assert len(ami_edges) == 48, f"expected edge count {len(ami_edges)}"
        # ami_nodes = ami_graph.create_ami_nodes_from_ids(node_ids)

        edge_tool = AmiEdgeTool.create_tool(ami_graph, ami_edges=ami_edges)
        assert edge_tool is not None
        assert len(edge_tool.ami_nodes) == 36, f"expected {len(edge_tool.ami_nodes)} nodes in edge_tool"
        new_ami_nodes, new_ami_edges = edge_tool.analyze_topology()
        assert len(new_ami_nodes) == 36 and len(new_ami_edges) == 48, \
            f"nodes {len(new_ami_nodes)} edges {len(new_ami_edges)}"

    def test_create_straight_edges(self):
        """tests straightness between nodes (horiz and vert)
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        node_ids = small_plot_island.node_ids
        pixel_error = 2
        horizontal_lines, vertical_lines, non_hv_lines = ami_graph.extract_aligned_node_lists(node_ids, pixel_error)

        for vertical_line in vertical_lines:
            print(f"vline {vertical_line}")

        edges = small_plot_island.create_nx_edges()
        for edge in edges:
            print(f"edge {edge}")

    def test_create_line_segments(self):
        """segments the edge into straight-lines (AmiLine) and finds axially aligned corners
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        ami_edges = small_plot_island.get_or_create_ami_edges()
        assert len(ami_edges) == 48, f"found {len(ami_edges)}"
        tolerance = 2

        single_lines = []
        for ami_edge in ami_edges:
            single_line = ami_edge.find_single_line(tolerance)
            if single_line is not None:
                single_lines.append(single_line)
        assert len(single_lines) == 34, f"all single lines {len(single_lines)}"

    def test_filter_line_segments(self):
        """filters segments the edge into straight-lines (AmiLine) and finds axially aligned corners
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        ami_edges = small_plot_island.get_or_create_ami_edges()

        horizontal_edges = AmiEdge.get_horizontal_edges(ami_edges, tolerance=2)
        assert len(horizontal_edges) == 5
        vertical_edges = AmiEdge.get_vertical_edges(ami_edges, tolerance=2)
        assert len(vertical_edges) == 29
        horiz_ami_lines = AmiEdge.get_single_lines(horizontal_edges)
        assert len(horiz_ami_lines) == 5
        vert_ami_lines = AmiEdge.get_single_lines(vertical_edges)
        assert len(vert_ami_lines) == 29

        # axial polylines
        tolerance = 2
        axial_polylines = AmiEdge.get_axial_polylines(ami_edges, tolerance=tolerance)
        assert len(axial_polylines) == 2, f"found {len(axial_polylines)}"
        assert type(axial_polylines) is list
        assert type(axial_polylines[0]) is list
        assert type(axial_polylines[0][0]) is AmiLine
        assert len(axial_polylines[0]) == 3
        assert len(axial_polylines[1]) == 2
        # I don't like the str(...) but how to compare lists of coords? probably need a polyline class
        assert str(axial_polylines[0][0]) == str([[295, 96], [294, 61]])
        assert str(axial_polylines[0]) == str([[[295, 96], [294, 61]], [[294, 61], [66, 61]], [[66, 61], [66, 131]]])

        for axial_polyline in axial_polylines:
            for ami_line in axial_polyline:
                if ami_line.is_vertical(tolerance=tolerance):
                    vert_ami_lines.append(ami_line)
                elif ami_line.is_horizontal(tolerance=tolerance):
                    horiz_ami_lines.append(ami_line)
                else:
                    raise ValueError(f"line {ami_line} must be horizontal or vertical")

        vert_dict = AmiLine.get_horiz_vert_counter(vert_ami_lines, xy_index=0)
        assert vert_dict == Counter({66: 13, 295: 12, 65: 1, 107: 1, 149: 1, 191: 1, 232: 1, 274: 1, 294: 1}), \
            f"found {vert_dict}"
        horiz_dict = AmiLine.get_horiz_vert_counter(horiz_ami_lines, xy_index=1)
        assert horiz_dict == Counter({759: 4, 758: 2, 61: 1}), f"found {horiz_dict}"

    #TODO upload ami_edge_analyzer
    @unittest.skip("AmiEdgeAnalyzer not uploaded yet")
    def test_edge_manager(self):
        """

        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        ami_edges = small_plot_island.get_or_create_ami_edges()
        assert len(ami_edges) == 48, f"found {len(ami_edges)}"

        edge_manager = AmiEdgeAnalyzer(tolerance=2)
        edge_manager.read_edges(ami_edges)
        assert edge_manager.vert_dict == Counter({66: 13, 295: 12, 65: 1, 107: 1, 149: 1, 191: 1, 232: 1, 274: 1, 294: 1}), \
            f"found {edge_manager.vert_dict}"
        assert edge_manager.horiz_dict == Counter({759: 4, 758: 2, 61: 1}), f"found {edge_manager.horiz_dict}"

        counts_by_xcoord, counts_by_ycoord = edge_manager.merge_neighbouring_coords()
        assert counts_by_xcoord == [[66, 14], [295, 13], [107, 1], [149, 1], [191, 1], [232, 1], [274, 1]]
        assert counts_by_ycoord == [[759, 6], [61, 1]]

    @unittest.skip("Not yet implemented")
    def test_join_horiz_vert_lines(self):
        """

        :return:
        """
        ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.YW5003_5)
        small_plot_island = ami_graph.get_or_create_ami_islands(mindim=50, maxmindim=300)[0]
        ami_edges = small_plot_island.get_or_create_ami_edges()

        edge_manager = AmiEdgeAnalyzer(tolerance=2)
        edge_manager.read_edges(ami_edges)
        counts_by_xcoord, counts_by_ycoord = edge_manager.merge_neighbouring_coords()
        assert counts_by_xcoord == [[66, 14], [295, 13], [107, 1], [149, 1], [191, 1], [232, 1], [274, 1]]
        assert counts_by_ycoord == [[759, 6], [61, 1]]
        new_vert_lines = edge_manager.join_ami_lines(X)
        print(f"new vert {new_vert_lines}")


    # =====================================
    # test helpers
    # =====================================

    @classmethod
    def display_erode_dilate(cls, image, nx_graph, radius=3, erode=False, dilate=False):
        islands = AmiGraph(nx_graph).get_ami_islands_from_nx_graph()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(9, 6))
        ax1.set_title("raw image")
        ax1.imshow(image)
        image_inv = np.invert(image)
        AmiGraph.plot_axis(image_inv, ax2, islands, title="inverted")
        if dilate:
            image_dilate = morphology.dilation(image_inv, morphology.disk(radius))
            AmiGraph.plot_axis(image_dilate, ax3, islands, title="dilated")
        if erode:
            image_erode = morphology.erosion(image_inv, morphology.disk(radius))
            AmiGraph.plot_axis(image_erode, ax4, islands, title="eroded")
        fig.tight_layout()
        if interactive:
            plt.show()
