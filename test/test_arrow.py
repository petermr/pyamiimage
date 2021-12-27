
import matplotlib.pyplot as plt
import pytest
import logging

from ..pyimage.ami_graph_all import AmiGraph, AmiIsland
from ..pyimage.ami_arrow import AmiArrow
# local
from ..test.resources import Resources

logger = logging.getLogger(__name__)

class TestArrow:

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
        self.arrows1_ami_graph = self.resources.arrows1_ami_graph
        self.islands1 = self.arrows1_ami_graph.get_or_create_ami_islands()
        assert 4 == len(self.islands1)
        self.double_arrow_island = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads_island = self.islands1[2]
        self.one_head_island = self.islands1[3]
        assert [21, 22, 23, 24, 25] == list(self.one_head_island.node_ids)
        assert self.one_head_island.ami_graph == self.arrows1_ami_graph
        assert self.one_head_island.island_nx_graph is not None

        # complete image includes arrows and text
        # self.biosynth1_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1)
        # self.biosynth3_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH3)
        self.biosynth1_ami_graph = self.resources.biosynth1_ami_graph
        self.biosynth3_ami_graph = self.resources.biosynth3_ami_graph

    def test_extract_single_arrow(self):
        ami_graph = self.one_head_island.ami_graph
        assert len(self.one_head_island.node_ids) == 5, \
            f"single arrow should have 5 nodes, found {len(self.one_head_island.node_ids)}"
        list1 = AmiGraph.get_node_ids_from_graph_with_degree(ami_graph.nx_graph, 1)
        assert len(list1) == 20
        list2 = AmiGraph.get_node_ids_from_graph_with_degree(self.one_head_island.island_nx_graph, 1)
        assert list2 == [21, 22, 23, 25], f"{__name__} ligands found {list2} expected {[21, 22, 23, 25]}"
        longest_edge = ami_graph.find_longest_edge(24)
        assert longest_edge[0] == (24, 21)
        assert longest_edge[1] == pytest.approx(30.0)
        node0, central, other_dict = ami_graph.get_angles_round_node(24)
        for idx in other_dict:
            print(f"{node0} - {central} - {idx} = {other_dict[idx]}")

    def test_double_arrow(self):
        assert len(self.double_arrow_island.node_ids) == 8, \
            f"double arrow should have 8 nodes, found {len(self.double_arrow_island.node_ids)}"
        nodes4 = self.double_arrow_island.get_node_ids_of_degree(4)
        assert nodes4 == [2, 4], f"nodes or degree 4 should be {[2, 4]}"
        assert self.double_arrow_island.get_node_ids_of_degree(3) == []
        assert self.double_arrow_island.get_node_ids_of_degree(1) == [0, 1, 3, 5, 6, 7]

    def test_branched_two_heads(self):
        """
        one-tailed arrow that bifurcates into 2 heads
        :return:
        """
        TestArrow.assert_arrows(self.branched_two_heads_island,
                                {1: [10, 11, 14, 15, 16, 17, 20], 2: [], 3: [12], 4: [13, 18]})

    def test_no_heads(self):
        assert len(self.no_heads.node_ids) == 4, \
            f"no heads should have 4 nodes, found {len(self.no_heads.node_ids)}"
        TestArrow.assert_arrows(self.no_heads, {1: [8, 9, 26], 2: [], 3: [19], 4: []})

    def test_get_edges_and_lengths(self):
        node_id = 24
        nx_edges = self.arrows1_ami_graph.get_nx_edge_list_for_node(node_id)
        assert [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        edge_length_dict = self.arrows1_ami_graph.get_nx_edge_lengths_by_edge_list_for_node(node_id)
        edge_lengths = [v for v in edge_length_dict.values()]
        assert pytest.approx(edge_lengths, rel=0.001) == [30.00, 8.944, 9.848, 12.041]

    def test_get_interedge_angles(self):
        node_id = 24
        interactive = False
        nx_edges = self.arrows1_ami_graph.get_nx_edge_list_for_node(node_id)
        if interactive:
            self.arrows1_ami_graph.pre_plot_edges(plt.gca())
            self.arrows1_ami_graph.pre_plot_nodes(plot_ids=True)
            plt.show()

        assert [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)] == nx_edges,  \
            "edges should be [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)], found {nx_edges}"
        angles = []

        for edge0 in nx_edges:
            for edge1 in nx_edges:
                # only do upper triangle
                if (edge0 is not edge1) and edge0[1] < edge1[1]:
                    angle = self.arrows1_ami_graph.get_interedge_angle(edge0, edge1)
                    angles.append(angle)
        expected = [-1.107, 1.152, 3.058, 2.259, -2.117, 1.906]

        assert expected == pytest.approx(angles, 0.001), \
            f"expected {expected} found { pytest.approx(angles, 0.001)}"

    def test_whole_image_biosynth3(self):
        assert self.biosynth3_ami_graph is not None
        islands = self.biosynth3_ami_graph.get_or_create_ami_islands()
        assert len(islands) == 436
        big_islands = AmiIsland.get_islands_with_min_dimension(40, islands)
        assert len(big_islands) == 5

        test_arrows = [
            "tail 293 - head 384 > point 384 barbs [378, 379]",
            "tail 476 - head 592 > point 592 barbs [572, 573]",
            str(None),
            "tail 628 - head 728 > point 728 barbs [719, 720]",
            "tail 1083 - head 1192 > point 1192 barbs [1178, 1179]",
        ]
        for i, island in enumerate(big_islands):
            ami_arrow = AmiArrow.create_arrow(island)
            assert str(ami_arrow) == test_arrows[i]


    # -------------------- helpers ---------------------

    @classmethod
    def assert_arrows(cls, ami_graph, node_id_dict):
        """

        :param ami_graph: ami_graph or island
        :param node_id_dict:
        :return:
        """
        for degree in node_id_dict:
            AmiGraph.assert_nodes_of_degree(ami_graph, degree, node_id_dict[degree])
