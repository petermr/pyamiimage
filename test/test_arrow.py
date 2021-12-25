
import matplotlib.pyplot as plt
import pytest

from ..pyimage.ami_graph_all import AmiGraph

# local
from ..test.resources import Resources
from ..test.test_ami_graph import TestAmiGraph


class TestArrow:
    def setup_method(self, method):
        self.ami_graph1 = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        self.islands1 = self.ami_graph1.get_or_create_ami_islands()
        assert 4 == len(self.islands1)
        self.double_arrow_island = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads_island = self.islands1[2]
        self.one_head_island = self.islands1[3]
        assert [21, 22, 23, 24, 25] == list(self.one_head_island.node_ids)
        assert self.one_head_island.ami_graph == self.ami_graph1
        assert self.one_head_island.island_nx_graph is not None

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
        # neighbours = self.one_head_island.get_neighbour_list(24)
        node_id = 24
        nx_edges = self.ami_graph1.get_nx_edge_list_for_node(node_id)
        assert [(24, 21), (24, 22), (24, 23), (24, 25)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        edge_length_dict = self.ami_graph1.get_nx_edge_lengths_by_edge_list_for_node(node_id)
        edge_lengths = [v for v in edge_length_dict.values()]
        assert pytest.approx(edge_lengths, rel=0.001) == [30.00, 8.944, 9.848, 12.041]

    def test_get_interedge_angles(self):
        node_id = 24
        interactive = False
        nx_edges = self.ami_graph1.get_nx_edge_list_for_node(node_id)
        if interactive:
            self.ami_graph1.pre_plot_edges(plt.gca())
            self.ami_graph1.pre_plot_nodes(plot_ids=True)
            plt.show()

        assert [(24, 21), (24, 22), (24, 23), (24, 25)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        angles = []

        for edge0 in nx_edges:
            for edge1 in nx_edges:
                # only do upper triangle
                if (edge0 is not edge1) and edge0[1] < edge1[1]:
                    angle = self.ami_graph1.get_interedge_angle(edge0, edge1)
                    angles.append(angle)
        expected = [-1.107, 1.152, 3.058, 2.259, -2.117, 1.906]

        assert expected == pytest.approx(angles, 0.001), \
            f"expected {expected} found { pytest.approx(angles, 0.001)}"

# --- helpers

    @classmethod
    def assert_arrows(cls, ami_graph, node_id_dict):
        """

        :param ami_graph: ami_graph or island
        :param node_id_dict:
        :return:
        """
        for degree in node_id_dict:
            TestAmiGraph.assert_nodes_of_degree(ami_graph, degree, node_id_dict[degree])
