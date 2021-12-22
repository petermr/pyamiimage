import pytest
import matplotlib.pyplot as plt
import math
# local
from ..test.resources import Resources
from ..pyimage.ami_graph_all import AmiGraph, AmiIsland
from ..pyimage.ami_arrow import AmiArrow




class TestArrow:
    def setup_method(self, method):
        self.ami_graph1 = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        self.islands1 = self.ami_graph1.get_or_create_ami_islands()
        assert 4 == len(self.islands1)
        self.double_arrow = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads = self.islands1[2]
        self.one_head_island = self.islands1[3]
        assert [21, 22, 23, 24, 25] == list(self.one_head_island.node_ids)
        assert self.one_head_island.ami_graph == self.ami_graph1
        assert self.one_head_island.island_nx_graph is not None


    def test_no_heads(self):
        assert len(self.no_heads.node_ids) == 4, \
            f"no heads should have 4 nodes, found {len(self.no_heads.node_ids)}"

    def test_extract_single_arrow(self):
        ami_graph = self.one_head_island.ami_graph
        assert len(self.one_head_island.node_ids) == 5, \
            f"single arrow should have 5 nodes, found {len(self.one_head_island.node_ids)}"
        list1 = AmiGraph.get_node_ids_from_graph_with_degree(ami_graph.nx_graph, 1)
        assert len(list1) == 20
        list2 = AmiGraph.get_node_ids_from_graph_with_degree(self.one_head_island.island_nx_graph, 1)
        assert list2 == [21, 22, 23, 25], f"{__name__} ligands found {list2} expected {[21, 22, 23, 25]}"
        longest_edge = ami_graph.find_longest_edge(24)
        # ami_graph.get_angles

    def test_double_arrow(self):
        assert len(self.double_arrow.node_ids) == 8, \
            f"double arrow should have 8 nodes, found {len(self.double_arrow.node_ids)}"

    def test_branched_two_heads(self):
        assert len(self.branched_two_heads.node_ids) == 10, \
            f"double arrow should have 10 nodes, found {len(self.branched_two_heads.node_ids)}"

    def test_get_edges_and_lengths(self):
        # neighbours = self.one_head_island.get_neighbour_list(24)
        node_id = 24
        nx_edges = self.ami_graph1.get_nx_edge_list_for_node(node_id)
        assert [(24, 21), (24, 22), (24, 23), (24, 25)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        edge_lengths = self.ami_graph1.get_nx_edge_lengths_by_edge_list_for_node(node_id)
        assert pytest.approx(edge_lengths, rel=0.001) == [30.0041, 9.3941, 9.39414, 12.01041]

    def test_get_interedge_angles(self):
        node_id = 24
        interactive = True
        interactive = False
        nx_edges = self.ami_graph1.get_nx_edge_list_for_node(node_id)
        if interactive:
            self.ami_graph1.pre_plot_edges(plt.gca())
            self.ami_graph1.pre_plot_nodes(plot_ids=True)
            plt.show()

        # print("nx type", type(self.ami_graph1.nx_graph))
        assert [(24, 21), (24, 22), (24, 23), (24, 25)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        angles = []

        # edge0 = (24, 21)
        # edge1 = (24, 22)
        # angle = self.ami_graph1.get_interedge_angle(edge0, edge1)
        # print(angle)
        # print(f"angle {angle}")

        for edge0 in nx_edges:
            for edge1 in nx_edges:
                # only do upper triangle
                if (edge0 is not edge1) and edge0[1] < edge1[1]:
                    angle = self.ami_graph1.get_interedge_angle(edge0, edge1)
                    angles.append(angle)
                    # print(f"edge0 {edge0} edge1 {edge1} => {angle:.3f}")
        non_multi_expected = [-1.114, 1.148, 3.116, 2.262, -2.052, 1.969]
        expected = [-1.114, 1.148, 3.116, 2.262, -2.052, 1.969]

        assert expected == pytest.approx(angles, 0.001), \
            f"expected {expected} found { pytest.approx(angles, 0.001)}"

    def test_annotate_arrows(self):
        AmiArrow.annotate_island(self.one_head_island)










