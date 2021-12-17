import pytest
from ..test.resources import Resources
from ..pyimage.ami_graph_all import AmiGraph
from ..pyimage.ami_arrow import AmiArrow
from ..pyimage.ami_util import Vector2


class TestArrow:
    def setup_method(self, method):
        self.ami_graph1 = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
        self.islands1 = self.ami_graph1.get_or_create_ami_islands()
        self.double_arrow = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads = self.islands1[2]
        self.one_head_island = self.islands1[3]

    def test_no_heads(self):
        assert len(self.no_heads.node_ids) == 4, \
            f"no heads should have 4 nodes, found {len(self.no_heads.node_ids)}"

    def test_extract_single_arrow(self):
        assert len(self.one_head_island.node_ids) == 5, \
            f"single arrow should have 5 nodes, found {len(self.one_head_island.node_ids)}"
        nlist = self.one_head_island.get_lists_of_neighbour_lists(4)
        assert [[21, 22, 23, 25]] == nlist, f"list of lists found {nlist} expected {[[21, 22, 23, 25]]}"
        list1 = self.one_head_island.get_node_ids_of_degree(1)
        assert list1 == [21, 22, 23, 25], f"{__name__} ligands found {list1} expected {[21, 22, 23, 25]}"
        longest_edge = AmiArrow.find_longest_edge(24)

    def test_double_arrow(self):
        assert len(self.double_arrow.node_ids) == 8, \
            f"double arrow should have 8 nodes, found {len(self.double_arrow.node_ids)}"

    def test_branched_two_heads(self):
        assert len(self.branched_two_heads.node_ids) == 10, \
            f"double arrow should have 10 nodes, found {len(self.branched_two_heads.node_ids)}"

    def test_get_longest_edge(self):
        # neighbours = self.one_head_island.get_neighbour_list(24)
        node_id = 24
        nx_edges = AmiGraph.get_nx_edge_list_for_node(self.one_head_island.ami_graph.nx_graph, node_id)
        assert [(24, 21), (24, 22), (24, 23), (24, 25)] == nx_edges,  \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"

    def test_annotate_arrows(self):
        AmiArrow.annotate_island(self.one_head_island)










