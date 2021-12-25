# local
from ..pyimage.ami_graph_all import AmiNode, AmiEdge

class AmiArrow:
    def __init__(self, ami_island=None):
        self.ami_island = ami_island
        if self.ami_island is None:
            raise ValueError(f"AmiArrow much have an island")

    @classmethod
    def annotate_graph(cls, nx_graph):
        cls.annotate_4_nodes(nx_graph)
        cls.annotate_3_nodes(nx_graph)
        cls.annotate_1_nodes(nx_graph)

    @classmethod
    def find_arrow_heads(cls, island):
        node_dict = island.create_node_degree_dict()
        for node_id in node_dict[3]:
            ami_node = AmiNode(node_id,ami_graph=island.ami_graph)
            angles = island.ami_graph.get_angles_round_node(node_id)
            print(f"{ami_node.node_id} ... {ami_node.get_neighbors()}")
            print(f"edge0, node, others {angles}")
            ami_edges = ami_node.get_or_create_ami_edges()
            for ami_edge in ami_edges:
                print(f"{__name__} edge: {ami_edge}")

            # print(f"edges to {node_id} . {edges}")


# ----------- utils -----------
