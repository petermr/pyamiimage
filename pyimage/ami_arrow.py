

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
    def annotate_node4(cls, node4):
        pass

    @classmethod
    def annotate_island(cls, island):

        pass

    @classmethod
    def find_longest_edge(cls, node_id):

        edges = AmiGraph.get_edges(node_id)


# ----------- utils -----------


