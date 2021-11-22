"""graph classes and algorithms for managing pixel analyses"""

from ..pyimage import graph_lib
# import numpy
from pathlib import Path
from skimage import io
from ..pyimage.graph_lib import AmiGraph, AmiNode, AmiEdge, AmiGraphError
"""
These tests are designed to test the preprocessing module
containing the graph_lib module 

These tests are for Test Driven Development
"""

class TestGraphLib():

    @classmethod
    def create_from(cls, pathname):
        img = io.imread(pathname)
        return img

    # apparently doesn't like init in Test class
    # def __init__(self):
    TEST_DIR = Path(Path(__file__).parent, "resources")
    SIMPLE_2N1E = "simple_2n1e"
    SIMPLE_2N1E_PATH = Path(TEST_DIR, SIMPLE_2N1E+".png")
    GRAPH_DICT = {
    }
    EDGES1 =  [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6), (8, 19), (9, 19), (10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15), (16, 18), (17, 18), (18, 20), (19, 26), (21, 24), (22, 24), (23, 24), (24, 25)]
    NODES1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]


    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        assert self.SIMPLE_2N1E_PATH.exists(), f"{self.SIMPLE_2N1E_PATH} not found"
        png = self.create_from(self.SIMPLE_2N1E_PATH)
        assert png is not None, f"cannot read PNG {self.SIMPLE_2N1E_PATH} "
        self.GRAPH_DICT = {
            self.SIMPLE_2N1E: png,
        }

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        pass

    def test_load_create_image(self):
        img = self.GRAPH_DICT[self.SIMPLE_2N1E]
        assert img is not None
        width = 5
        height = 4
        nplanes = 3
        assert img.shape == (height, width, nplanes)

    def test_img_contents(self):
        img = self.GRAPH_DICT[self.SIMPLE_2N1E]
        # print("img", img)

    def test_read_graph_nodes(self):
        ami_graph = AmiGraph()
        ami_graph.read_nodes(self.NODES1)
        assert len(ami_graph.ami_node_dict.keys()) == 27, f"ami_graph has {len(self.NODES1)} nodes"

    def test_read_graph_edges(self):
        ami_graph = AmiGraph()
        ami_graph.read_nodes(self.NODES1)
        ami_graph.read_edges(self.EDGES1)
        print("\nnodes + edges: "+str(ami_graph))
        assert len(ami_graph.edges) == 23, f"ami_graph has {len(self.EDGES1)} edges"

    def test_read_edges_without_nodes(self):
        ami_graph = AmiGraph()
        ami_graph.read_edges(self.EDGES1)

    def test_create_islands(self):
        ami_graph = AmiGraph()
        ami_graph.read_nodes(self.NODES1)
        ami_graph.read_edges(self.EDGES1)
        print("\nnodes + edges: "+str(ami_graph))
        assert len(ami_graph.edges) == 23, f"ami_graph has {len(self.EDGES1)} edges"
        print("info", ami_graph.get_graph_info())





