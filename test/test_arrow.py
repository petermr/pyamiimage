from skimage import io

from test.resources import Resources
from pyimage.tesseract_hocr import TesseractOCR
from pyimage.text_box import TextBox
from pyimage.ami_graph_all import AmiGraph

class TestArrow:
    def setup_method(self, method):
        ami_graph1 = AmiGraph.create_ami_graph_from_file(Resources.BIOSYNTH1_ARROWS)
        self.islands1 = ami_graph1.get_or_create_ami_islands()
        self.double_arrow = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads = self.islands1[2]
        self.one_head = self.islands1[3]

    def test_no_heads(self):
        assert len(self.no_heads.node_ids) == 4, \
            f"double arrow should have 4 nodes, found {len(self.no_heads.node_ids)}"

    def test_extract_single_arrow(self):
        assert len(self.one_head.node_ids) == 5, \
            f"single arrow should have 5 nodes, found {len(self.one_head.node_ids)}"
        print (self.one_head.ami_graph)

    def test_double_arrow(self):
        assert len(self.double_arrow.node_ids) == 8, \
            f"double arrow should have 8 nodes, found {len(self.double_arrow.node_ids)}"

    def test_branched_two_heads(self):
        assert len(self.branched_two_heads.node_ids) == 10, \
            f"double arrow should have 10 nodes, found {len(self.branched_two_heads.node_ids)}"

