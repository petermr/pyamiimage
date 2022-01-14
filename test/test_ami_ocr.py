from ..pyimage.ami_ocr import AmiOCR
from ..test.resources import Resources

class TestAmiOCR:
    def setup_method(self, method):
        self.img_ocr = AmiOCR()
        self.biosynth2 = Resources.BIOSYNTH2
        self.img_ocr.set_image_path(self.biosynth2)
        self.img_ocr.run_ocr()

    def teardown_method(self, method):
        img_ocr = None
        self.biosynth2 = None
        self.biosynth2_hocr = None
        self.biosynth2_elem = None

    def test_parse_hocr_string(self):
        assert self.img_ocr.hocr is not None

    def test_parse_hocr_tree(self):
        self.img_ocr.parse_hocr_tree()
        assert len(self.img_ocr.textboxes) == 79

    def test_find_phrases(self):

