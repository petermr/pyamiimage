import unittest
from skimage import io
from ..pyimage.ami_ocr import TextBox, AmiOCR
from ..test.resources import Resources # Asserting all images take time

class TestTextBox:
    def setup_method(self, method):
        self.textbox = TextBox('hello world', [[10, 50], [40, 50]])

    def teardown_method(self, method):
        self.textbox = None

    def test_get_text(self):
        text = self.textbox.get_text()
        assert text == 'hello world'

    def test_set_text(self):
        self.textbox.set_text("hello peter")
        assert self.textbox.get_text() == "hello peter" 

    def test_get_w_h(self):
        assert self.textbox.get_ranges() == [[10, 50], [40, 50]]

    def plot_bbox():
        pass

class TestAmiOCR:
    def setup_method(self, method):
        self.biosynth2 = Resources.BIOSYNTH2
        self.biosynth2_img = io.imread(self.biosynth2)
        self.biosynth2_ocr = AmiOCR(self.biosynth2)

    def teardown_method(self, method):
        self.biosynth2 = None
        self.img_ocr = None

    def test_words(self):
        words = self.biosynth2_ocr.get_words()
        assert len(words) == 79, f"words are {len(words)}"
        expected_textbox = TextBox("Glycolysis", [[405, 638], [1, 57]])
        assert words[0] == expected_textbox, f"{expected_textbox} and found: {words[0]}" 
        assert words[0:5] ==[TextBox("Glycolysis", [[405, 638], [1, 57]]), 
                             TextBox("Terpene", [[182, 349], [57, 99]]), 
                             TextBox("\'", [[521, 525], [65, 93]]), 
                             TextBox("Biosynthetic", [[140, 390], [111, 145]]), 
                             TextBox(";", [[1037, 1045], [131, 137]])], f"words and bounds are {words[:5]}"

    @unittest.skip("NYI")
    def test_phrases(self):
        phrases = self.biosynth2_ocr.get_phrases()
        assert phrases is None

    @unittest.skip("NYI")
    def test_groups(self):
        groups = self.biosynth2_ocr.get_groups()

    def test_clean(self):
        pass

    def test_plot_bbox_on_image(self):
        words = self.biosynth2_ocr.get_words()
        biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(self.biosynth2_img, words)
        io.imshow(biosynth2_img_bboxes)
        io.show()

