from skimage import io

from test.resources import Resources
from pyimage.tesseract_hocr import TesseractOCR
from pyimage.text_box import TextBox

class TestTextBox:
    def setup_method(self, method):
        self.cropped1_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH1_CROPPED)
        self.cropped1_elem = TesseractOCR.parse_hocr_string(self.cropped1_hocr)
        self.biosynth3_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH3)
        self.biosynth3_elem = TesseractOCR.parse_hocr_string(self.biosynth3_hocr)

    def test_extract_phrases_boxes0(self):
        phrases, bboxes = TesseractOCR.find_phrases(self.cropped1_elem)
        assert len(bboxes) == 12
        assert phrases[0] == "Isomerase (?)"

    def test_extract_text_bboxes(self):
        text_boxes = TextBox.find_text_boxes(self.cropped1_elem)
        assert len(text_boxes) == 12
        assert text_boxes[0].text == "Isomerase (?)"
        assert text_boxes[0].bbox == [[684, 843], [38, 65]]
        assert text_boxes[4].text == "Dimethylallyl diphosphate"
        assert text_boxes[4].bbox == [[895, 1214], [70, 98]]
        assert text_boxes[10].text == "GPP synthase"
        assert text_boxes[10].bbox == [[568, 732], [281, 308]]



