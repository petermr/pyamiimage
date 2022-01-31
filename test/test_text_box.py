from skimage import io
import unittest
from ..test.resources import Resources
from ..pyimage.tesseract_hocr import TesseractOCR
from ..pyimage.text_box import TextBox

class TestTextBox:
    def setup_method(self, method):
        self.cropped1_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH1_CROPPED)
        self.cropped1_elem = TesseractOCR.parse_hocr_string(self.cropped1_hocr)
        self.biosynth2_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH2)
        self.biosynth2_elem = TesseractOCR.parse_hocr_string(self.biosynth2_hocr)
        self.biosynth3_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH3)
        self.biosynth3_elem = TesseractOCR.parse_hocr_string(self.biosynth3_hocr)

    def test_extract_phrases_boxes0(self):
        phrases, bboxes = TesseractOCR.find_phrases(self.cropped1_elem)
        assert len(bboxes) == 12
        assert phrases[0] == "Isomerase (?)"

    def test_extract_text_bboxes(self):
        # TODO repplace bboxes with whitespace
        text_boxes = TextBox.find_text_boxes(self.cropped1_elem)
        assert len(text_boxes) == 12
        assert type(text_boxes[0]) is TextBox
        assert text_boxes[0].text == "Isomerase (?)"
        assert text_boxes[0].bbox.xy_ranges == [[684, 843], [38, 65]]
        assert text_boxes[4].text == "Dimethylallyl diphosphate"
        assert text_boxes[4].bbox.xy_ranges == [[895, 1214], [70, 98]]
        assert text_boxes[10].text == "GPP synthase"
        assert text_boxes[10].bbox.xy_ranges == [[568, 732], [281, 308]]
        for text_box in text_boxes:
            # print(text_box.text)
            pass

    def test_extract_text_path2(self):
        text_boxes = TextBox.find_text_boxes(self.biosynth2_elem)
        assert len(text_boxes) == 67
        for text_box in text_boxes:
            # print(text_box.text, text_box.bbox)
            pass


