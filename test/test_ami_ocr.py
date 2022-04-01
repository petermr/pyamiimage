import unittest

import context
from pyamiimage.ami_ocr import TextBox, AmiOCR
from pyamiimage.bbox import BBox
from pyamiimage.ami_image import AmiImage

from resources import Resources

Interactive = False
heavy = True

class TestTextBox:
    def setup_method(self, method):
        self.textbox = TextBox('hello world', [[10, 50], [40, 50]])

    def teardown_method(self, method):
        self.textbox = None

    def test_get_text(self):
        assert self.textbox.text == 'hello world'

    def test_set_text(self):
        self.textbox.set_text("hello peter")
        assert self.textbox.text == "hello peter" 

    def test_get_ranges(self):
        assert self.textbox.bbox.get_ranges() == [[10, 50], [40, 50]]

class TestAmiOCR:
    def setup_method(self, method):
        # test image
        self.biosynth3_path = Resources.BIOSYNTH3_RAW
        self.test_image = Resources.AMIOCR_TEST_RAW

    def teardown_method(self, method):
        # clear variables
        self.biosynth3_path = None
        self.test_image = None

    def test_run_ocr_no_image(self):
        no_image_amiocr = AmiOCR()
        textboxes = no_image_amiocr.get_textboxes()
        assert len(textboxes) == 0, f'Number of textboxes found {len(textboxes)}'

    def test_run_ocr_biosynth3_image(self):
        biosynth3_image = AmiImage.read(self.biosynth3_path)
        biosynth3_amiocr = AmiOCR(biosynth3_image)
        textboxes = biosynth3_amiocr.get_textboxes()
        assert len(textboxes) == 29, f'Number of textboxes found {len(textboxes)}'

    def test_run_ocr_biosynth3_path(self):
        biosynth3_amiocr = AmiOCR(self.biosynth3_path)
        textboxes = biosynth3_amiocr.get_textboxes()
        assert len(textboxes) == 29, f'Number of textboxes found {len(textboxes)}'

    def test_run_ocr_biosynth3_path_str(self):
        biosynth3_amiocr = AmiOCR(str(self.biosynth3_path))
        textboxes = biosynth3_amiocr.get_textboxes()
        assert len(textboxes) == 29, f'Number of textboxes found {len(textboxes)}'

    def test_easyocr_from_file(self):
        test_ocr = AmiOCR(self.test_image, backend='easyocr')
        textboxes = test_ocr.get_textboxes()
        assert len(textboxes) == 55
    
    def test_easyocr_from_image(self):
        test_image = AmiImage.read(self.test_image)
        test_ocr = AmiOCR(test_image, backend='easyocr')
        textboxes = test_ocr.get_textboxes()
        assert len(textboxes) == 55

    def test_tesseract_from_file(self):
        test_ocr = AmiOCR(self.test_image, backend='tesseract')
        textboxes = test_ocr.get_textboxes()
        # this is system dependent
        assert 62 <= len(textboxes) <= 83

    def test_tesseract_from_image(self):
        test_image = AmiImage.read(self.test_image)
        test_ocr = AmiOCR(test_image, backend='tesseract')
        textboxes = test_ocr.get_textboxes()
        # this is system dependent
        assert 62 <= len(textboxes) <= 83

