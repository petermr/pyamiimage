# from ..pyimage import tesseract_hocr

from skimage import io
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from skan.pre import threshold
import unittest
from lxml import etree as ET
import logging
# local
from ..pyimage.tesseract_hocr import TesseractOCR
from ..test.resources import Resources

logger = logging.getLogger(__name__)

"""
Tests for tesseract_hocr.py

(to run tesseract:
tesseract <pathname> <output_root> hocr
e.g.,
tesseract test/resources/biosynth_path_1_cropped_arrows_removed.png arrows_removed hocr
creates:
arrows_removed.hocr (actually html file)
we may have to manually rename these to .html
"""

skip_long_tests = True


class TestTesseractHOCR:
    interactive = False

    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.biosynth1 = Resources.BIOSYNTH1
        self.biosynth1_hocr = TesseractOCR.hocr_from_image_path(self.biosynth1)
        self.biosynth1_elem = TesseractOCR.parse_hocr_string(self.biosynth1_hocr)

        self.biosynth2 = Resources.BIOSYNTH2
        self.biosynth2_hocr = TesseractOCR.hocr_from_image_path(self.biosynth2)
        self.biosynth2_elem = TesseractOCR.parse_hocr_string(self.biosynth2_hocr)
        
        self.biosynth3 = Resources.BIOSYNTH3
        self.biosynth3_hocr = TesseractOCR.hocr_from_image_path(self.biosynth3)
        self.biosynth3_elem = TesseractOCR.parse_hocr_string(self.biosynth3_hocr)

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        self.biosynth1 = None
        self.biosynth1_hocr = None
        self.biosynth1_elem = None

        self.biosynth3 = None
        self.biosynth3_hocr = None
        self.biosynth3_elem = None

    def test_basics_biosynth3(self):
        """Primarily for validating the image data which will be used elsewhere
        Uncomment for debug-like printing
        The values of assertion are specific to the image used"""

        assert self.biosynth3.exists()
        image = io.imread(self.biosynth3)
        if self.interactive:
            io.imshow(image)
            io.show()
        assert image.shape == (972, 1020)
        npix = image.size
        nwhite = np.sum(image == 255)
        assert nwhite == 941622
        nblack = np.sum(image == 0)
        assert nblack == 9812
        ndark = np.sum(image <= 127)
        assert ndark == 28888
        nlight = np.sum(image > 127)
        assert nlight == 962552
        print(f"\nnpix {npix}, nwhite {nwhite}, nblack {nblack}  nother {npix - nwhite - nblack}, ndark {ndark}, "
              f"nlight {nlight}")
        # images are not shown in tests, I think
        fig, ax = plt.subplots()
        if self.interactive:
            ax.imshow(image, cmap='gray')

        binary = threshold(image)
        assert binary.shape == (972, 1020)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 960392
        nblack = npix - nwhite
        # print(f"npix {npix}, nwhite {nwhite} nblack {nblack} nother {npix - nwhite - nblack}")

        fig, ax = plt.subplots()
        ax.imshow(binary, cmap="gray")

        binary = np.invert(binary)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 31048
        if self.interactive:
            ax.imshow(binary, cmap="gray")
            plt.show()

        return

    def test_pretty_print_html(self):
        TesseractOCR.pretty_print_hocr(self.biosynth3_elem)

    def test_extract_bbox_from_hocr(self):
        """

        :return:
        """
        bbox, words = TesseractOCR.extract_bbox_from_hocr(self.biosynth3_elem)
        assert len(words) == 60
        assert words[:3] == ['Straight', 'chain', 'ester']
        assert len(bbox) == 60
        assert list(bbox[0]) == [201, 45, 302, 75]

    def test_find_phrases(self):
        phrases, bboxes = TesseractOCR.find_phrases(self.biosynth3_elem)
        assert phrases is not None
        assert len(phrases) == 29
        assert len(bboxes) == 29
        assert bboxes[0] == [201, 45, 830, 68]
        assert phrases[0] == "Straight chain ester biosynthesis from fatty acids"

    def test_find_text_group_biosynth2(self):
        biosynth2_img = io.imread(self.biosynth2)
        
        word_bboxes, words = TesseractOCR.extract_bbox_from_hocr(self.biosynth2_elem)
        raw_tesseract = TesseractOCR.draw_bbox_around_words(image=biosynth2_img, bbox_coordinates=word_bboxes)
        
        io.imshow(raw_tesseract)
        io.show()

        phrases, phrases_bboxes = TesseractOCR.find_phrases(self.biosynth2_elem)
        phrases_tess = TesseractOCR.draw_bbox_around_words(image=biosynth2_img, bbox_coordinates=phrases_bboxes)
        io.imshow(phrases_tess)
        io.show()

    @unittest.skip("TesseractOCR is deprecated")
    def test_find_text_group(self):
        biosynth1_img = io.imread(self.biosynth1)
        
        word_bboxes, words = TesseractOCR.extract_bbox_from_hocr(self.biosynth1_elem)
        raw_tesseract = TesseractOCR.draw_bbox_around_words(image=biosynth1_img, bbox_coordinates=word_bboxes)
        
        # io.imshow(raw_tesseract)
        # io.show()

        phrases, phrase_bboxes = TesseractOCR.find_phrases(self.biosynth1_elem)
        groups_bboxes = TesseractOCR.find_word_groups(bbox_of_phrases=phrase_bboxes)
        grouped_text = TesseractOCR.draw_bbox_around_words(image=biosynth1_img, bbox_coordinates=groups_bboxes)

        io.imshow(grouped_text)
        io.show()
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(raw_tesseract)
        # ax[1].imshow(grouped_text)

        # plt.show()

    def test_cropped_test_group(self):
        biosynth2_img = io.imread(self.biosynth2)
        tiles, limits = TesseractOCR.split_image_into_snippets(biosynth2_img)
        assert limits is not None
        for tile in tiles:
            io.imshow(tile)
            io.show()


    @unittest.skipIf(skip_long_tests, "wikidata lookup")
    def test_phrase_wikidata_search(self):
        path = Resources.BIOSYNTH3
        hocr = TesseractOCR.hocr_from_image_path(path)
        root = TesseractOCR.parse_hocr_string(hocr)
        phrases, bbox_for_phrases = TesseractOCR.find_phrases(root)
        try:
            qitems, desc = TesseractOCR.wikidata_lookup(phrases)
        except Exception:
            logger.warning("Wikidata lookup not working")

    def test_output_phrases_to_file(self):
        sample_phrases = ["test phrase", "more test phrase", "one more"]
        file = TesseractOCR.output_phrases_to_file(sample_phrases, 'test_file.txt')
        # phrases = []
        with open(file, 'r') as f:
            phrases = f.read().split('\n')
        phrases.pop(-1)  # remove empty string associated with last \n
        assert file.exists()
        assert phrases == sample_phrases

    def test_extract_bbox_from_hocr3(self):
        test_hocr_file = Path(Path(__file__).parent, 'resources/tesseract_biosynth_path_3.hocr.html')
        root = TesseractOCR.read_hocr_file(test_hocr_file)
        bboxes, words = TesseractOCR.extract_bbox_from_hocr(root)
        assert len(bboxes) == 60

    def test_extract_bboxes_from_image(self):
        image_path = Path(Path(__file__).parent, 'resources/biosynth_path_3.png')
        bboxes, words = TesseractOCR.extract_bbox_from_image(image_path)
        assert len(bboxes) == 60
        assert str(bboxes[0]) == "[201  45 302  75]"
        assert words[0] == "Straight"

    def test_create_svg_rect_from_bbox(self):
        bbox = [[10, 20], [30, 50]]
        svg_rect = TesseractOCR.create_svg_rect_from_bbox(bbox, height=None)
        svg_str = ET.tostring(svg_rect).decode('utf-8')
        assert svg_str == '<svg:rect xmlns:svg="http://www.w3.org/2000/svg" x="10" width="10" ' \
                          'y="30" height="20" stroke-width="1.0" stroke="red" fill="none"/>'

    @unittest.skip("TesseractOCR is deprecated")
    def test_envelope(self):
        phrases, bboxes = TesseractOCR.find_phrases(self.biosynth1_elem)
        full_box = TesseractOCR.envelop(bboxes)
        biosynth1_img = io.imread(self.biosynth1)
        boxed = TesseractOCR.draw_bbox_around_words(image=biosynth1_img, bbox_coordinates=[full_box])
        io.imshow(boxed)
        io.show()
