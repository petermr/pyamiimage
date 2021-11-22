from pyimage import tesseract_hocr

from test.resources import Resources
from skimage import io
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from skan.pre import threshold


"""
These tests are desinged to test tesseract hocr

These tests are for Test Driven Development
"""

class TestTesseractHOCR():
        
    def setup_method(self, method):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.path = Resources.BIOSYNTH3
        self.ocr = tesseract_hocr.TesseractOCR()
        self.hocr = self.ocr.hocr_from_image_path(self.path)
        self.root = self.ocr.parse_hocr_string(self.hocr)

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        self.path = None
        self.ocr = None
        self.hocr = None
        self.root = None

    def test_basics_biosynth3(self):
        """Primarily for validating the image data which will be used elsewhere
        Uncomment for debug-like printing
        The values of assertion are specific to the image used"""

        file = Resources.BIOSYNTH3
        assert file.exists()
        image = io.imread(file)
        # io.imshow(image)
        # io.show()
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
        # print(image)
        # images are not shown in tests, I think
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')

        binary = threshold(image)
        assert binary.shape == (972, 1020)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 960392
        nblack = npix - nwhite
        print(f"npix {npix}, nwhite {nwhite} nblack {nblack} nother {npix - nwhite - nblack}")
        print(binary)

        fig, ax = plt.subplots()
        ax.imshow(binary, cmap="gray")

        binary = np.invert(binary)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 31048
        ax.imshow(binary, cmap="gray")
        plt.show()

        return

    def test_pretty_print_html(self):
        self.ocr.pretty_print_hocr(self.root)

    def test_extract_bbox_from_hocr(self):
        bbox, words = self.ocr.extract_bbox_from_hocr(self.root)
        print("Words: ", words)

    def test_find_phrases(self):
        phrases, bbox_for_phrases = self.ocr.find_phrases(self.root)
        print("Phrases:", phrases)
        print("Bounding Boxes for phrases:", bbox_for_phrases)
        assert phrases is not None
        assert len(phrases) == 29
        assert len(bbox_for_phrases) == 29


    def test_phrase_wikidata_search(self):
        path = Resources.BIOSYNTH3
        hocr = self.ocr.hocr_from_image_path(path)
        root = self.ocr.parse_hocr_string(hocr)
        phrases, bbox_for_phrases = self.ocr.find_phrases(root)
        try:
            qitems, desc = self.ocr.wikidata_lookup(phrases)
            print("qitems: ", qitems)
            print("desc: ", desc)
        except:
            print("Wikidata lookup not working")


    def test_output_phrases_to_file(self):
        sample_phrases = ["test phrase", "more test phrase", "one more"]
        file = self.ocr.output_phrases_to_file(sample_phrases, 'test_file.txt')
        phrases = []
        with open(file, 'r') as f:
            phrases = f.read().split('\n')
        phrases.pop(-1) # remove empty string associated with last \n
        assert file.exists()
        assert phrases == sample_phrases

    def test_extract_bbox_from_hocr(self):
        test_hocr_file = Path(Path(__file__).parent, 'resources/tesseract_biosynth_path_3.hocr.html')
        root = self.ocr.read_hocr_file(test_hocr_file)
        bboxes, words = self.ocr.extract_bbox_from_hocr(root)
        assert len(bboxes) == 60

    def test_extract_bbox_from_image(self):
        image_path = Path(Path(__file__).parent, 'resources/biosynth_path_3.png')
        bboxes, words = self.ocr.extract_bbox_from_image(image_path)
        assert len(bboxes) == 60
        