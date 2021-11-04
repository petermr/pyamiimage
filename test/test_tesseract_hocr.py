from pyimage import tesseract_hocr

from test.resources import Resources
from skimage import io
from matplotlib import pyplot as plt
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
        self.hocr = tesseract_hocr.hocr_from_image_path(self.path)
        self.root = tesseract_hocr.parse_hocr_string(self.hocr)

    def teardown_method(self, method):
        """teardown any state that was previously setup with a setup_method
        call."""
        pass

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
        tesseract_hocr.pretty_print_html(self.root)

    def test_extract_bbox_from_hocr(self):
        bbox, words = tesseract_hocr.extract_bbox_from_hocr(self.root)
        print(words)

    def test_find_phrases(self):
        phrases, bbox_for_phrases = tesseract_hocr.find_phrases(self.root)
        print(phrases)
        print(bbox_for_phrases)
        assert phrases is not None
        assert len(phrases) == 29
        assert len(bbox_for_phrases) == 29

    def test_phrase_wikidata_search(self):
        path = Resources.BIOSYNTH3
        hocr = tesseract_hocr.hocr_from_image_path(path)
        root = tesseract_hocr.parse_hocr_string(hocr)
        phrases, bbox_for_phrases = tesseract_hocr.find_phrases(root)
        qitems, desc = tesseract_hocr.wikidata_lookup(phrases)    
        print("qitems: ", qitems)
        print("desc: ", desc)    

def main():
    tester = TestTesseractHOCR()
    # tester.test_pretty_print_html()
    # tester.test_extract_bbox_from_hocr()
    # tester.test_find_phrases()
    tester.test_phrase_wikidata_search()

if __name__ == '__main__':
    main()