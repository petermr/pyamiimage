import unittest
from ..pyimage.bbox import BBox
from skimage import io

from ..pyimage.ami_image import AmiImage
from ..pyimage.ami_ocr import TextBox, AmiOCR
from ..test.resources import Resources # Asserting all images take time

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

    def plot_bbox():
        pass

class TestAmiOCR:
    def setup_method(self, method):
        self.biosynth2 = Resources.BIOSYNTH2
        self.biosynth2_img = io.imread(self.biosynth2)
        self.biosynth2_ocr = AmiOCR(self.biosynth2)

        self.med_xrd = Resources.MED_XRD_FIG_A
        self.med_xrd_img = io.imread(self.med_xrd)

    def teardown_method(self, method):
        self.biosynth2 = None
        self.img_ocr = None

    def test_words(self):
        words = self.biosynth2_ocr.get_words()
        assert len(words) == 56, f"words are {len(words)}"
        expected_textbox = TextBox("Glycolysis", [[405, 638], [1, 57]])
        assert words[0] == expected_textbox, f"{expected_textbox} and found: {words[0]}" 
        assert words[0:5] ==[TextBox("Glycolysis", [[405, 638], [1, 57]]), 
                             TextBox("Terpene", [[182, 349], [57, 99]]),
                             TextBox("Biosynthetic", [[140, 390], [111, 145]]), 
                             TextBox("Bethea", [[178, 329], [122, 200]]),
                             TextBox("Acetyl-Co", [[606, 798], [149, 187]])], f"words and bounds are {words[:5]}"

    def test_phrases(self):
        phrases = self.biosynth2_ocr.get_phrases()
        assert len(phrases) == 46, f"phrases are {len(phrases)}"

    def test_groups(self):
        groups = self.biosynth2_ocr.get_groups()
        assert len(groups) == 46, f"groups are {len(groups)}"

    def test_clean(self):
        pass

    def test_plot_bbox_on_image(self):
        words = self.biosynth2_ocr.get_words()
        biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(self.biosynth2_img, words)
        io.imshow(biosynth2_img_bboxes)
        io.show()
    
    def test_bbox_fill(self):
        """tests filling background in a given bbox in an image"""
        box = BBox([[82, 389], [28, 386]])
        test_img = AmiOCR.set_bbox_to_bg(self.med_xrd_img, box)
        io.imshow(test_img)
        io.show()

    def test_extract_labels_from_plot(self):
        """test that labels are correctly OCRd in a plot"""
        box = BBox([[82, 389], [28, 386]])
        AmiOCR.extract_labels_from_plot(self.med_xrd_img, box)

    def test_plot_pixel_stats(self):
        """tests that plots the pixel stats in an image"""
        AmiOCR.plot_image_pixel_stats(self.med_xrd_img, 255, axis=1)

    def test_img_rotation(self):
        """tests if an image can be rotated"""
        med_xrd_img_45 = AmiOCR.image_rotate(self.med_xrd_img, 45)
        io.imshow(med_xrd_img_45)
        io.show()

    def test_rotated_image_pixel_stats(self):
        """test to check if pixel statistics work on rotated image"""
        med_xrd_img_45 = AmiOCR.image_rotate(self.med_xrd_img, 45)
        AmiOCR.plot_image_pixel_stats(med_xrd_img_45, 255, axis=1)

    def test_shapes_pixel_stats(self):
        """pixel statistics of common shapes"""
        shapes1 = Resources.SHAPES_1
        shapes1_img = io.imread(shapes1)
        shapes1_img_bin = AmiImage.create_white_binary_from_image(shapes1_img)
        io.imshow(shapes1_img)
        io.show()
        signal = AmiOCR.image_pixel_stats(shapes1_img_bin, 255, axis=1)
        from matplotlib import pyplot as plt
        import numpy as np
        row = np.arange(0, len(signal))
        plt.imshow(shapes1_img)
        plt.plot(row, signal, color='red')
        plt.show()

    