import unittest
from skimage import io

from resources import Resources # Asserting all images take time

import context
from pyamiimage.bbox import BBox
from pyamiimage.ami_ocr import TextBox, AmiOCR
from pyamiimage.ami_image import AmiImage


Interactive = False

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
        self.biosynth2 = Resources.BIOSYNTH2_RAW
        self.biosynth2_img = io.imread(self.biosynth2)
        self.biosynth2_ocr = AmiOCR(self.biosynth2)

        self.med_xrd = Resources.MED_XRD_FIG_A_RAW
        self.med_xrd_img = io.imread(self.med_xrd)

    def teardown_method(self, method):
        self.biosynth2 = None
        self.img_ocr = None

    def test_words(self):
        words = self.biosynth2_ocr.get_words()
        # clean unbalanced quotes out of output
        words = AmiOCR.clean_all(words)
        assert len(words) == 56, f"words are {len(words)}"
        expected_textbox = TextBox("Glycolysis", [[405, 638], [1, 57]])
        assert words[0] == expected_textbox, f"{expected_textbox} and found: {words[0]}"
        # assert words[0:3] == [TextBox("Glycolysis", [[405, 638], [1, 57]]),
        #                      TextBox("Terpene", [[182, 349], [57, 99]]),
        #                      TextBox("Biosynthetic", [[140, 390], [111, 145]]),
        #                      # TextBox("Bethea", [[178, 329], [122, 200]]),
        #                      # TextBox("Acetyl-Co", [[606, 798], [149, 187]])], \
        #                      ], f"words and bounds are {words[0:3]}"
        assert words[0] == expected_textbox, f"{expected_textbox} and found: {words[0]}"
        print(f"temp {words[0:5]}")
        assert words[0:5] ==[TextBox("Glycolysis", [[405, 638], [1, 57]]),
                             TextBox("Terpene", [[182, 349], [53, 103]]),
                             TextBox("Biosynthetic", [[140, 390], [111, 145]]),
                             TextBox("Bethea", [[178, 329], [122, 200]]),
                             TextBox("Acetyl-Co", [[606, 798], [149, 187]])], f"words and bounds are {words[:5]}"
        """E       assert [Textbox(Glycolysis, [[405, 638], [1, 57]]),\n Textbox(Terpene, [[182, 349], [53, 103]]),\n Textbox(Biosynthetic, [[140, 390], [111, 145]]),\n Textbox(Bethea, [[178, 329], [122, 200]]),\n Textbox(Acetyl-Co, [[606, 798], [149, 187]])] ==
                          [Textbox(Glycolysis, [[405, 638], [1, 57]]),\n Textbox(Terpene, [[182, 349], [57, 99]]),\n Textbox(Biosynthetic, [[140, 390], [111, 145]]),\n Textbox(Bethea, [[178, 329], [122, 200]]),\n Textbox(Acetyl-Co, [[606, 798], [149, 187]])]
"""

    def test_phrases(self):
        phrases = self.biosynth2_ocr.get_phrases()
        assert len(phrases) == 59, f"phrases are {len(phrases)}"

    def test_groups(self):
        groups = self.biosynth2_ocr.get_groups()
        assert len(groups) == 59, f"groups are {len(groups)}"

    def test_clean(self):
        pass

    @unittest.skipUnless(Interactive, "interactive" )
    def test_plot_bbox_on_image(self):
        words = self.biosynth2_ocr.get_words()
        biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(self.biosynth2_img, words)
        io.imshow(biosynth2_img_bboxes)
        io.show()
    
    @unittest.skipUnless(Interactive, "interactive" )
    def test_bbox_fill(self):
        """tests filling background in a given bbox in an image"""
        box = BBox([[82, 389], [28, 386]])
        test_img = AmiOCR.set_bbox_to_bg(self.med_xrd_img, box)
        io.imshow(test_img)
        io.show()

    @unittest.skipUnless(Interactive, "interactive" )
    def test_extract_labels_from_plot(self):
        """test that labels are correctly OCRd in a plot"""
        box = BBox([[82, 389], [28, 386]])
        labels = AmiOCR.extract_labels_from_plot(self.med_xrd_img, box)

    @unittest.skipUnless(Interactive, "interactive" )
    def test_plot_pixel_stats(self):
        """tests that plots the pixel stats in an image"""
        AmiOCR.plot_image_pixel_stats(self.med_xrd_img, 255, axis=1)

    @unittest.skipUnless(Interactive, "interactive" )
    def test_img_rotation(self):
        """tests if an image can be rotated"""
        med_xrd_img_45 = AmiOCR.image_rotate(self.med_xrd_img, 45)
        io.imshow(med_xrd_img_45)
        io.show()

    @unittest.skipUnless(Interactive, "interactive" )
    def test_rotated_image_pixel_stats(self):
        """test to check if pixel statistics work on rotated image"""
        med_xrd_img_45 = AmiOCR.image_rotate(self.med_xrd_img, 45)
        AmiOCR.plot_image_pixel_stats(med_xrd_img_45, 255, axis=1)

    @unittest.skipUnless(Interactive, "interactive" )
    def test_shapes_pixel_stats(self):
        """pixel statistics of common shapes"""
        shapes1 = Resources.SHAPES_1_RAW
        shapes1_img = io.imread(shapes1)
        shapes1_img_bin = AmiImage.create_white_binary_from_image(shapes1_img)
        AmiOCR.plot_image_pixel_stats(shapes1_img_bin, 255, axis=1)

    