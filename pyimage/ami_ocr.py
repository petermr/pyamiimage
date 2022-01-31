from os import path
from pathlib import Path
from tkinter import BASELINE
import numpy as np
import codecs
import pytesseract
from lxml import etree as et
from PIL import Image
from skimage import io

from pyimage.ami_image import AmiImage
try:
    from pyimage.bbox import BBox
    from pyimage.cleaner import WordCleaner
except: 
    from ..pyimage.bbox import BBox
    from ..pyimage.cleaner import WordCleaner

class TextBox():
    # TextBox inherits BBox
    def __init__(self, text, xy_ranges, baseline=0) -> None:
        self.text = text
        self.bbox = BBox(xy_ranges)
        self.baseline = baseline
    
    def __repr__(self): 
        return f"Textbox({self.text}, {self.bbox.xy_ranges})"

    def __str__(self): 
        return f"text: {self.text} bbox: {self.bbox.xy_ranges}"

    def __eq__(self, other):
        if type(other) is not TextBox:
            return False
        return self.text == other.text and self.bbox.xy_ranges == other.bbox.xy_ranges

    def set_text(self, text):
        self.text = text

    def set_bbox(self, bbox):
        self.bbox = bbox

    

class AmiOCR:
    TESSERACT_TEMP_PATH = Path(Path(__file__).parent.parent, "temp/tesseract/")
    def __init__(self, path=None, image=None, filename="default.png") -> None:
        """Creates an OCR object from path and images, if both path and images are given, path takes precendence"""
        self.hocr_string = None
        if path is not None:
            self.hocr = self.run_ocr(path)
        elif image is not None:
            filepath = Path(AmiOCR.TESSERACT_TEMP_PATH, filename)
            print(filepath)
            AmiImage.write(filepath, image, mkdir=False)
            self.hocr = self.run_ocr(filepath)
        else:
            self.hocr = None
        self.words = []
        self.phrases = []
        self.groups = []

    def run_ocr(self, path=None):
        if path is None:
            return None
        else:
            return self.hocr_from_image_path(path)


    def get_words(self):
        if self.words == []:
            self.words = self.parse_hocr_tree()
        return self.words
        
    def get_phrases(self):
        if self.phrases == []:
            self.phrases = self.find_phrases()
        return self.phrases
    
    def get_groups(self):
        if self.groups == []:
            self.groups = self.find_word_groups()
        return self.phrases

    @classmethod
    def pretty_print_hocr(cls, hocr_element):
        """Prints html string to console with proper indentation
        input: object of lxml.etree.Element class
        returns: None
        """
        print(et.tostring(hocr_element, pretty_print=True).decode("utf-8"))

    def hocr_from_image_path(self, path):
        """Runs tesseract hocr on the given image
        :param: Path
        :returns: hocr
        """
        # pytesseract only seems to accept string for image path
        if path is not str:
            path = str(path)

        self.hocr_string = pytesseract.image_to_pdf_or_hocr(path, extension='hocr', config='11')
        # self.hocr_string = codecs.decode(self.hocr_string, 'UTF-8')
        # print(self.hocr_string)
        hocr = self.parse_hocr_string(self.hocr_string)
        print(hocr)
        return hocr

    def parse_hocr_string(self, hocr_string):
        """Parses hocr output in string format as a tree using lxml
        :input: hocr html as string
        :returns: root of hocr as an object of lxml.etree.Element class
        """
        parser = et.HTMLParser()
        root = et.HTML(hocr_string, parser)
        return root

    def parse_hocr_tree(self, hocr_element=None):
        """to extract bbox coordinates from html
        :param hocr_element: lxml.etree.Element object
        :returns: tuple of (list of bboxes, list of words)
        numpy array has the format: [x1, y1, x2, y2]
        where values corresponding to the bounding box are
        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2    
        """
        if hocr_element is None:
            hocr_element = self.hocr
        words = []

        # Each of the bbox coordinates are in span
        # iterdescendants('span') iterates over all the span elements in the tree
        for child in hocr_element.iterdescendants('span'):
            # the class of span for bounding box around each word is 'ocrx_word'
            if child.attrib['class'] == 'ocrx_word':
                # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
                # we are only interested in the coordinates, so we split at ; and append the first part
                bbox_string = child.attrib['title'].split(';')[0]
                xy_range = self.create_xy_range_from_bbox_string(bbox_string)
                textbox = TextBox(child.text, xy_range)
                words.append(textbox)
        self.words = AmiOCR.clean_all(words)
        return self.words
    
    def find_words_from_image_path(self, image_path):
        self.hocr_from_image_path(image_path)
        self.parse_hocr_tree(self.hocr)
        return self.words

    def create_xy_range_from_bbox_string(self, bbox_string):
        # now we have a list with string elenments like 'bbox 333 74 471 102'
        # first we convert each string into a list, and then remove 'bbox', leaving just the coordinate
        bbox_list = bbox_string.split()[1:]

        # finally we convert the coordinates from string to integer
        bbox_list = [int(coordinate) for coordinate in bbox_list]
        # bbox_list = [x1, y1, x2, y2]
        xy_range = [[bbox_list[0],bbox_list[2]], [bbox_list[1], bbox_list[3]]]
        return xy_range

    def find_phrases(self, hocr_element=None, word_separation=20, min_y_overlap=0.99):
        """
        finds phrases and their bboxes from HOCR output
        adjusts the classification from Tesseract which is often fragile
        :param hocr_element: output from Tesseract
        :param word_separation: maximum? separation (in pixels) of words to create phrases
        :param y_tolerance: maximum tolerance (in pixels) of words on same line
        :return: tuple (list of phrases, list of boxes)
        """
        phrases = []
        words = self.get_words()
        
        for i in range(len(words)):
            phrase = words[i] # phrase is a TextBox object like word
            for j in range(i+1, len(words)):
                # check if words are on the same line
                # if words on a different line, break loop
                if not AmiOCR.y_overlap(phrase, words[j]) >= min_y_overlap:
                    break
                # If the separation is small enough, add the word to the phrase
                if AmiOCR.textbox_horizontal_seperation(phrase, words[j]) < word_separation:
                    phrase.set_text(phrase.text + " " + words[j].text)
                    phrase.set_bbox(phrase.bbox.union(words[j].bbox))
                else:
                    break

            if phrase.text == ' ':
                # if phrase is empty do not add to the list of phrases
                continue
            elif not phrases:
                # if array is empty add element
                phrases.append(phrase)
            elif not phrases[-1].text.endswith(phrase.text):
                # only add phrase if the last phrase added does not end with current phrase
                phrases.append(phrase)
        self.phrases = phrases
        return phrases

    @classmethod
    def textbox_horizontal_seperation(cls, textbox_1, textbox_2):
        """we assume that textbox_1 comes before textbox_2"""
        return textbox_2.bbox.get_xrange()[0]-textbox_1.bbox.get_xrange()[1]

    @classmethod
    def textboxes_in_same_line(cls, textbox_1, textbox_2, y_tolerance):
        textbox_1_yrange = textbox_1.bbox.get_yrange()
        textbox_2_yrange = textbox_2.bbox.get_yrange()
        if abs(textbox_1_yrange[0] - textbox_2_yrange[0]) <= y_tolerance and \
            abs(textbox_1_yrange[1] - textbox_1_yrange[1]) <= y_tolerance:
            return True
        else:
            return False

    def find_word_groups(self, phrases=None, line_seperation=10, min_x_overlap=0.20):
        """
        :param bbox_of_phrases: bounding boxes of phrases in an image
        :type bbox_of_phrases: list
        :param line_seperation: allowable distance between two lines
        :type line_seperation: int
        :param min_x_overlap: ratio of horizontal overlap between two bboxes
        :type min_x_overlap: float
        """
        if phrases == None:
            phrases = self.get_phrases()

        groups = []
        
        for i in range(len(phrases)):
            group = phrases[i] # group is a TextBox object 
            for j in range(i+1, len(phrases)):
                # check if there is significant x range overlap between the boxes
                if AmiOCR.x_overlap(group, phrases[j]) < min_x_overlap:
                    break
                # If the separation is small enough, add the phrase to the group
                if AmiOCR.textbox_vertical_seperation(group, phrases[j]) < line_seperation:
                    group.set_text(group.text + " " + phrases[j].text)
                    group.set_bbox(group.bbox.union(phrases[j].bbox))
                else:
                    break

            if group.text == ' ':
                # if group is empty do not add to the list of groups
                continue
            elif not groups:
                # if array is empty add element
                groups.append(group)
            elif not groups[-1].text.endswith(group.text):
                # only add phrase if the last phrase added does not end with current phrase
                groups.append(group)
        return groups

    @classmethod
    def textbox_vertical_seperation(cls, textbox1, textbox2):
        bbox1_y = textbox1.bbox.get_yrange()
        bbox2_y = textbox2.bbox.get_yrange()

        # find which box is top and which box is bottom
        top_bbox = bbox1_y if bbox1_y[0] < bbox2_y[0] else bbox2_y
        bottom_bbox = bbox2_y if top_bbox == bbox1_y else bbox1_y

        # if the bounding boxes intersect then we will have a negative difference but that's fine
        return bottom_bbox[0] - top_bbox[1]



    @classmethod
    def y_overlap(cls, textbox1, textbox2):
        # find which bbox is wider
        bbox1_y = textbox1.bbox.get_yrange()
        bbox2_y = textbox2.bbox.get_yrange()

        pixel_overlap = max(0, min(bbox1_y[1], bbox2_y[1]) - max(bbox1_y[0], bbox2_y[0]) + 1)
        
        if pixel_overlap == 0:
            return 0

        shorter_bbox_height = min(textbox1.bbox.get_height(), textbox2.bbox.get_height())
        
        overlap = pixel_overlap/shorter_bbox_height
        return overlap

    @classmethod
    def x_overlap(cls, textbox1, textbox2):
        # find which bbox is wider
        bbox1_x = textbox1.bbox.get_xrange()
        bbox2_x = textbox2.bbox.get_xrange()
        pixel_overlap = max(0, min(bbox1_x[1], bbox2_x[1]) - max(bbox1_x[0], bbox2_x[0]) + 1)
        
        if pixel_overlap == 0:
            return 0

        narrower_bbox_width = min(textbox1.bbox.get_width(), textbox2.bbox.get_width())
        
        overlap = pixel_overlap/narrower_bbox_width
        return overlap

    @classmethod
    def envelope_box(cls, bboxes):
        # given list of bboxes gives the whole box enveloping all the bboxes
        min_x, min_y, max_x, max_y = bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3] 
        for bbox in bboxes:
            if bbox[0] < min_x:
                min_x = bbox[0]
            if bbox[1] < min_y:
                min_y = bbox[1]
            if bbox[2] > max_x:
                max_x = bbox[2]
            if bbox[3] > max_y:
                max_y = bbox[3]
        return [min_x, min_y, max_x, max_y]

    @classmethod
    def plot_bboxes_on_image(self, image, textboxes):
        """draws bboxes on image 
        :param: image
        :type: numpy array
        :textboxes: array of TextBox objects
        :returns: image
        """
        for textbox in textboxes:
            try:
                image = BBox.plot_bbox_on(image, textbox.bbox)
            except IndexError as e:
                continue
        return image
    
    def parse_hocr_title(self, title):
        """
         title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7"

        :param title:
        :return:
        """
        if title is None:
            return None
        parts = title.split("; ")
        title_dict = {}
        for part in parts:
            vals = part.split()
            kw = vals[0]
            if kw == self.A_BBOX:
                val = ((vals[1], vals[3]), (vals[2], vals[4]))
            else:
                val = vals[1:]
            title_dict[kw] = val
        return title_dict

    def find_baseline(self, hocr_element):
        if hocr_element is None:
            hocr_element = self.hocr
        words = []

        # Each of the bbox coordinates are in span
        # iterdescendants('span') iterates over all the span elements in the tree
        for child in hocr_element.iterdescendants('span'):
            # the class of span for bounding box around each word is 'ocrx_word'
            if child.attrib['class'] == 'ocr_line':
                # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
                # we are only interested in the coordinates, so we split at ; and append the first part
                print("pause here")
                baseline = child.attrib['title'].split(';')[1]
                bbox_string = child.attrib['title'].split(';')[0]
                xy_range = self.create_xy_range_from_bbox_string(bbox_string)
                textbox = TextBox(child.text, xy_range, baseline)
                words.append(textbox)
        self.words = AmiOCR.clean_all(words)
        return self.words
        

    @classmethod
    def bounding_box_patches(cls, image, textboxes):
        """given an image and textboxes, it will extract all the bounding boxes
        and return a list of numpy arrays"""
        patches = []
        for textbox in textboxes:
            xy_range = textbox.bbox.xy_ranges
            min_y = xy_range[1][0]
            max_y = xy_range[1][1]
            min_x = xy_range[0][0]
            max_x = xy_range[0][1]
            print(f"min_y: {min_y}", f"max_y: {max_y}", f"min_x: {min_x}", f"max_x: {max_x}")
            patch = image[min_y:max_y, min_x:max_x]
            print(patch.shape)
            patches.append(patch)
        
        return patches

    @classmethod
    def clean_all(self, textboxes):
        cleaned = WordCleaner.remove_leading_and_trailing_special_characters(textboxes)
        cleaned = WordCleaner.remove_all_single_characters(cleaned)
        cleaned = WordCleaner.remove_all_sequences_of_special_characters(cleaned)
        cleaned = WordCleaner.remove_misread_letters(cleaned)
        cleaned = WordCleaner.remove_numbers_only(cleaned)
        return cleaned