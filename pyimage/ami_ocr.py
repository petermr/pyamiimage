from os import path
import numpy as np
import codecs
import pytesseract
from lxml import etree as et
from PIL import Image
from skimage import io
try:
    from pyimage.bbox import BBox
    from pyimage.cleaner import WordCleaner
except: 
    from ..pyimage.bbox import BBox
    from ..pyimage.cleaner import WordCleaner

class TextBox(BBox):
    # TextBox inherits BBox
    def __init__(self, text, xy_ranges) -> None:
        self.text = text
        BBox.__init__(self, xy_ranges)
    
    def __repr__(self): 
        return f"Textbox({self.text}, {self.xy_ranges})"

    def __str__(self): 
        return f"text: {self.text} bbox: {self.xy_ranges}"

    def __eq__(self, other):
        if type(other) is not TextBox:
            return False
        return self.text == other.text and self.xy_ranges == other.xy_ranges

    def set_text(self, text):
        self.text = text
    
    def get_text(self):
        return self.text
class AmiOCR:
    def __init__(self, path=None) -> None:
        self.hocr_string = None
        self.hocr = self.run_ocr(path)
        self.raw_tesseract = []
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
        self.words = AmiOCR.clean(words)
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

    def find_phrases(self, hocr_element, word_separation=20, y_tolerance=10):
        """
        finds phrases and their bboxes from HOCR output
        adjusts the classification from Tesseract which is often fragile
        :param hocr_element: output from Tesseract
        :param word_separation: maximum? separation (in pixels) of words to create phrases
        :param y_tolerance: maximum tolerance (in pixels) of words on same line
        :return: tuple (list of phrases, list of boxes)
        """
        phrases = []
        bbboxes = []
        bboxes, words = self.extract_bbox_from_hocr(hocr_element)
        
        # # TODO verify this doesn't break the system
        # words, bboxes = TesseractOCR.remove_bad_word_bboxes(words, bboxes)
        
        for i in range(len(words)):
            phrase = [words[i]]
            last_bbox = bboxes[i]
            for j in range(i+1, len(words)):
                # check if words are on the same line
                # if words on a different line, break loop
                if abs(bboxes[j][1] - bboxes[i][1]) > y_tolerance or abs(bboxes[j][3] - bboxes[i][3]) > y_tolerance:
                    break
                # If the separation is small enough, add the word to the phrase
                if (bboxes[j][0] - last_bbox[2]) < word_separation:
                    phrase.append(words[j])
                    last_bbox = bboxes[j]
                else:
                    break
            phrase = " ".join(phrase)
            if phrase == ' ':
                # if phrase is empty do not add to the list of phrases
                continue
            elif not phrases:
                # if array is empty add element
                phrases.append(phrase)
                bbboxes.append([bboxes[i][0], bboxes[i][1], last_bbox[2], last_bbox[3]])
            elif not phrases[-1].endswith(phrase):
                # only add phrase if the last phrase added does not end with current phrase
                phrases.append(phrase)
                bbboxes.append([bboxes[i][0], bboxes[i][1], last_bbox[2], last_bbox[3]])
        return phrases, bbboxes

    def find_word_groups(self, bbox_of_phrases, line_seperation=10, min_x_overlap=0.20):
        """
        :param bbox_of_phrases: bounding boxes of phrases in an image
        :type bbox_of_phrases: list
        :param line_seperation: allowable distance between two lines
        :type line_seperation: int
        :param min_x_overlap: ratio of horizontal overlap between two bboxes
        :type min_x_overlap: float
        """
        groups = []
        for bbox in bbox_of_phrases:
            # sort each bbox into a group
            
            # if no group exists create a group
            if len(groups) == 0:
                groups.append(bbox)
                continue
            
            group_found = False
            for index, group in enumerate(groups):
                if abs(group[3] - bbox[1]) < line_seperation or abs(group[1] - bbox[3]) < line_seperation:
                    if self.x_overlap(group, bbox) > min_x_overlap:
                        groups[index] = AmiOCR.envelope_box([group, bbox])
                        group_found = True
                        break

            # if bbox doesn't fit a group, create a new group
            if not group_found:
                groups.append(bbox)

        self.groups = groups
        return groups

    def x_overlap(self, bbox1, bbox2):
        # find which bbox is wider
        pixel_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1)
        
        if pixel_overlap == 0:
            return 0
        
        bbox1_width = bbox1[2] - bbox1[0]
        bbox2_width = bbox2[2] - bbox2[0]
        narrower_bbox_width = min(bbox1_width, bbox2_width)
        
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
                image = BBox.plot_bbox_on(image, textbox)
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

    def find_baseline(self):
        # html = et.parse(hocr_html)
        # assert self.hocr == "hello world", f'should be {self.hocr}'
        html = et.parse(self.hocr_string)
        print(html)
        line_spans = html.findall(".//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
        # line_spans = self.hocr.findall("span[@class='ocr_line']")
        print("Hello world")
        assert line_spans is None
        for line_span in line_spans:
            title = line_span.attrib['title']
            title_dict = self.parse_hocr_title(title)
            for item in title_dict:
                print(item)
            # bbox = title_dict['bbox']
        

    @classmethod
    def clean(self, textboxes):
        cleaned = WordCleaner.remove_trailing_special_characters(textboxes)
        cleaned = WordCleaner.remove_all_single_characters(cleaned)
        cleaned = WordCleaner.remove_all_sequences_of_special_characters(cleaned)
        return cleaned
