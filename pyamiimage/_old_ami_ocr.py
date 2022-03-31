from os import path
from pathlib import Path
from tkinter import BASELINE
import numpy as np
import pytesseract
from lxml import etree as et
from PIL import Image
import scipy.ndimage as ndimage
from skimage import io
from matplotlib import pyplot as plt
from configparser import ConfigParser

from pyamiimage.bbox import BBox
from pyamiimage.cleaner import WordCleaner
from pyamiimage.ami_image import AmiImage

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
    def __init__(self, path=None, image=None) -> None:
        """Creates an OCR object from path and images, if both path and images are given, path takes precendence"""
        if path is not None:
            self.hocr = AmiOCR.run_ocr(path)
        elif image is not None:
            self.hocr = AmiOCR.run_ocr_on_image(image)
        else:
            self.hocr = None
        self.words = []
        self.phrases = []
        self.groups = []

    @classmethod
    def run_ocr_on_image(cls, image, filename="default.png"):
        """
        saves image as a temp file then ocr the file
        image is a 2D numpy arr
        """
        filepath = Path(AmiOCR.TESSERACT_TEMP_PATH, filename)
        AmiImage.write(filepath, image, mkdir=False)
        hocr = AmiOCR.hocr_from_image_path(filepath)
        return hocr
    
    @classmethod
    def create_temp_file(cls, image, filename):
        """creates a temporary file given path and image"""
        filepath = Path(AmiOCR.TESSERACT_TEMP_PATH, filename)
        AmiImage.write(filepath, image, mkdir=False)


    @classmethod
    def run_ocr(cls, path=None):
        if path is None:
            return None
        else:
            hocr = AmiOCR.hocr_from_image_path(path)
            return hocr


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

    @classmethod
    def hocr_from_image_path(cls, path, psm='12'):
        """Runs tesseract hocr on the given image
        :param: Path
        :returns: hocr
        """
        hocr_string = AmiOCR.hocr_string_from_path(path, psm)
        hocr = AmiOCR.parse_hocr_string(hocr_string)
        return hocr

    @classmethod
    def hocr_string_from_path(cls, path, psm='12'):
        # pytesseract only seems to accept string for image path
        if path is not str:
            path = str(path)

        hocr_string = pytesseract.image_to_pdf_or_hocr(path, extension='hocr', config=psm)
        return hocr_string


    @classmethod
    def parse_hocr_string(cls, hocr_string):
        """Parses hocr output in string format as a tree using lxml
        :input: hocr html as string
        :returns: root of hocr as an object of lxml.etree.Element class
        """
        parser = et.HTMLParser()
        root = et.HTML(hocr_string, parser)
        return root

    def parse_hocr_tree(self, hocr_element=None, cleaning=False):
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
        if cleaning:
            self.words = AmiOCR.clean_all(words)
        else:
            self.words = words
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
        temp = np.copy(image)
        for textbox in textboxes:
            try:
                temp = BBox.plot_bbox_on(temp, textbox.bbox)
            except IndexError as e:
                continue
        return temp
    
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
            patch = AmiOCR.copy_textbox_values(image, textbox)
            patches.append(patch)   
        return patches

    @classmethod
    def copy_textbox_values(cls, image, textbox, padding=0):
        """given image and textbox, will return the numpy array of the size of the textbox from the image"""
        xy_range = textbox.bbox.xy_ranges
        min_y = xy_range[1][0]
        max_y = xy_range[1][1]
        min_x = xy_range[0][0]
        max_x = xy_range[0][1]
        # print(f"min_y: {min_y}", f"max_y: {max_y}", f"min_x: {min_x}", f"max_x: {max_x}")
        patch = image[min_y:max_y, min_x:max_x]
        return patch

    @classmethod
    def read_textbox(cls, image, textbox):
        """rereads the textboxes with some added padding"""
        patch = AmiOCR.copy_textbox_values(image, textbox)
        patch_ocr = AmiOCR(image=patch)
        return patch_ocr

    @classmethod
    def clean_all(self, textboxes):
        cleaned = WordCleaner.remove_leading_and_trailing_special_characters(textboxes)
        cleaned = WordCleaner.remove_all_single_characters(cleaned)
        cleaned = WordCleaner.remove_all_sequences_of_special_characters(cleaned)
        cleaned = WordCleaner.remove_misread_letters(cleaned)
        #cleaned = WordCleaner.remove_numbers_only(cleaned)
        return cleaned

    @classmethod
    def remove_textboxes_from_image(cls, image, textboxes):
        """Given an image and a list of textboxes, sets the bounding boxes to bg value"""
        for textbox in textboxes:
            # bg = AmiOCR.find_bg() #TBI
            image = cls.set_bbox_to_bg(image, textbox.bbox)
        return image

    @classmethod
    def set_bbox_to_bg(cls, image, bbox, bg = 255):
        """given a bounding box set the value in the bounding box in the image to bg color"""
        row_range = bbox.get_yrange()
        column_range = bbox.get_xrange()
        image[row_range[0] : row_range[1]+1, column_range[0]: column_range[1]+1] = bg
        return image

    @classmethod
    def extract_labels_from_plot(cls, image, plot_area_bbox):
        """
        given the image and plot boundaries extracts vertical y labels and horizontal x labels
        :param: image 
        :type: numpy array
        :param: plot_bbox 
        :type: BBox object
        :returns: AmiOCR object for the whole plot
        """
        new_img = BBox.plot_bbox_on(image, plot_area_bbox)
        label_bboxes = AmiOCR.label_bboxes_from_plot_bbox(image, plot_area_bbox)
        # # x_label = AmiOCR.copy_bbox_from_img(image, label_bboxes['x'])
        # # y_label = AmiOCR.copy_bbox_from_img(image, label_bboxes['y'])
        
        #================= X label ===============
        # x_label_ocr = AmiOCR.ocr_subimage(image, label_bboxes['x'])
        # x_items = x_label_ocr.get_words()

        # full_img_with_xlabels = AmiOCR.plot_bboxes_on_image(image, x_items) 
        # io.imshow(full_img_with_xlabels)

        #================ Y label ===============
        # Does not work yet
        # y_label_ocr = AmiOCR.ocr_subimage(image, label_bboxes['y'])
        # y_items = y_label_ocr.get_words()

        # full_img_with_labels = AmiOCR.plot_bboxes_on_image(full_img_with_xlabels, y_items)
        
        y_ticks_ocr = AmiOCR.ocr_subimage(image, label_bboxes['y_ticks'])
        y_tk_items = y_ticks_ocr.get_words()
        for item in y_tk_items:
            print(item)

        full_img_with_ticks = AmiOCR.plot_bboxes_on_image(image, y_tk_items)
        io.imshow(full_img_with_ticks)
        io.show()

    @classmethod
    def ocr_subimage(cls, image, bbox):
        """given an image and bbox, return AmiOCR object of the image with only the bbox being in OCR
        :param: image
        :type: numpy array
        :bbox: bounding box for subimage to OCR
        :type: BBox
        :returns: AmiOCR object
        """
        subimage = AmiOCR.copy_bbox_from_img(image, bbox)
        io.imshow(subimage)
        io.show()
        subimage_ocr = AmiOCR(image=subimage)
        starting_point = bbox.get_point_pair()[0]
        col_shift = starting_point[1]
        row_shift = starting_point[0]
        words = subimage_ocr.get_words()
        print(len(words))
        for word in subimage_ocr.words:
            print("word with range: ", word)
            old_range = word.bbox.get_ranges()
            # shift the coordinates of the bounding boxes for the coordinates of the whole image
            new_range = [[x+col_shift for x in old_range[0]], [y+row_shift for y in old_range[1]]]
            word.bbox.set_ranges(new_range)
            print("words with new range: ", word)
        return subimage_ocr


    def join(self, other):
        """Combines two AmiOCR objects into one"""
        if other is AmiOCR:
            self.words += other.words
            self.phrases += other.phrases
            self.groups += other.groups
        else:
            print("parameter must be AmiOCR object")

    @classmethod
    def label_bboxes_from_plot_bbox(cls, image, plot_bbox, tick_dist = 42):
        """given image and plot_bbox generate x_label bbox and y_label bbox
        :param: image
        :type: numpy array
        :param: plot_bbox
        :type: BBox object
        """
        plot_col_range = plot_bbox.get_xrange()
        plot_row_range = plot_bbox.get_yrange()
        img_height = image.shape[0]
        img_width = image.shape[1]
        x_tk_bbox = BBox([[plot_col_range[0], img_width], [plot_row_range[1],plot_row_range[1]+tick_dist]])
        y_tk_bbox = BBox([[plot_col_range[0]-45, plot_col_range[0]],[0, plot_row_range[1]]])
        x_bbox = BBox([[plot_col_range[0], img_width], [plot_row_range[1],img_height]])
        y_bbox = BBox([[0, plot_col_range[0]],[0, plot_row_range[1]]])
        return {'x': x_bbox, 'y':y_bbox, 'y_ticks':y_tk_bbox, 'x_ticks':x_tk_bbox}

    @classmethod
    def copy_bbox_from_img(cls, image, bbox):
        """given an image and a bounding box, returns a image having values from the image within the bbox
        :param: image
        :type: numpy array
        :param: bbox
        :type: BBox
        :returns: numpy array (image)
        """
        row_range = bbox.get_yrange()
        col_range = bbox.get_xrange()
        snippet = image[row_range[0]:row_range[1], col_range[0]: col_range[1]]
        return snippet

    @classmethod
    def image_pixel_stats(cls, image, signal_val, axis=1):
        """sum of signal values along a axis in an array"""
        signal = np.count_nonzero(image==signal_val, axis=axis)
        signal = np.array(signal)
        return signal

    @classmethod
    def image_rotate(cls, image, degrees):
        return ndimage.rotate(image, degrees, reshape=True)

    @classmethod
    def plot_image_pixel_stats(cls, image, signal_val, axis=1):
        image_bin = AmiImage.create_white_binary_from_image(image)
        signal = AmiOCR.image_pixel_stats(image_bin, signal_val, axis-1)
        row = np.arange(0, len(signal))
        plt.imshow(image_bin)
        plt.plot(row, signal, color='red')
        plt.show()

    
    def write_list_to_file(self, list, filename):
        with open(filename, 'w') as f:
            for textbox in list:
                f.write(textbox.text + "\n")
