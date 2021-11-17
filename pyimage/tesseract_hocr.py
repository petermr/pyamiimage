import pytesseract
from pathlib import Path
from lxml import etree as et
import numpy as np
from pyimage.preprocessing import ImageProcessor
try:
    from pyami.py4ami import wikimedia
except:
    print("Cannot load pyami")

"""
This file is to play with the output of hocr
"""
class TesseractOCR:
    """Wrapper for pytesseract + addtional methods for phrase detection"""
    def __init__(self) -> None:
        self.root = None

    def pretty_print_hocr(self, root):
        """Prints html string to console with proper indentation
        input: object of lxml.etree.Element class
        returns: None
        """
        print(et.tostring(root, pretty_print=True).decode("utf-8"))

    def hocr_from_image_path(self, path):
        """Runs tesseract hocr on the given image
        :param: Path
        :returns: hocr string
        """
        # pytesseract only seems to accept string for image path
        if path is not str:
            path = str(path)

        hocr = pytesseract.image_to_pdf_or_hocr(path, extension='hocr', config='11')
        root = self.parse_hocr_string(hocr)
        return hocr

    def read_hocr_file(self, path):
        """Reads hocr html file and return root of hocr
        :input: path
        :returns: lxml.etree.Element object
        """
        # open file in bianry mode to prevent python from implicitly decoding
        # the bytes in the file as unicode, lxml parser does not like unicode
        with open(path, 'rb') as file:
            hocr = file.read()
        root = self.parse_hocr_string(hocr)
        return root

    def parse_hocr_string(self, hocr_string):
        """Parses hocr output in string format as a tree using lxml
        :input: hocr html as string
        :returns: root of hocr as an object of lxml.etree.Element class
        """
        parser = et.HTMLParser()
        self.root = et.HTML(hocr_string, parser)
        return self.root
        
    def draw_bbox_around_words(self, image, bbox_coordinates):
        """Given bbox coordinates, draw bounding box on the image
        :input: numpy array, numpy array
        :returns: numpy array
        """
        max_row = image.shape[0]
        max_column = image.shape[1]
        for bbox in bbox_coordinates:
            for column in range(bbox[0], bbox[2]+1):
                for row in range(bbox[1], bbox[3]+1):
                    if row >= max_row or column >= max_column:
                        continue
                    if row == bbox[1] or row == bbox[3]:
                        image[row][column] = 0
                    if column == bbox[0] or column == bbox[2]:
                        image[row][column] = 0
        return image

    def extract_bbox_from_hocr(self, root):
        """to extract bbox coordinates from html
        :input: lxml.etree.Element object
        :returns: numpy array
        numpy array has the format: [x1, y1, x2, y2]
        where values corresponding to the bounding box are
        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2    
        """
        bbox_for_words = []
        words = []

        # Each of the bbox coordinates are in span
        # iterdescendants('span') iterates over all the span elements in the tree
        for child in root.iterdescendants('span'):
            # the class of span for bounding box around each word is 'ocrx_word'
            if child.attrib['class'] == 'ocrx_word':
                # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
                # we are only interested in the coordinates, so we split at ; and append the first part
                bbox_for_words.append(child.attrib['title'].split(';')[0])
                words.append(child.text)

        # now we have a list with string elenments like 'bbox 333 74 471 102'
        # first we convert each string into a list, and then remove 'bbox', leaving just the coordinate
        bbox_for_words = [bbox.split()[1:] for bbox in bbox_for_words]

        # finally we convert the coordinates from string to integer
        bbox_for_words = [[int(coordinate) for coordinate in bbox] for bbox in bbox_for_words]

        # and return a numpy array from the generated list
        bbox_for_words =  np.array(bbox_for_words)
        return bbox_for_words, words

    def find_phrases(self, root, word_seperation=20, y_tolerance=10):
        phrases = []
        bbox_for_phrases = []
        bboxes, words = self.extract_bbox_from_hocr(root)
        for i in range(len(words)):
            phrase = [words[i]]
            last_bbox = bboxes[i]
            for j in range(i+1, len(words)):
                # check if words are on the same line
                # if words on a different line, break loop
                if abs(bboxes[j][1] - bboxes[i][1]) > y_tolerance or abs(bboxes[j][3] - bboxes[i][3]) > y_tolerance:
                    break
                # If the seperation is small enough, add the word to the phrase
                if (bboxes[j][0] - last_bbox[2]) < word_seperation:
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
                bbox_for_phrases.append([bboxes[i][0], bboxes[i][1], last_bbox[2], last_bbox[3]])
            elif not phrases[-1].endswith(phrase):
                # only add phrase if the last phrase added does not end with current phrase
                phrases.append(phrase)
                bbox_for_phrases.append([bboxes[i][0], bboxes[i][1], last_bbox[2], last_bbox[3]])
        return phrases, bbox_for_phrases

    def wikidata_lookup(self, phrases):
        lookup = wikimedia.WikidataLookup()
        qitems, desc = lookup.lookup_items(phrases)
        return qitems, desc

    def output_phrases_to_file(self, list, output_file):
        """
        :param: list of phrases, output file name
        :returns: output file path"""
        BIOSYNTH3_PHRASES = Path(Path(__file__).parent.parent, f"temp/{output_file}")
        with open(BIOSYNTH3_PHRASES, 'w') as f:
            for item in list:
                f.write("%s\n" % item)
        return BIOSYNTH3_PHRASES

    def extract_bbox_from_image(self, path):
        """Given an image path, returns the coordinates for bboxes around the words
        :input: path
        :returns: numpy array
        """
        hocr = self.hocr_from_image_path(path)
        root = self.parse_hocr_string(hocr)
        self.pretty_print_hocr(root)
        bbox_for_words, words = self.extract_bbox_from_hocr(root)
        
        return bbox_for_words, words