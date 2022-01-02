import pytesseract
from pathlib import Path
from lxml import etree as et
import numpy as np
from lxml.etree import Element, QName
from lxml import etree

from ..pyimage.svg import XMLNamespaces

try:
    from pyami.py4ami import wikimedia
except ImportError:
    print("Cannot load pyami wikimedia")

"""
This file is to play with the output of hocr
"""
A_BBOX = "bbox"
A_FILL = "fill"
A_FONT_SIZE = "font-size"
A_FONT_FAMILY = "font-family"
A_HEIGHT = "height"
A_STROKE = "stroke"
A_STROKE_WIDTH = "stroke-width"
A_TITLE = "title"
A_WIDTH = "width"
A_XLINK = 'xlink'
A_X = "x"
A_Y = "y"

E_G = 'g'
E_RECT = 'rect'
E_SVG = 'svg'
E_TEXT = "text"

XHTML_OCRX_WORD = "//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']"

TEXT_BOX_STROKE = "blue"
TEXT_BOX_FONT = "sans-serif"


class TesseractOCR:
    """Wrapper for pytesseract + addtional methods for phrase detection"""
    def __init__(self) -> None:
        pass

    @classmethod
    def pretty_print_hocr(cls, hocr_element):
        """Prints html string to console with proper indentation
        input: object of lxml.etree.Element class
        returns: None
        """
        print(et.tostring(hocr_element, pretty_print=True).decode("utf-8"))

    @classmethod
    def hocr_from_image_path(cls, path):
        """Runs tesseract hocr on the given image
        :param: Path
        :returns: hocr string
        """
        # pytesseract only seems to accept string for image path
        if path is not str:
            path = str(path)

        hocr = pytesseract.image_to_pdf_or_hocr(path, extension='hocr', config='11')
        # TODO what is this doing?
        cls.parse_hocr_string(hocr)
        return hocr

    @classmethod
    def read_hocr_file(cls, path):
        """Reads hocr html file and return root of hocr
        :input: path
        :returns: lxml.etree.Element object
        """
        # open file in bianry mode to prevent python from implicitly decoding
        # the bytes in the file as unicode, lxml parser does not like unicode
        with open(path, 'rb') as file:
            hocr = file.read()
        root = cls.parse_hocr_string(hocr)
        return root

    @classmethod
    def parse_hocr_string(cls, hocr_string):
        """Parses hocr output in string format as a tree using lxml
        :input: hocr html as string
        :returns: root of hocr as an object of lxml.etree.Element class
        """
        parser = et.HTMLParser()
        root = et.HTML(hocr_string, parser)
        return root

    @classmethod
    def draw_bbox_around_words(cls, image, bbox_coordinates):
        """Given bbox coordinates, draw bounding box on the image
        :param image: image (will be modified, is this OK?)
        :param bbox_coordinates: as x1, y1, x2, y2
        :returns: modified image
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

    @classmethod
    def extract_bbox_from_hocr(cls, hocr_element):
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
        bboxes = []
        words = []

        # Each of the bbox coordinates are in span
        # iterdescendants('span') iterates over all the span elements in the tree
        for child in hocr_element.iterdescendants('span'):
            # the class of span for bounding box around each word is 'ocrx_word'
            if child.attrib['class'] == 'ocrx_word':
                # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
                # we are only interested in the coordinates, so we split at ; and append the first part
                bboxes.append(child.attrib['title'].split(';')[0])
                words.append(child.text)

        # now we have a list with string elenments like 'bbox 333 74 471 102'
        # first we convert each string into a list, and then remove 'bbox', leaving just the coordinate
        bboxes = [bbox.split()[1:] for bbox in bboxes]

        # finally we convert the coordinates from string to integer
        bboxes = [[int(coordinate) for coordinate in bbox] for bbox in bboxes]

        # and return a numpy array from the generated list
        bboxes = np.array(bboxes)
        return bboxes, words

    @classmethod
    def find_phrases(cls, hocr_element, word_separation=20, y_tolerance=10):
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
        bboxes, words = cls.extract_bbox_from_hocr(hocr_element)
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

    @classmethod
    def wikidata_lookup(cls, phrases):
        """
        looks up phrases in wikidata.
        Requires Internet, May take a long time
        :param phrases: to lookup
        :return: tuple (list of qitem ids, list of descriptions
        """
        lookup = wikimedia.WikidataLookup()
        qitems, desc = lookup.lookup_items(phrases)
        return qitems, desc

    @classmethod
    def output_phrases_to_file(cls, phrase_list, output_file):
        """
        :param phrase_list: list of phrases_file
        :param output file: full output filename
        :returns: output file path
        """
        # TODO this doesn't look the right output directory
        phrases_file = Path(Path(__file__).parent.parent, f"temp/{output_file}")
        with open(phrases_file, 'w') as f:
            for item in phrase_list:
                f.write("%s\n" % item)
        return phrases_file

    @classmethod
    def extract_bbox_from_image(cls, path):
        """Given an image path, returns the coordinates for bboxes around the words
        :input: path
        :returns: numpy array
        """
        hocr = cls.hocr_from_image_path(path)
        root = cls.parse_hocr_string(hocr)
        bbox_for_words, words = cls.extract_bbox_from_hocr(root)
        
        return bbox_for_words, words

    @classmethod
    def parse_hocr_title(cls, title):
        """
         title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7"

        main function is to re-order the x, y values in bbox
        :param title:
        :return: dictionary with same kws
        """
        if title is None:
            return None
        parts = title.split("; ")
        title_dict = {}
        for part in parts:
            vals = part.split()
            kw = vals[0]
            if kw == A_BBOX:
                val = ((vals[1], vals[3]), (vals[2], vals[4]))
            else:
                val = vals[1:]
            title_dict[kw] = val
        return title_dict

    @classmethod
    def create_svg_from_hocr(cls, hocr_html):
        """

        :param hocr_html: input HOCR
        :return:
        """
        html = etree.parse(hocr_html)
        word_spans = html.findall(XHTML_OCRX_WORD)
        svg = Element(QName(XMLNamespaces.SVG_NS, E_SVG), nsmap={
            E_SVG: XMLNamespaces.SVG_NS,
            A_XLINK: XMLNamespaces.XLINK_NS,
        })
        for word_span in word_spans:
            title = word_span.attrib[A_TITLE]
            title_dict = cls.parse_hocr_title_HOCR(title)
            bbox = title_dict[A_BBOX]
            text = word_span.text
            g = cls.create_svg_text_box_from_hocr(bbox, text)
            svg.append(g)
        bb = etree.tostring(svg, encoding='utf-8', method='xml')
        s = bb.decode("utf-8")
        path_svg = Path(Path(__file__).parent.parent, "temp", "textbox.svg")
        with open(path_svg, "w", encoding="UTF-8") as f:
            f.write(s)
            print(f"Wrote textboxes to {path_svg}")

    @classmethod
    def create_svg_text_box_from_hocr(cls, bbox, txt):
        """
        creates a 'text-box' of form
        <g>
          <rect x="x" y="y" width="w" height="h"/>
          <text x="x" y="y" >string</text>
        </g>
        Assumes that bbox and txt have been created consistently,

        :param bbox: nested 2-arrays [[x0, x1], [y0,y1]]
        :param txt: string of constant y1 (since text is upward) starting at x1
        :return: <g> element with rect and text children
        """
        g = Element(QName(XMLNamespaces.SVG_NS, E_G))
        height = int(bbox[1][1]) - int(bbox[1][0])

        rect = cls.create_svg_rect_from_bbox(bbox, height)
        g.append(rect)

        text = Element(QName(XMLNamespaces.SVG_NS, E_TEXT))
        text.attrib[A_X] = bbox[0][0]
        text.attrib[A_Y] = str(int(bbox[1][0]) + height)
        text.attrib[A_FONT_SIZE] = str(0.9 * height)
        text.attrib[A_STROKE] = TEXT_BOX_STROKE
        text.attrib[A_FONT_FAMILY] = TEXT_BOX_FONT

        text.text = txt

        g.append(text)

        return g

    @classmethod
    def create_svg_rect_from_bbox(cls, bbox, height=None):
        """
        Create from bounding box but override height if None

        :param bbox:
        :param height:
        :return: svg:rect object
        """
        assert len(bbox) == 2
        assert len(bbox[0]) == 2
        assert len(bbox[1]) == 2
        rect = Element(QName(XMLNamespaces.SVG_NS, E_RECT))
        rect.attrib[A_X] = str(bbox[0][0])
        rect.attrib[A_WIDTH] = str(int(bbox[0][1]) - int(bbox[0][0]))
        rect.attrib[A_Y] = str(int(bbox[1][0]))  # kludge for offset of inverted text
        rect.attrib[A_HEIGHT] = str(height) if height is not None else str(int(bbox[1][1]) - int(bbox[1][0]))
        rect.attrib[A_STROKE_WIDTH] = "1.0"
        rect.attrib[A_STROKE] = "red"
        rect.attrib[A_FILL] = "none"
        return rect
