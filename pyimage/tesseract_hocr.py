import pytesseract
from pathlib import Path
from lxml import etree as et
import numpy as np

"""
This file is to play with the output of hocr
"""

def pretty_print_html(root):
    """Prints html string to console with proper indentation
    input: object of lxml.etree.Element class
    returns: None
    """
    print(et.tostring(root, pretty_print=True).decode("utf-8"))

def hocr_on_image(path):
    """Runs tesseract hocr on the given image
    input: Path
    returns: hocr string
    """
    # pytesseract only seems to accept string for image path
    if path is not str:
        path = str(path)

    hocr = pytesseract.image_to_pdf_or_hocr(path, extension='hocr')
    return hocr

def parse_hocr_string(hocr):
    """Parses hocr output in string format as a tree using lxml
    :input: hocr html as string
    :returns: root of hocr as an object of lxml.etree.Element class
    """
    parser = et.HTMLParser()
    root = et.HTML(hocr, parser)
    return root
    

def extract_bbox_from_hocr(root):
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

    # Each of the bbox coordinates are in span
    # iterdescendants('span') iterates over all the span elements in the tree
    for child in root.iterdescendants('span'):
        # the class of span for bounding box around each word is 'ocrx_word'
        if child.attrib['class'] == 'ocrx_word':
            # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
            # we are only interested in the coordinates, so we split at ; and append the first part
            bbox_for_words.append(child.attrib['title'].split(';')[0])

    # now we have a list with string elenments like 'bbox 333 74 471 102'
    # first we convert each string into a list, and then remove 'bbox', leaving just the coordinate
    bbox_for_words = [bbox.split()[1:] for bbox in bbox_for_words]

    # finally we convert the coordinates from string to integer
    bbox_for_words = [[int(coordinate) for coordinate in bbox] for bbox in bbox_for_words]

    # and return a numpy array from the generated list
    bbox_for_words =  np.array(bbox_for_words)
    return bbox_for_words

def extract_bbox_from_image(path):
    """Given an image path, returns the coordinates for bboxes around the words
    :input: path
    :returns: numpy array
    """
    hocr = hocr_on_image(path)
    root = parse_hocr_string(hocr)
    bbox_for_words = extract_bbox_from_hocr(root)
    
    return bbox_for_words


def example_extract_bbox_for_image_without_arrows():
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped_arrows_removed.png")
    bbox_coordinates = extract_bbox_from_image(IMAGE_PATH)
    print("bbox coordinates: ", bbox_coordinates)

def main():
    example_extract_bbox_for_image_without_arrows()

if __name__ == '__main__':
    main()