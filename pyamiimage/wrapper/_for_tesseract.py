from lxml import etree as et

import pytesseract

class PyTesseractWrapper:
    def __init__(self):
        # Tesseract Config
        self.lang = 'eng'
        self.psm = '11'
        self.oem = '3'

    def readtext(self, image):
        """
        Given an image returns all the detected text, bounding boxes and confidence in a 2D array
        The order of columns per entry is: bounding box(xy_range), text, confidence
        
            Parameters:
                image (numpy array or filepath(str)): image to be read
        
            Returns:
                data (list): 2D list of detected text
        """
        hocr = self.hocr_from_image(image)       
        data = self.parse_hocr_tree(hocr)
        return data


    def parse_hocr_tree(self, hocr_element):
        '''
        to extract bbox coordinates from html
        :param hocr_element: lxml.etree.Element object
        :returns: tuple of (list of bboxes, list of words)
        numpy array has the format: [x1, y1, x2, y2]
        where values corresponding to the bounding box are
        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2    
        '''

        textboxes = []

        # Each of the bbox coordinates are in span
        # iterdescendants('span') iterates over all the span elements in the tree
        for child in hocr_element.iterdescendants('span'):
            # the class of span for bounding box around each word is 'ocrx_word'
            if child.attrib['class'] == 'ocrx_word':
                # coordinates for bbox has the following format: 'bbox 333 74 471 102; x_wconf 76'
                # we are only interested in the coordinates, so we split at ; and append the first part
                bbox_string = child.attrib['title'].split(';')[0]
                xy_range = self._create_xy_range_from_bbox_string(bbox_string)
                textboxes.append([xy_range, child.text, 1]) # TODO need to extract w_conf 
        return textboxes

    def _create_xy_range_from_bbox_string(self, bbox_string):
        # now we have a list with string elenments like 'bbox 333 74 471 102'
        # first we convert the string into a list, and then remove 'bbox', leaving just the coordinate
        bbox_list = bbox_string.split()[1:]

        # finally we convert the coordinates from string to integer
        bbox_list = [int(coordinate) for coordinate in bbox_list]
        # bbox_list = [x1, y1, x2, y2]
        xy_range = [[bbox_list[0],bbox_list[2]], [bbox_list[1], bbox_list[3]]]
        return xy_range

    def hocr_from_image(self, image):
        '''
        Runs Tesseract on the given image or image path 

            Parameters:
                image (str or numpy): Image to run tesseract on

            Returns:
                hocr (ElementTree object): html tree containing Tesseract output
        '''
        hocr_string = pytesseract.image_to_pdf_or_hocr(image, extension='hocr', config=self.psm)
        hocr = self.parse_hocr_string(hocr_string)
        return hocr

    def parse_hocr_string(self, hocr_string):
        """Parses hocr output in string format as a tree using lxml
        :input: hocr html as string
        :returns: root of hocr as an object of lxml.etree.Element class
        """
        parser = et.HTMLParser()
        root = et.HTML(hocr_string, parser)
        return root