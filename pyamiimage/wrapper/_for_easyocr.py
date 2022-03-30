import easyocr

class EasyOCRWrapper:
    def __init__(self, lang='en'):
        self.reader = easyocr.Reader([lang])

    def readtext(self, image):
        """
        Given an image returns all the detected text, bounding boxes and confidence in a 2D array
        The order of columns per entry is: bounding box(xy_range), text, confidence
        
            Parameters:
                image (numpy array or filepath(str)): image to be read
        
            Returns:
                data (list): 2D list of detected text
        """
        data = self.reader.readtext(image)
        data = [[self._create_xy_range_from_bbox(entry[0]), entry[1], entry[2]] for entry in data]
        return data

    # Internal Functions
    def _create_xy_range_from_bbox(self, bbox):
        """bbox is in format [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]"""
        xy_range = [[bbox[0][0], bbox[2][0]], [bbox[0][1], bbox[2][1]]]
        return xy_range