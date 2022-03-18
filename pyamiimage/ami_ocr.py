import logging
from configparser import ConfigParser

import numpy as np
import pathlib

from pyamiimage.bbox import BBox
from pyamiimage.ami_image import AmiImage

CONFIG = 'config.ini'
config = ConfigParser()
config.read(CONFIG)
backend =  config['ocr']['backend']

# Backend Imports
if backend == 'easyocr':
    from pyamiimage.wrapper._for_easyocr import EasyOCRWrapper
    ocr_wrapper = EasyOCRWrapper()
elif backend == 'tesseract':
    from pyamiimage.wrapper._for_tesseract import PyTesseractWrapper
    ocr_wrapper = PyTesseractWrapper()
else:
    logging.error('OCR wrapper not found, check configuration file')
    exit()

class TextBox():
    '''
    Defines a TextBox object. The purpose of this object is to store location and string text of a textbox.

        Parameters:
            text (str): The text inside the textbox
            xy_ranges (list): xy ranges of the bounding box of the textbox
    '''
    def __init__(self, text, xy_ranges, baseline=0) -> None:
        self.text = text
        self.bbox = BBox(xy_ranges)
    
    def __repr__(self): 
        return f'Textbox({self.text}, {self.bbox.xy_ranges})'

    def __str__(self): 
        return f'text: {self.text} bbox: {self.bbox.xy_ranges}'

    def __eq__(self, other):
        if type(other) is not TextBox:
            return False
        return self.text == other.text and self.bbox.xy_ranges == other.bbox.xy_ranges

    def set_text(self, text):
        self.text = text

    def set_bbox(self, bbox):
        self.bbox = bbox

class AmiOCR:

    def __init__(self, image=None) -> None:
        '''Creates an OCR object from path and images, if both path and images are given, path takes precendence'''
        self.image = None
        self.textboxes = []
        if image is not None:
            try:
                self.set_image(image)
            except TypeError:
                logging.error('Could not initilize AmiOCR. Invalid image or Path.')
        else:
            logging.warning('AmiOCR initilized without image. No image or Path provided. Use set_image().')

    def set_image(self, image):
        '''
        Set image for AmiOCR object
        
            Parameters:
                image (Path or numpy.ndarray): Image or Path to run AmiOCR on
        '''
        if is_valid_image(image):
            self.image = image
        elif is_valid_path(image):
            self.image = AmiImage.read(image)
        else:
            raise TypeError('Only image (numpy.ndarray), path (str) or Path (pathlib.Path) allowed.')
    
    def get_textboxes(self, use_cache=True):
        if self.textboxes == [] or not use_cache:
            if self.image is not None:
                self.textboxes = AmiOCR.run_ocr_on_image(self.image)
            else:
                logging.error('No image to ocr, run set_image().')
                
        return self.textboxes

    @classmethod
    def run_ocr_on_image(cls, image):
        '''
        Given an image, runs ocr using the preferred backend.

            Parameters:
                image (Path or numpy.ndarray): The target image to run OCR on

            Returns:
                textboxes (list): List of TextBox objects
        '''
        logging.info('Running OCR on image, please wait...')
        textboxes = AmiOCR._generate_textboxes(ocr_wrapper.readtext(image))
        logging.info('Done')
        return textboxes

    
    def _generate_textboxes(ocr_wrapper_output):
        '''
        Generates list of textboxes from ocr_wrapper output
        
            Parameters:
                ocr_wrapper_output (list): raw data from ocr engine

            Returns:
                textboxes (list): List of TextBox objects
        '''
        textboxes = []
        for row in ocr_wrapper_output:
            xy_range_of_bbox = row[0]
            text = row[1]
            confidence = row[2]
            textboxes.append(TextBox(text, xy_range_of_bbox))
        return textboxes



##### Utilities #####

def is_valid_image(object):
    '''
    Given an object checks if it is a valid image or not
    
        Parameters:
        object (any): Object to be tested

        Returns:
        (bool): True if object is a valid image
    '''
    if isinstance(object, np.ndarray):
        if len(object.shape) > 1 and len(object.shape) <=3:
            return True
    return False

def is_valid_path(object):
    '''
    Given an object checks if it a valid path or not
    
        Parameters:
        object (any): Object to be tested
        
        Returns:
        (bool): True if object is a valid path
    '''
    if isinstance(object, str):
        # If path is a string convert into a Path object
        object = pathlib.Path(object)
    
    if isinstance(object, pathlib.PurePath):
        # check if the path exists
        return object.exists()