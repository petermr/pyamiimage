import numpy as np
from skimage import io
from skimage import color
from skimage.color.colorconv import rgb2gray

"""
The ImageProcessor class is current in development by PMR and Anuv for preprocessing images
as a part of the project: "Extraction of biosynthetic pathway from images"

Some part of the code has been copied from ImageLib written by PMR for openDiagram
We decided against continuing development on openDiagram library because the size
of the repository exceeded 2 gigabytes

The ImageLib module has been included in this repository for testing and reference 
"""

class ImageProcessor():
    # setting a sample image for default path
    DEFAULT_PATH = "assets/purple_ocimum_basilicum.png"

    def __init__(self) -> None:
        self.image = None

    def load_image(self, path):
        """
        loads image with io.imread
        resets self.image
        input: path
        returns: None if path is None
        """
        self.image = None
        if path is not None:
            self.image = io.imread(path)
        return self.image
        
    def to_gray(self):
        """convert existing self.image to grayscale
        uses rgb2gray from skimage.color.colorconv
        """
        self.image_gray = None
        if self.image is not None:
            self.image_gray = rgb2gray(self.image)
        return self.image_gray

    def show_image(self):
        """
        Shows self.image in a seperate window
        """
        if self.image is None:
            self.load_image()
        self.to_gray()
        io.imshow(self.image)
        io.show()
        return True