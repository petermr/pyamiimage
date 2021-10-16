import numpy as np
from skimage import io
from skimage import color
from skimage.color.colorconv import rgb2gray

class ImageProcessor():
    
    def __init__(self) -> None:
        self.image = None
        # setting a sample image for default path
        self.path = "../assets/purple_ocimum_basilicum.png"
    
    def load_image(self, path=None):
        if path is None:
            path = self.path
        self.image = io.imread(path)
        
    def to_gray(self):
        self.image = rgb2gray(self.image)

    def show_image(self):
        if self.image is None:
            self.load_image()
        self.to_gray()
        io.imshow(self.image)
        io.show()
        return True