import numpy as np
from skimage import io

class Image():
    
    def __init__(self) -> None:
        self.image = None
    
    def image_import(self, path=None):
        if path is None:
            path = self.path
        self.image = io.imread(path)
        