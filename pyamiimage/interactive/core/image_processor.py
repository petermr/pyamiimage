"""
Image processing module for ATPOE skeletonization engine.

Handles image loading, preprocessing, and binarization with multiple threshold methods.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


class ImageProcessor:
    """Handles image loading, preprocessing, and binarization."""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.binary_image = None
        
    def load_image(self, image_path: Union[str, Path]) -> bool:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try OpenCV first for better performance
            self.original_image = cv2.imread(str(image_path))
            if self.original_image is not None:
                # Convert BGR to RGB for consistency
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                return True
        except Exception:
            pass
            
        try:
            # Fallback to PIL
            pil_image = Image.open(image_path)
            self.original_image = np.array(pil_image)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def get_image_info(self) -> Optional[dict]:
        """Get information about the loaded image."""
        if self.original_image is None:
            return None
            
        return {
            'shape': self.original_image.shape,
            'dtype': str(self.original_image.dtype),
            'min_value': np.min(self.original_image),
            'max_value': np.max(self.original_image),
            'mean_value': np.mean(self.original_image)
        }
    
    def convert_to_grayscale(self) -> np.ndarray:
        """Convert image to grayscale if it's not already."""
        if self.original_image is None:
            return None
            
        if len(self.original_image.shape) == 3:
            # Convert RGB to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.original_image.copy()
            
        self.processed_image = gray
        return gray
    
    def apply_preprocessing(self, blur_kernel: int = 5, median_kernel: int = 5) -> np.ndarray:
        """
        Apply preprocessing filters to the image.
        
        Args:
            blur_kernel: Gaussian blur kernel size
            median_kernel: Median filter kernel size
            
        Returns:
            Preprocessed image
        """
        if self.processed_image is None:
            self.convert_to_grayscale()
            
        if self.processed_image is None:
            return None
            
        # Apply Gaussian blur
        if blur_kernel > 1:
            self.processed_image = cv2.GaussianBlur(
                self.processed_image, (blur_kernel, blur_kernel), 0
            )
            
        # Apply median filter
        if median_kernel > 1:
            self.processed_image = cv2.medianBlur(
                self.processed_image, median_kernel
            )
            
        return self.processed_image
    
    def binarize_image(self, method: str = 'otsu', threshold: Optional[int] = None) -> np.ndarray:
        """
        Binarize the image using specified method.
        
        Args:
            method: Binarization method ('otsu', 'triangle', 'yen', 'manual')
            threshold: Manual threshold value (0-255)
            
        Returns:
            Binary image
        """
        if self.processed_image is None:
            self.convert_to_grayscale()
            
        if self.processed_image is None:
            return None
            
        if method == 'manual' and threshold is not None:
            # Manual threshold
            _, binary = cv2.threshold(
                self.processed_image, threshold, 255, cv2.THRESH_BINARY
            )
        elif method == 'otsu':
            # Otsu's method
            _, binary = cv2.threshold(
                self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == 'triangle':
            # Triangle method
            _, binary = cv2.threshold(
                self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
            )
        elif method == 'yen':
            # Yen's method (using skimage)
            try:
                from skimage.filters import threshold_yen
                thresh = threshold_yen(self.processed_image)
                binary = (self.processed_image > thresh).astype(np.uint8) * 255
            except ImportError:
                # Fallback to Otsu if skimage not available
                _, binary = cv2.threshold(
                    self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
        else:
            # Default to Otsu
            _, binary = cv2.threshold(
                self.processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
        self.binary_image = binary
        return binary
    
    def get_available_threshold_methods(self) -> list:
        """Get list of available threshold methods."""
        methods = ['otsu', 'triangle', 'manual']
        
        try:
            from skimage.filters import threshold_yen
            methods.append('yen')
        except ImportError:
            pass
            
        return methods
    
    def get_image(self, image_type: str = 'original') -> Optional[np.ndarray]:
        """
        Get image of specified type.
        
        Args:
            image_type: 'original', 'processed', or 'binary'
            
        Returns:
            Requested image or None
        """
        if image_type == 'original':
            return self.original_image
        elif image_type == 'processed':
            return self.processed_image
        elif image_type == 'binary':
            return self.binary_image
        else:
            return None




