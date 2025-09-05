#!/usr/bin/env python3
"""
Test script for skeletonization functionality with different methods
"""
import sys
import numpy as np
import imageio.v3 as iio
from pathlib import Path

# Use proper package imports - no sys.path manipulation

from pyamiimage.ami_image import AmiImage

def test_skeletonization_methods():
    """Test different skeletonization methods"""
    
    # Use a test image from the test resources
    test_image_path = Path("test/resources/biosynth1_cropped/text_removed.png")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    print(f"Testing skeletonization with image: {test_image_path}")
    
    # Read the test image
    image = iio.imread(test_image_path)
    print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    
    # Test different methods
    methods = ['medial_axis', 'skeletonize', 'thin']
    
    for method in methods:
        try:
            print(f"\n--- Testing {method} method ---")
            skeleton = AmiImage.create_white_skeleton_from_image(image, method=method)
            
            # Count white pixels
            white_pixels = np.sum(skeleton == 255)
            print(f"Method: {method}")
            print(f"White pixels: {white_pixels}")
            print(f"Output shape: {skeleton.shape}")
            print(f"Output dtype: {skeleton.dtype}")
            
            # Save output for comparison
            output_path = Path(f"temp/skeleton_{method}.png")
            output_path.parent.mkdir(exist_ok=True)
            iio.imwrite(output_path, skeleton)
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    print("\n--- Skeletonization test complete ---")

if __name__ == "__main__":
    test_skeletonization_methods()
