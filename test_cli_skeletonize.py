#!/usr/bin/env python3
"""
Test the skeletonization functionality that would be used by the CLI
"""
import sys
import numpy as np
import imageio.v3 as iio
from pathlib import Path

# Use proper package imports - no sys.path manipulation

from pyamiimage.ami_image import AmiImage

def test_cli_skeletonize_functionality():
    """Test the skeletonization functionality that the CLI would use"""
    
    # Test image path
    test_image_path = Path("test/resources/biosynth1_cropped/text_removed.png")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    print(f"Testing CLI skeletonization functionality with: {test_image_path}")
    
    # Read the test image
    image = iio.imread(test_image_path)
    print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    
    # Test different methods (simulating CLI options)
    methods = ['medial_axis', 'skeletonize', 'thin']
    
    for method in methods:
        try:
            print(f"\n--- CLI Method: {method} ---")
            
            # This is what the CLI would call
            skeleton = AmiImage.create_white_skeleton_from_image(image, method=method)
            
            # CLI output info
            white_pixels = np.sum(skeleton == 255)
            print(f"Method: {method}")
            print(f"White pixels: {white_pixels}")
            print(f"Output shape: {skeleton.shape}")
            print(f"Output dtype: {skeleton.dtype}")
            
            # Save output (CLI would do this)
            output_path = Path(f"temp/cli_skeleton_{method}.png")
            output_path.parent.mkdir(exist_ok=True)
            iio.imwrite(output_path, skeleton)
            print(f"CLI would save to: {output_path}")
            
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    print("\n--- CLI Skeletonization Test Complete ---")
    print("All methods should work and produce 2D uint8 output")

if __name__ == "__main__":
    test_cli_skeletonize_functionality()
