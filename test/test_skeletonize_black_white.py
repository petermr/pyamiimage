#!/usr/bin/env python3
"""
Test skeletonization of black and white images with proper monochrome conversion
"""
import unittest
import numpy as np
import imageio.v3 as iio
from pathlib import Path

from pyamiimage.ami_image import AmiImage
from pyamiimage.ami_image import AmiImageReader


class TestSkeletonizeBlackWhite(unittest.TestCase):
    """Test skeletonization of black and white images"""

    def test_skeletonize_black_white_image(self):
        """Test skeletonization of black and white image with proper monochrome conversion"""
        
        # 1. Read the test image
        input_path = Path("test/resources/black-and-white.png")
        if not input_path.exists():
            self.skipTest(f"Test image not found: {input_path}")
        
        image = AmiImageReader.read_image(input_path)
        print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        
        # 2. Convert to monochrome (ensure 2D grayscale)
        if len(image.shape) > 2:
            image = AmiImage.create_grayscale_from_image(image)
            print(f"Converted to grayscale: {image.shape}")
        
        # 3. Create output directory
        output_dir = Path("temp/skeletonize_black_white")
        output_dir.mkdir(exist_ok=True)
        
        # 4. Skeletonize to represent the black (invert if needed)
        # Check if black is the signal or background
        black_pixels = np.sum(image == 0)
        white_pixels = np.sum(image == 255)
        
        print(f"Black pixels: {black_pixels}, White pixels: {white_pixels}")
        
        if black_pixels > white_pixels:
            # Black is the signal, invert to make it white for skeletonization
            image = AmiImage.create_inverted_image(image)
            print("Inverted image: black signal → white for skeletonization")
        
        # 5. Test all three skeletonization methods
        methods = ['medial_axis', 'skeletonize', 'thin']
        
        for method in methods:
            try:
                print(f"\n--- Testing {method} method ---")
                
                skeleton = AmiImage.create_white_skeleton_from_image(image, method=method)
                
                # 6. Output the skeletonized image
                output_path = output_dir / f"skeleton_{method}.png"
                iio.imwrite(output_path, skeleton)
                
                # 7. Validate output
                self.assertEqual(len(skeleton.shape), 2, "Output should be 2D")
                self.assertEqual(skeleton.dtype, np.uint8, "Output should be uint8")
                self.assertEqual(np.max(skeleton), 255, "Skeleton should have white pixels")
                
                white_pixel_count = np.sum(skeleton == 255)
                print(f"Method {method}: {white_pixel_count} white pixels, saved to {output_path}")
                
                # Additional validation
                self.assertGreater(white_pixel_count, 0, "Skeleton should have some white pixels")
                
            except Exception as e:
                self.fail(f"Method {method} failed: {e}")
        
        print(f"\n--- Test complete. Outputs saved to: {output_dir} ---")


if __name__ == "__main__":
    unittest.main()
