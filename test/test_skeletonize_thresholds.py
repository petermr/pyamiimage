#!/usr/bin/env python3
"""
Test different thresholding approaches to improve skeleton quality
"""
import unittest
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from skimage import filters

from pyamiimage.ami_image import AmiImage
from pyamiimage.ami_image import AmiImageReader


class TestSkeletonizeThresholds(unittest.TestCase):
    """Test skeletonization with different thresholding approaches"""

    def test_threshold_experiments(self):
        """Test different thresholding approaches to improve skeleton quality"""
        
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
        
        # 3. Check if black is the signal and invert if needed
        black_pixels = np.sum(image == 0)
        white_pixels = np.sum(image == 255)
        
        print(f"Black pixels: {black_pixels}, White pixels: {white_pixels}")
        
        if black_pixels > white_pixels:
            # Black is the signal, invert to make it white for skeletonization
            image = AmiImage.create_inverted_image(image)
            print("Inverted image: black signal → white for skeletonization")
        
        # 4. Create output directory
        output_dir = Path("temp/skeletonize_thresholds")
        output_dir.mkdir(exist_ok=True)
        
        # 5. Test different thresholding approaches
        threshold_experiments = [
            {
                'name': 'otsu',
                'description': 'Otsu thresholding (automatic)',
                'threshold_func': lambda img: filters.threshold_otsu(img)
            },
            {
                'name': 'triangle',
                'description': 'Triangle thresholding',
                'threshold_func': lambda img: filters.threshold_triangle(img)
            },
            {
                'name': 'yen',
                'description': 'Yen thresholding',
                'threshold_func': lambda img: filters.threshold_yen(img)
            },
            {
                'name': 'isodata',
                'description': 'Isodata thresholding',
                'threshold_func': lambda img: filters.threshold_isodata(img)
            },
            {
                'name': 'li',
                'description': 'Li thresholding',
                'threshold_func': lambda img: filters.threshold_li(img)
            },
            {
                'name': 'mean',
                'description': 'Mean thresholding',
                'threshold_func': lambda img: filters.threshold_mean(img)
            },
            {
                'name': 'minimum',
                'description': 'Minimum thresholding',
                'threshold_func': lambda img: filters.threshold_minimum(img)
            }
        ]
        
        # 6. Test each thresholding approach
        for exp in threshold_experiments:
            try:
                print(f"\n--- Experiment: {exp['name']} ---")
                print(f"Description: {exp['description']}")
                
                # Calculate threshold
                threshold = exp['threshold_func'](image)
                print(f"Calculated threshold: {threshold:.1f}")
                
                # Apply threshold to create binary image
                binary_image = (image > threshold).astype(np.uint8) * 255
                
                # Count pixels
                binary_white = np.sum(binary_image == 255)
                binary_black = np.sum(binary_image == 0)
                print(f"Binary image - White: {binary_white:,}, Black: {binary_black:,}")
                
                # Save binary image for comparison
                binary_path = output_dir / f"binary_{exp['name']}.png"
                iio.imwrite(binary_path, binary_image)
                print(f"Binary image saved to: {binary_path}")
                
                # Skeletonize the binary image
                skeleton = AmiImage.create_white_skeleton_from_image(binary_image, method='medial_axis')
                
                # 7. Output the skeletonized image
                output_path = output_dir / f"skeleton_{exp['name']}.png"
                iio.imwrite(output_path, skeleton)
                
                # 8. Analyze skeleton results
                white_pixel_count = np.sum(skeleton == 255)
                total_pixels = skeleton.size
                skeleton_density = white_pixel_count / total_pixels * 100
                
                print(f"Skeleton white pixels: {white_pixel_count:,}")
                print(f"Skeleton density: {skeleton_density:.2f}%")
                print(f"Skeleton saved to: {output_path}")
                
            except Exception as e:
                print(f"Experiment {exp['name']} failed: {e}")
                continue
        
        # 9. Test manual threshold adjustments around Otsu
        print(f"\n--- Manual threshold adjustments around Otsu ---")
        otsu_threshold = filters.threshold_otsu(image)
        
        for offset in [-20, -10, -5, 0, 5, 10, 20]:
            try:
                manual_threshold = otsu_threshold + offset
                print(f"\n--- Manual threshold: {manual_threshold:.1f} (Otsu + {offset}) ---")
                
                # Apply manual threshold
                binary_image = (image > manual_threshold).astype(np.uint8) * 255
                
                # Count pixels
                binary_white = np.sum(binary_image == 255)
                binary_black = np.sum(binary_image == 0)
                print(f"Binary image - White: {binary_white:,}, Black: {binary_black:,}")
                
                # Save binary image
                binary_path = output_dir / f"binary_manual_{offset:+d}.png"
                iio.imwrite(binary_path, binary_image)
                
                # Skeletonize
                skeleton = AmiImage.create_white_skeleton_from_image(binary_image, method='medial_axis')
                
                # Save skeleton
                output_path = output_dir / f"skeleton_manual_{offset:+d}.png"
                iio.imwrite(output_path, skeleton)
                
                # Analyze
                white_pixel_count = np.sum(skeleton == 255)
                skeleton_density = white_pixel_count / total_pixels * 100
                print(f"Skeleton white pixels: {white_pixel_count:,}")
                print(f"Skeleton density: {skeleton_density:.2f}%")
                
            except Exception as e:
                print(f"Manual threshold {offset:+d} failed: {e}")
                continue
        
        print(f"\n--- All threshold experiments complete. Outputs saved to: {output_dir} ---")
        print("Compare the binary and skeleton images to find the best threshold!")


if __name__ == "__main__":
    unittest.main()
