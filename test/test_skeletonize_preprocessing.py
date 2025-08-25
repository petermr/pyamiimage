#!/usr/bin/env python3
"""
Test different preprocessing approaches to improve skeleton quality
"""
import unittest
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from skimage import filters, morphology

from pyamiimage.ami_image import AmiImage
from pyamiimage.ami_image import AmiImageReader


class TestSkeletonizePreprocessing(unittest.TestCase):
    """Test skeletonization with different preprocessing approaches"""

    def test_preprocessing_experiments(self):
        """Test different preprocessing approaches to improve skeleton quality"""
        
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
        output_dir = Path("temp/skeletonize_preprocessing")
        output_dir.mkdir(exist_ok=True)
        
        # 5. Run different preprocessing experiments
        experiments = [
            {
                'name': 'original',
                'description': 'No preprocessing - direct skeletonization',
                'preprocess': lambda img: img
            },
            {
                'name': 'sharpen',
                'description': 'Sharpen image before skeletonization',
                'preprocess': lambda img: filters.unsharp_mask(img, radius=1, amount=2)
            },
            {
                'name': 'dilate_erode',
                'description': 'Dilate then erode (closing) before skeletonization',
                'preprocess': lambda img: morphology.binary_closing(img > 127, morphology.disk(2))
            },
            {
                'name': 'erode_dilate',
                'description': 'Erode then dilate (opening) before skeletonization',
                'preprocess': lambda img: morphology.binary_opening(img > 127, morphology.disk(1))
            },
            {
                'name': 'smooth_sharpen',
                'description': 'Smooth then sharpen before skeletonization',
                'preprocess': lambda img: filters.unsharp_mask(
                    filters.gaussian(img, sigma=1), radius=1, amount=1.5
                )
            }
        ]
        
        # 6. Test each preprocessing approach with medial_axis
        for exp in experiments:
            try:
                print(f"\n--- Experiment: {exp['name']} ---")
                print(f"Description: {exp['description']}")
                
                # Apply preprocessing
                processed_image = exp['preprocess'](image)
                
                # Ensure binary for skeletonization
                if processed_image.dtype == bool:
                    processed_image = processed_image.astype(np.uint8) * 255
                elif processed_image.dtype == float:
                    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
                
                # Skeletonize with medial_axis (best method from previous tests)
                skeleton = AmiImage.create_white_skeleton_from_image(processed_image, method='medial_axis')
                
                # 7. Output the skeletonized image
                output_path = output_dir / f"skeleton_{exp['name']}.png"
                iio.imwrite(output_path, skeleton)
                
                # 8. Analyze results
                white_pixel_count = np.sum(skeleton == 255)
                total_pixels = skeleton.size
                skeleton_density = white_pixel_count / total_pixels * 100
                
                print(f"White pixels: {white_pixel_count:,}")
                print(f"Skeleton density: {skeleton_density:.2f}%")
                print(f"Saved to: {output_path}")
                
                # 9. Save intermediate processed image for comparison
                if exp['name'] != 'original':
                    processed_path = output_dir / f"processed_{exp['name']}.png"
                    iio.imwrite(processed_path, processed_image)
                    print(f"Processed image saved to: {processed_path}")
                
            except Exception as e:
                print(f"Experiment {exp['name']} failed: {e}")
                continue
        
        print(f"\n--- All experiments complete. Outputs saved to: {output_dir} ---")
        print("Compare the skeleton images to see which preprocessing works best!")


if __name__ == "__main__":
    unittest.main()





