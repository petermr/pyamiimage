"""
Tests for ImageViewer component.
"""

import unittest
import numpy as np
import tkinter as tk
from unittest.mock import Mock, patch

# Import the component to test
from pyamiimage.atpoe.gui.image_viewer import ImageViewer


class TestImageViewer(unittest.TestCase):
    """Test cases for ImageViewer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'root'):
            self.root.destroy()
            
    def test_image_viewer_creation(self):
        """Test that ImageViewer can be created."""
        viewer = ImageViewer(self.root, title="Test Viewer")
        self.assertIsNotNone(viewer)
        self.assertEqual(viewer.title, "Test Viewer")
        
    def test_set_image(self):
        """Test setting an image."""
        viewer = ImageViewer(self.root)
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Set the image
        viewer.set_image(test_image)
        
        # Check that image was set
        self.assertIsNotNone(viewer.image)
        np.testing.assert_array_equal(viewer.image, test_image)
        
    def test_get_image(self):
        """Test getting the current image."""
        viewer = ImageViewer(self.root)
        
        # Initially no image
        self.assertIsNone(viewer.get_image())
        
        # Set an image
        test_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        viewer.set_image(test_image)
        
        # Get the image
        retrieved_image = viewer.get_image()
        self.assertIsNotNone(retrieved_image)
        np.testing.assert_array_equal(retrieved_image, test_image)
        
    def test_zoom_controls(self):
        """Test zoom controls."""
        viewer = ImageViewer(self.root)
        
        # Set initial zoom
        initial_zoom = viewer.zoom_factor
        
        # Test zoom in
        viewer._zoom_in()
        self.assertGreater(viewer.zoom_factor, initial_zoom)
        
        # Test zoom out
        viewer._zoom_out()
        self.assertLess(viewer.zoom_factor, viewer.zoom_factor)
        
    def test_reset_view(self):
        """Test resetting the view."""
        viewer = ImageViewer(self.root)
        
        # Change zoom and pan
        viewer.zoom_factor = 2.0
        viewer.pan_x = 100
        viewer.pan_y = 100
        
        # Reset view
        viewer._reset_view()
        
        # Check reset values
        self.assertEqual(viewer.zoom_factor, 1.0)
        self.assertEqual(viewer.pan_x, 0)
        self.assertEqual(viewer.pan_y, 0)
        
    def test_viewport_overlay(self):
        """Test viewport overlay functionality."""
        viewer = ImageViewer(self.root)
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        viewer.set_image(test_image)
        
        # Set viewport overlay
        viewport = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
        viewer.set_viewport_overlay(viewport)
        
        # Check viewport was set
        self.assertEqual(viewer.viewport_overlay, viewport)
        
    def test_get_viewport(self):
        """Test getting viewport information."""
        viewer = ImageViewer(self.root)
        
        # Initially no image
        self.assertIsNone(viewer.get_viewport())
        
        # Set an image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        viewer.set_image(test_image)
        
        # Get viewport
        viewport = viewer.get_viewport()
        self.assertIsNotNone(viewport)
        self.assertIn('x', viewport)
        self.assertIn('y', viewport)
        self.assertIn('width', viewport)
        self.assertIn('height', viewport)
        
    def test_color_image_handling(self):
        """Test handling of color images."""
        viewer = ImageViewer(self.root)
        
        # Create a color test image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Set the color image
        viewer.set_image(color_image)
        
        # Check that image was set correctly
        self.assertIsNotNone(viewer.image)
        self.assertEqual(len(viewer.image.shape), 3)
        np.testing.assert_array_equal(viewer.image, color_image)
        
    def test_grayscale_image_handling(self):
        """Test handling of grayscale images."""
        viewer = ImageViewer(self.root)
        
        # Create a grayscale test image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Set the grayscale image
        viewer.set_image(gray_image)
        
        # Check that image was set correctly
        self.assertIsNotNone(viewer.image)
        self.assertEqual(len(viewer.image.shape), 2)
        np.testing.assert_array_equal(viewer.image, gray_image)


if __name__ == '__main__':
    unittest.main()




