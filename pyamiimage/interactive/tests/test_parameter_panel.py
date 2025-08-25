"""
Tests for ParameterPanel component.
"""

import unittest
import tkinter as tk
from unittest.mock import Mock

# Import the component to test
from pyamiimage.interactive.gui.parameter_panel import ParameterPanel


class TestParameterPanel(unittest.TestCase):
    """Test cases for ParameterPanel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'root'):
            self.root.destroy()
            
    def test_parameter_panel_creation(self):
        """Test that ParameterPanel can be created."""
        panel = ParameterPanel(self.root)
        self.assertIsNotNone(panel)
        
    def test_default_parameters(self):
        """Test default parameter values."""
        panel = ParameterPanel(self.root)
        
        expected_defaults = {
            'skeleton_method': 'medial_axis',
            'gaussian_blur': 5,
            'median_filter': 5,
            'threshold_method': 'otsu',
            'manual_threshold': 128,
            'branch_threshold': 10,
            'min_path_length': 5,
            'min_node_size': 3
        }
        
        actual_params = panel.get_parameters()
        for key, expected_value in expected_defaults.items():
            self.assertEqual(actual_params[key], expected_value)
            
    def test_parameter_change_callback(self):
        """Test that parameter changes trigger callback."""
        mock_callback = Mock()
        panel = ParameterPanel(self.root, callback=mock_callback)
        
        # Change a parameter
        panel.method_var.set('skeletonize')
        
        # Check that callback was called
        mock_callback.assert_called()
        
    def test_get_parameters(self):
        """Test getting current parameters."""
        panel = ParameterPanel(self.root)
        
        params = panel.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn('skeleton_method', params)
        self.assertIn('gaussian_blur', params)
        self.assertIn('threshold_method', params)
        
    def test_set_parameters(self):
        """Test setting parameters from dictionary."""
        panel = ParameterPanel(self.root)
        
        new_params = {
            'skeleton_method': 'thin',
            'gaussian_blur': 10,
            'manual_threshold': 200
        }
        
        panel.set_parameters(new_params)
        
        # Check that parameters were updated
        current_params = panel.get_parameters()
        for key, value in new_params.items():
            self.assertEqual(current_params[key], value)
            
    def test_set_enabled(self):
        """Test enabling/disabling parameter controls."""
        panel = ParameterPanel(self.root)
        
        # Initially enabled
        self.assertEqual(panel.apply_button.cget('state'), 'normal')
        
        # Disable
        panel.set_enabled(False)
        self.assertEqual(panel.apply_button.cget('state'), 'disabled')
        
        # Re-enable
        panel.set_enabled(True)
        self.assertEqual(panel.apply_button.cget('state'), 'normal')
        
    def test_reset_to_defaults(self):
        """Test resetting parameters to defaults."""
        panel = ParameterPanel(self.root)
        
        # Change some parameters
        panel.method_var.set('thin')
        panel.blur_var.set(15)
        panel.manual_thresh_var.set(100)
        
        # Reset to defaults
        panel.reset_to_defaults()
        
        # Check that parameters were reset
        params = panel.get_parameters()
        self.assertEqual(params['skeleton_method'], 'medial_axis')
        self.assertEqual(params['gaussian_blur'], 5)
        self.assertEqual(params['manual_threshold'], 128)
        
    def test_apply_button_state(self):
        """Test apply button state changes."""
        panel = ParameterPanel(self.root)
        
        # Initially disabled
        self.assertEqual(panel.apply_button.cget('state'), 'disabled')
        
        # Change a parameter to enable it
        panel.method_var.set('skeletonize')
        self.assertEqual(panel.apply_button.cget('state'), 'normal')
        
        # Apply parameters to disable it
        panel._apply_parameters()
        self.assertEqual(panel.apply_button.cget('state'), 'disabled')
        
    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        panel = ParameterPanel(self.root)
        
        # Test that parameters are within expected ranges
        params = panel.get_parameters()
        
        # Gaussian blur should be 1-21
        self.assertGreaterEqual(params['gaussian_blur'], 1)
        self.assertLessEqual(params['gaussian_blur'], 21)
        
        # Median filter should be 1-21
        self.assertGreaterEqual(params['median_filter'], 1)
        self.assertLessEqual(params['median_filter'], 21)
        
        # Manual threshold should be 0-255
        self.assertGreaterEqual(params['manual_threshold'], 0)
        self.assertLessEqual(params['manual_threshold'], 255)
        
        # Branch threshold should be 1-50
        self.assertGreaterEqual(params['branch_threshold'], 1)
        self.assertLessEqual(params['branch_threshold'], 50)
        
    def test_method_selection(self):
        """Test skeletonization method selection."""
        panel = ParameterPanel(self.root)
        
        # Check available methods
        expected_methods = ['medial_axis', 'skeletonize', 'thin']
        actual_methods = panel.method_var.cget('values')
        self.assertEqual(actual_methods, expected_methods)
        
        # Test setting different methods
        for method in expected_methods:
            panel.method_var.set(method)
            self.assertEqual(panel.method_var.get(), method)
            
    def test_threshold_method_selection(self):
        """Test threshold method selection."""
        panel = ParameterPanel(self.root)
        
        # Check available threshold methods
        expected_methods = ['otsu', 'triangle', 'yen', 'manual']
        actual_methods = panel.thresh_method_var.cget('values')
        self.assertEqual(actual_methods, expected_methods)
        
        # Test setting different methods
        for method in expected_methods:
            panel.thresh_method_var.set(method)
            self.assertEqual(panel.thresh_method_var.get(), method)


if __name__ == '__main__':
    unittest.main()




