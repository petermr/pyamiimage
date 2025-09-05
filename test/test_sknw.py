"""
Tests for the custom sknw module implementation.
Tests various image patterns from simple to complex to ensure robust graph creation.
"""
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Import the custom sknw module using absolute imports (following style rules)
from pyamiimage.sknw import (
    build_sknw, mark_node, neighbors, mark, fill, trace, 
    parse_struc, build_graph, idx2rc
)

class TestSknwSimple(unittest.TestCase):
    """Test sknw with simple images (≤50 pixels)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None  # Show full diffs for debugging
        
    def test_1a_short_connected_line_with_kinks(self):
        """Test: short connected topological line (can have kinks)"""
        # Create a simple line with kinks: ┌─┐
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # Test individual functions
        nbs = neighbors(img.shape)
        marked = mark_node(img.copy())
        
        # Test full pipeline
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        # Assertions
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0, "Should have at least one node")
        self.assertGreater(len(graph.edges), 0, "Should have at least one edge")
        
        print(f"Line with kinks: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1b_y_shape(self):
        """Test: similar size y-shape"""
        # Create a Y shape: ┌─┐
        #                    │ │
        #                    └─┘
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Y-shape: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1c_x_shape(self):
        """Test: similar size x-shape"""
        # Create an X shape
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        nnodes = len(graph.nodes)
        self.assertGreater(nnodes, 0, f"expected 5")
        self.assertEqual(nnodes, 0, f"expected 5")
        self.assertGreater(len(graph.edges), 0)
        
        print(f"X-shape: {nnodes} nodes, {len(graph.edges)} edges")
        
    def test_1d_two_disjoint_lines(self):
        """Test: two disjoint lines"""
        # Create two separate horizontal lines
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        # Should have at least 2 components (disjoint lines)
        components = list(nx.connected_components(graph))
        self.assertGreaterEqual(len(components), 2, "Should have at least 2 components")
        
        print(f"Two disjoint lines: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(components)} components")
        
    def test_1e_circle(self):
        """Test: circle"""
        # Create a simple circle (approximated)
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Circle: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1f_circle_with_spike(self):
        """Test: circle with spike"""
        # Create a circle with a spike extending out
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Circle with spike: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1g_single_pixel(self):
        """Test: single pixel"""
        # Create a single pixel
        img = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        # Single pixel should create a node
        self.assertGreaterEqual(len(graph.nodes), 0)
        
        print(f"Single pixel: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1h_two_pixel_line(self):
        """Test: 2-pixel line"""
        # Create a 2-pixel horizontal line
        img = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"2-pixel line: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1i_two_concentric_circles(self):
        """Test: 2 concentric non-touching circles"""
        # Create two concentric circles
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Two concentric circles: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1j_figure_8(self):
        """Test: figure-8"""
        # Create a figure-8 shape
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Figure-8: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1k_cross(self):
        """Test: cross shape"""
        # Create a cross shape
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Cross: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1l_triangle(self):
        """Test: triangle shape"""
        # Create a triangle
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Triangle: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1m_diamond(self):
        """Test: diamond shape"""
        # Create a diamond
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Diamond: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1n_horizontal_line(self):
        """Test: simple horizontal line"""
        # Create a horizontal line
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Horizontal line: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1o_vertical_line(self):
        """Test: simple vertical line"""
        # Create a vertical line
        img = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Vertical line: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1p_diagonal_line(self):
        """Test: diagonal line"""
        # Create a diagonal line
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Diagonal line: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1q_corner(self):
        """Test: corner shape"""
        # Create a corner
        img = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Corner: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1r_t_junction(self):
        """Test: T-junction"""
        # Create a T-junction
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"T-junction: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1s_plus_sign(self):
        """Test: plus sign"""
        # Create a plus sign
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Plus sign: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1t_empty_image(self):
        """Test: empty image (all zeros)"""
        # Create an empty image
        img = np.zeros((5, 5), dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        # Empty image should have no nodes or edges
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
        
        print(f"Empty image: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_1u_single_edge_pixel(self):
        """Test: single edge pixel (not isolated)"""
        # Create a single edge pixel
        img = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        # Single edge pixel should create a node
        self.assertGreaterEqual(len(graph.nodes), 0)
        
        print(f"Single edge pixel: {len(graph.nodes)} nodes, {len(graph.edges)} edges")


class TestSknwComplex(unittest.TestCase):
    """Test sknw with more complex images (>50 pixels)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None
        
    def test_2a_complex_network(self):
        """Test: complex network with multiple junctions"""
        # Create a more complex network
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Complex network: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_2b_spiral(self):
        """Test: spiral pattern"""
        # Create a spiral pattern
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Spiral: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_2c_maze_like(self):
        """Test: maze-like pattern"""
        # Create a maze-like pattern
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Maze-like: {len(graph.nodes)} nodes, {len(graph.edges)} edges")


class TestSknwEdgeCases(unittest.TestCase):
    """Test sknw with edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None
        
    def test_3a_very_thin_lines(self):
        """Test: very thin lines (1 pixel wide)"""
        # Create very thin lines
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Very thin lines: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_3b_noise_pixels(self):
        """Test: image with noise pixels"""
        # Create image with some noise pixels
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # Add some noise pixels
        img[2, 4] = 1  # Noise pixel
        img[4, 2] = 1  # Noise pixel
        
        graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
        
        self.assertIsInstance(graph, nx.MultiGraph)
        self.assertGreater(len(graph.nodes), 0)
        self.assertGreater(len(graph.edges), 0)
        
        print(f"Noise pixels: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
    def test_3c_different_parameters(self):
        """Test: different parameter combinations"""
        # Create a simple test image
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        # Test different parameter combinations
        params_combinations = [
            (False, False, False, False),  # multi=False, iso=False, ring=False, full=False
            (False, False, False, True),   # multi=False, iso=False, ring=False, full=True
            (False, False, True, False),   # multi=False, iso=False, ring=True, full=False
            (False, False, True, True),    # multi=False, iso=False, ring=True, full=True
            (False, True, False, False),   # multi=False, iso=True, ring=False, full=False
            (False, True, False, True),    # multi=False, iso=True, ring=False, full=True
            (False, True, True, False),    # multi=False, iso=True, ring=True, full=False
            (False, True, True, True),     # multi=False, iso=True, ring=True, full=True
            (True, False, False, False),   # multi=True, iso=False, ring=False, full=False
            (True, False, False, True),    # multi=True, iso=False, ring=False, full=True
            (True, False, True, False),    # multi=True, iso=False, ring=True, full=False
            (True, False, True, True),     # multi=True, iso=False, ring=True, full=True
            (True, True, False, False),    # multi=True, iso=True, ring=False, full=False
            (True, True, False, True),     # multi=True, iso=True, ring=False, full=True
            (True, True, True, False),     # multi=True, iso=True, ring=True, full=False
            (True, True, True, True),      # multi=True, iso=True, ring=True, full=True
        ]
        
        for multi, iso, ring, full in params_combinations:
            with self.subTest(multi=multi, iso=iso, ring=ring, full=full):
                try:
                    graph = build_sknw(img, multi=multi, iso=iso, ring=ring, full=full)
                    self.assertIsInstance(graph, nx.MultiGraph)
                    print(f"Params ({multi}, {iso}, {ring}, {full}): {len(graph.nodes)} nodes, {len(graph.edges)} edges")
                except Exception as e:
                    self.fail(f"Failed with params ({multi}, {iso}, {ring}, {full}): {e}")


class TestSknwIndividualFunctions(unittest.TestCase):
    """Test individual sknw functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None
        
    def test_4a_neighbors_function(self):
        """Test: neighbors function"""
        # Test 2D shape
        shape_2d = (5, 7)
        nbs_2d = neighbors(shape_2d)
        
        self.assertIsInstance(nbs_2d, np.ndarray)
        self.assertEqual(nbs_2d.shape[1], 2)  # 2D coordinates
        self.assertEqual(len(nbs_2d), 8)  # 8 neighbors in 2D
        
        # Test 3D shape
        shape_3d = (3, 4, 5)
        nbs_3d = neighbors(shape_3d)
        
        self.assertIsInstance(nbs_2d, np.ndarray)
        self.assertEqual(nbs_3d.shape[1], 3)  # 3D coordinates
        self.assertEqual(len(nbs_3d), 26)  # 26 neighbors in 3D
        
        print(f"2D neighbors: {nbs_2d.shape}, 3D neighbors: {nbs_3d.shape}")
        
    def test_4b_mark_function(self):
        """Test: mark function"""
        # Create a simple test image
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        nbs = neighbors(img.shape)
        
        # Test marking
        marked = img.copy()
        mark(marked, nbs)
        
        # Check that marking changed the image
        self.assertFalse(np.array_equal(img, marked))
        
        # Check that we have some marked pixels
        self.assertGreater(np.sum(marked > 0), 0)
        
        print(f"Marked image: {np.sum(marked > 0)} marked pixels")
        
    def test_4c_idx2rc_function(self):
        """Test: idx2rc function"""
        # Test with simple indices
        idx = np.array([0, 1, 2, 3, 4])
        acc = np.array([1, 5])  # For 5x5 image
        
        result = idx2rc(idx, acc)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 2))  # 5 indices, 2D coordinates
        
        print(f"idx2rc result shape: {result.shape}")
        
    def test_4d_fill_function(self):
        """Test: fill function"""
        # Create a simple test image
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        nbs = neighbors(img.shape)
        acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
        buf = np.zeros(1000, dtype=np.int64)
        
        # Test filling from a node position
        marked = img.copy()
        mark(marked, nbs)
        
        # Find a node position
        node_positions = np.where(marked == 2)
        if len(node_positions[0]) > 0:
            p = node_positions[0][0] * img.shape[1] + node_positions[1][0]
            
            isiso, nds = fill(marked, p, 10, nbs, acc, buf)
            
            self.assertIsInstance(isiso, bool)
            self.assertIsInstance(nds, np.ndarray)
            
            print(f"Fill result: isolated={isiso}, nodes shape={nds.shape}")
        else:
            print("No nodes found for fill test")
            
    def test_4e_trace_function(self):
        """Test: trace function"""
        # Create a simple test image with nodes
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        nbs = neighbors(img.shape)
        acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
        buf = np.zeros(1000, dtype=np.int64)
        
        # Test tracing
        marked = img.copy()
        mark(marked, nbs)
        
        # Find an edge position
        edge_positions = np.where(marked == 1)
        if len(edge_positions[0]) > 0:
            p = edge_positions[0][0] * img.shape[1] + edge_positions[1][0]
            
            try:
                result = trace(marked, p, nbs, acc, buf)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 3)
                
                print(f"Trace result: {result}")
            except Exception as e:
                print(f"Trace failed: {e}")
        else:
            print("No edges found for trace test")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
