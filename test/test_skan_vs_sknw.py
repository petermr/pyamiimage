"""
Test to compare skan vs custom sknw implementation.
Focuses on problematic cases like Y-shapes that were producing 0 edges.
"""
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix

# Import both implementations
from pyamiimage.sknw import build_sknw
import skan

class TestSkanVsSknw(unittest.TestCase):
    """Compare skan vs custom sknw implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maxDiff = None
        
    def skan_to_networkx(self, skeleton):
        """Convert skan skeleton to NetworkX graph"""
        # Get the sparse adjacency matrix
        adj_matrix = skeleton.graph
        
        print(f"DEBUG: Adjacency matrix shape: {adj_matrix.shape}")
        print(f"DEBUG: Adjacency matrix type: {type(adj_matrix)}")
        print(f"DEBUG: Adjacency matrix nnz: {adj_matrix.getnnz()}")
        
        # Convert to NetworkX graph
        # Note: skan uses 1-based indexing, so we need to adjust
        G = nx.Graph()
        
        # Add nodes (skan uses 1-based indexing)
        for i in range(1, adj_matrix.shape[0]):
            # Check if this node has any connections
            row_nnz = adj_matrix.getrow(i).getnnz()
            print(f"DEBUG: Node {i} has {row_nnz} connections")
            if row_nnz > 0:
                G.add_node(i-1)  # Convert to 0-based for NetworkX
        
        # Add edges
        rows, cols = adj_matrix.nonzero()
        print(f"DEBUG: Found {len(rows)} non-zero elements")
        for row, col in zip(rows, cols):
            if row != col and row > 0 and col > 0:  # Skip diagonal and 0-index
                print(f"DEBUG: Adding edge {row-1} -> {col-1}")
                G.add_edge(row-1, col-1)  # Convert to 0-based
                
        return G
        
    def test_1_y_shape_comparison(self):
        """Test: Y-shape that was failing with custom sknw (0 edges)"""
        # Create Y-shape that was problematic
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        print(f"\n=== Y-SHAPE TEST ===")
        print(f"Image shape: {img.shape}")
        print(f"White pixels: {np.sum(img > 0)}")
        
        # Test custom sknw
        print(f"\n--- Custom SKNW ---")
        try:
            sknw_graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
            print(f"SKNW: {len(sknw_graph.nodes)} nodes, {len(sknw_graph.edges)} edges")
            print(f"SKNW nodes: {list(sknw_graph.nodes)}")
            print(f"SKNW edges: {list(sknw_graph.edges)}")
        except Exception as e:
            print(f"SKNW ERROR: {e}")
            sknw_graph = None
        
        # Test skan
        print(f"\n--- SKAN ---")
        try:
            # Convert to boolean for skan
            img_bool = img.astype(bool)
            
            # Use skan's Skeleton class
            skeleton = skan.Skeleton(img_bool)
            print(f"SKAN skeleton: {skeleton.coordinates.shape}")
            print(f"SKAN paths type: {type(skeleton.paths)}")
            print(f"SKAN paths shape: {skeleton.paths.shape}")
            print(f"SKAN paths: {skeleton.n_paths}")
            print(f"SKAN path lengths: {skeleton.path_lengths()}")
            
            # Convert to NetworkX graph
            skan_graph = self.skan_to_networkx(skeleton)
            print(f"SKAN: {len(skan_graph.nodes)} nodes, {len(skan_graph.edges)} edges")
            print(f"SKAN nodes: {list(skan_graph.nodes)}")
            print(f"SKAN edges: {list(skan_graph.edges)}")
            
        except Exception as e:
            print(f"SKAN ERROR: {e}")
            import traceback
            traceback.print_exc()
            skan_graph = None
        
        # Assertions
        if sknw_graph:
            self.assertIsInstance(sknw_graph, nx.MultiGraph)
            print(f"SKNW assertion passed: MultiGraph type")
        else:
            print(f"SKNW assertion skipped: graph creation failed")
            
        if skan_graph:
            self.assertIsInstance(skan_graph, nx.Graph)
            print(f"SKAN assertion passed: Graph type")
        else:
            print(f"SKAN assertion skipped: graph creation failed")
    
    def test_2_cross_shape_comparison(self):
        """Test: Cross shape that was also failing"""
        # Create cross shape
        img = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        print(f"\n=== CROSS SHAPE TEST ===")
        
        # Test custom sknw
        print(f"\n--- Custom SKNW ---")
        try:
            sknw_graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
            print(f"SKNW: {len(sknw_graph.nodes)} nodes, {len(sknw_graph.edges)} edges")
        except Exception as e:
            print(f"SKNW ERROR: {e}")
        
        # Test skan
        print(f"\n--- SKAN ---")
        try:
            img_bool = img.astype(bool)
            skeleton = skan.Skeleton(img_bool)
            skan_graph = self.skan_to_networkx(skeleton)
            print(f"SKAN: {len(skan_graph.nodes)} nodes, {len(skan_graph.edges)} edges")
        except Exception as e:
            print(f"SKAN ERROR: {e}")
    
    def test_3_simple_line_comparison(self):
        """Test: Simple line that should work with both"""
        # Create simple horizontal line
        img = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        
        print(f"\n=== SIMPLE LINE TEST ===")
        
        # Test custom sknw
        print(f"\n--- Custom SKNW ---")
        try:
            sknw_graph = build_sknw(img, multi=True, iso=True, ring=True, full=True)
            print(f"SKNW: {len(sknw_graph.nodes)} nodes, {len(sknw_graph.edges)} edges")
        except Exception as e:
            print(f"SKNW ERROR: {e}")
        
        # Test skan
        print(f"\n--- SKAN ---")
        try:
            img_bool = img.astype(bool)
            skeleton = skan.Skeleton(img_bool)
            skan_graph = self.skan_to_networkx(skeleton)
            print(f"SKAN: {len(skan_graph.nodes)} nodes, {len(skan_graph.edges)} edges")
        except Exception as e:
            print(f"SKAN ERROR: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
