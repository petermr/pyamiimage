"""
Tests for GraphViewer component.
"""

import unittest
import tkinter as tk
import networkx as nx
from unittest.mock import Mock, patch

# Import the component to test
from pyamiimage.interactive.gui.graph_viewer import GraphViewer


class TestGraphViewer(unittest.TestCase):
    """Test cases for GraphViewer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'root'):
            self.root.destroy()
            
    def test_graph_viewer_creation(self):
        """Test that GraphViewer can be created."""
        viewer = GraphViewer(self.root)
        self.assertIsNotNone(viewer)
        
    def test_initial_state(self):
        """Test initial state of the viewer."""
        viewer = GraphViewer(self.root)
        
        # Initially no graph
        self.assertIsNone(viewer.current_graph)
        
        # Buttons should be disabled
        self.assertEqual(viewer.export_button.cget('state'), 'disabled')
        self.assertEqual(viewer.analyze_button.cget('state'), 'disabled')
        
    def test_set_graph(self):
        """Test setting a graph."""
        viewer = GraphViewer(self.root)
        
        # Create a simple test graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        test_graph.add_edge(2, 3)
        
        # Set the graph
        viewer.set_graph(test_graph)
        
        # Check that graph was set
        self.assertIsNotNone(viewer.current_graph)
        self.assertEqual(viewer.current_graph, test_graph)
        
        # Buttons should be enabled
        self.assertEqual(viewer.export_button.cget('state'), 'normal')
        self.assertEqual(viewer.analyze_button.cget('state'), 'normal')
        
    def test_get_graph(self):
        """Test getting the current graph."""
        viewer = GraphViewer(self.root)
        
        # Initially no graph
        self.assertIsNone(viewer.get_graph())
        
        # Set a graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        viewer.set_graph(test_graph)
        
        # Get the graph
        retrieved_graph = viewer.get_graph()
        self.assertIsNotNone(retrieved_graph)
        self.assertEqual(retrieved_graph, test_graph)
        
    def test_clear(self):
        """Test clearing the graph viewer."""
        viewer = GraphViewer(self.root)
        
        # Set a graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        viewer.set_graph(test_graph)
        
        # Clear
        viewer.clear()
        
        # Check that graph was cleared
        self.assertIsNone(viewer.current_graph)
        
        # Buttons should be disabled
        self.assertEqual(viewer.export_button.cget('state'), 'disabled')
        self.assertEqual(viewer.analyze_button.cget('state'), 'disabled')
        
    def test_display_update_with_graph(self):
        """Test that display updates when graph is set."""
        viewer = GraphViewer(self.root)
        
        # Create a test graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        test_graph.add_edge(2, 3)
        test_graph.add_edge(3, 1)  # Creates a cycle
        
        # Set the graph
        viewer.set_graph(test_graph)
        
        # Check that display was updated
        self.assertEqual(viewer.nodes_label.cget('text'), 'Nodes: 3')
        self.assertEqual(viewer.edges_label.cget('text'), 'Edges: 3')
        
        # Check connectivity
        self.assertEqual(viewer.components_label.cget('text'), 'Components: 1')
        self.assertEqual(viewer.is_connected_label.cget('text'), 'Connected: Yes')
        
        # Check topology
        self.assertEqual(viewer.avg_degree_label.cget('text'), 'Avg Degree: 2.00')
        self.assertEqual(viewer.max_degree_label.cget('text'), 'Max Degree: 2')
        
        # Check node analysis
        self.assertEqual(viewer.end_nodes_label.cget('text'), 'End Nodes: 0')
        self.assertEqual(viewer.branch_nodes_label.cget('text'), 'Branch Nodes: 0')
        self.assertEqual(viewer.junction_nodes_label.cget('text'), 'Junction Nodes: 3')
        
    def test_display_update_with_disconnected_graph(self):
        """Test display update with disconnected graph."""
        viewer = GraphViewer(self.root)
        
        # Create a disconnected graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        test_graph.add_edge(3, 4)
        
        # Set the graph
        viewer.set_graph(test_graph)
        
        # Check connectivity
        self.assertEqual(viewer.components_label.cget('text'), 'Components: 2')
        self.assertEqual(viewer.is_connected_label.cget('text'), 'Connected: No')
        
        # Check node analysis
        self.assertEqual(viewer.end_nodes_label.cget('text'), 'End Nodes: 4')
        self.assertEqual(viewer.branch_nodes_label.cget('text'), 'Branch Nodes: 0')
        self.assertEqual(viewer.junction_nodes_label.cget('text'), 'Junction Nodes: 0')
        
    def test_display_update_with_complex_graph(self):
        """Test display update with more complex graph."""
        viewer = GraphViewer(self.root)
        
        # Create a star graph
        test_graph = nx.star_graph(5)
        
        # Set the graph
        viewer.set_graph(test_graph)
        
        # Check basic metrics
        self.assertEqual(viewer.nodes_label.cget('text'), 'Nodes: 6')
        self.assertEqual(viewer.edges_label.cget('text'), 'Edges: 5')
        
        # Check node analysis
        self.assertEqual(viewer.end_nodes_label.cget('text'), 'End Nodes: 5')
        self.assertEqual(viewer.branch_nodes_label.cget('text'), 'Branch Nodes: 1')
        self.assertEqual(viewer.junction_nodes_label.cget('text'), 'Junction Nodes: 0')
        
    def test_clear_display(self):
        """Test clearing the display."""
        viewer = GraphViewer(self.root)
        
        # Set a graph first
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        viewer.set_graph(test_graph)
        
        # Clear display
        viewer._clear_display()
        
        # Check that all labels are reset
        self.assertEqual(viewer.nodes_label.cget('text'), 'Nodes: 0')
        self.assertEqual(viewer.edges_label.cget('text'), 'Edges: 0')
        self.assertEqual(viewer.density_label.cget('text'), 'Density: 0.0')
        
        # Buttons should be disabled
        self.assertEqual(viewer.export_button.cget('state'), 'disabled')
        self.assertEqual(viewer.analyze_button.cget('state'), 'disabled')
        
    # @unittest.skip("Skipping export graph test - can segfault")
    # def test_export_graph(self):
    #     """Test graph export functionality."""
    #     viewer = GraphViewer(self.root)
        
    #     # Create a test graph
    #     test_graph = nx.Graph()
    #     test_graph.add_edge(1, 2)
    #     viewer.set_graph(test_graph)
        
    #     # Mock the export functionality
    #     with patch('builtins.print') as mock_print:
    #         viewer._export_graph()
    #         mock_print.assert_called_with("Graph exported: 2 nodes, 1 edges")
            
    def test_analyze_graph(self):
        """Test graph analysis functionality."""
        viewer = GraphViewer(self.root)
        
        # Create a test graph
        test_graph = nx.Graph()
        test_graph.add_edge(1, 2)
        test_graph.add_edge(2, 3)
        viewer.set_graph(test_graph)
        
        # Mock the analysis functionality
        with patch.object(viewer, '_perform_detailed_analysis') as mock_analysis:
            mock_analysis.return_value = {'test': 'result'}
            with patch.object(viewer, '_show_analysis_results') as mock_show:
                viewer._analyze_graph()
                mock_analysis.assert_called_once()
                mock_show.assert_called_once_with({'test': 'result'})


if __name__ == '__main__':
    unittest.main()




