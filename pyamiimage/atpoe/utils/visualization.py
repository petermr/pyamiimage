"""
Visualization utilities for ATPOE skeletonization dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, Dict, Any


class VisualizationUtils:
    """Utility class for visualization operations."""
    
    @staticmethod
    def create_comparison_plot(original: np.ndarray, skeleton: np.ndarray, 
                              title: str = "Image Comparison") -> plt.Figure:
        """
        Create a side-by-side comparison plot of original and skeleton images.
        
        Args:
            original: Original image array
            skeleton: Skeleton image array
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        if len(original.shape) == 3:
            ax1.imshow(original)
        else:
            ax1.imshow(original, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Skeleton image
        ax2.imshow(skeleton, cmap='gray')
        ax2.set_title("Skeleton")
        ax2.axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
        
    @staticmethod
    def overlay_viewport_on_image(image: np.ndarray, viewport: Dict[str, Any], 
                                color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Overlay a viewport rectangle on an image.
        
        Args:
            image: Input image array
            viewport: Dictionary with 'x', 'y', 'width', 'height' keys
            color: RGB color tuple for the overlay
            
        Returns:
            Image with viewport overlay
        """
        if image is None or viewport is None:
            return image
            
        # Create a copy to avoid modifying the original
        result = image.copy()
        
        # Extract viewport coordinates
        x = int(viewport['x'])
        y = int(viewport['y'])
        width = int(viewport['width'])
        height = int(viewport['height'])
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        
        # Draw rectangle overlay
        if len(result.shape) == 3:
            # Color image
            result[y:y+height, x] = color  # Left edge
            result[y:y+height, x+width-1] = color  # Right edge
            result[y, x:x+width] = color  # Top edge
            result[y+height-1, x:x+width] = color  # Bottom edge
        else:
            # Grayscale image
            result[y:y+height, x] = 255  # Left edge
            result[y:y+height, x+width-1] = 255  # Right edge
            result[y, x:x+width] = 255  # Top edge
            result[y+height-1, x:x+width] = 255  # Bottom edge
            
        return result
        
    @staticmethod
    def create_graph_visualization(graph, layout_type: str = 'spring') -> plt.Figure:
        """
        Create a visualization of the NetworkX graph.
        
        Args:
            graph: NetworkX graph object
            layout_type: Layout algorithm ('spring', 'circular', 'random')
            
        Returns:
            Matplotlib figure object
        """
        if graph is None or len(graph.nodes()) == 0:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Choose layout
        if layout_type == 'spring':
            pos = nx.spring_layout(graph)
        elif layout_type == 'circular':
            pos = nx.circular_layout(graph)
        elif layout_type == 'random':
            pos = nx.random_layout(graph)
        else:
            pos = nx.spring_layout(graph)
            
        # Draw the graph
        nx.draw(graph, pos, ax=ax, with_labels=True, 
                node_color='lightblue', 
                node_size=500, 
                font_size=8,
                font_weight='bold')
                
        ax.set_title(f"Graph Visualization ({len(graph.nodes())} nodes, {len(graph.edges())} edges)")
        
        return fig
        
    @staticmethod
    def save_visualization(fig: plt.Figure, filepath: str, 
                          dpi: int = 300, format: str = 'png') -> bool:
        """
        Save a matplotlib figure to file.
        
        Args:
            fig: Matplotlib figure object
            filepath: Output file path
            dpi: Resolution in dots per inch
            format: Output format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            return True
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False
            
    @staticmethod
    def create_parameter_summary(params: Dict[str, Any]) -> str:
        """
        Create a text summary of parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Formatted parameter summary string
        """
        summary = "Parameter Summary:\n"
        summary += "=" * 20 + "\n"
        
        for key, value in params.items():
            summary += f"{key}: {value}\n"
            
        return summary
