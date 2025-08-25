"""
Main window for the Interactive skeletonization dashboard.

Reuses existing pyamiimage skeletonization code and provides an interactive interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from pathlib import Path
from typing import Optional

# Reuse existing pyamiimage modules
from pyamiimage.ami_image import AmiImage
from pyamiimage.ami_skeleton import AmiSkeleton
from pyamiimage.ami_graph_all import AmiGraph

# Import networkx for graph analysis
import networkx as nx

from pyamiimage.interactive.gui.image_viewer import ImageViewer
from pyamiimage.interactive.gui.parameter_panel import ParameterPanel
from pyamiimage.interactive.gui.graph_viewer import GraphViewer


class SkeletonizationDashboard:
    """Main dashboard window for interactive skeletonization and graph analysis."""
    
    def __init__(self, root: Optional[tk.Tk] = None):
        """
        Initialize the dashboard.
        
        Args:
            root: Tkinter root window (creates new one if None)
        """
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
            
        self.root.title("Interactive - Skeletonization Dashboard")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.image_processor = None
        self.skeletonizer = None
        self.current_skeleton = None
        self.current_graph = None
        
        # Setup UI
        self._setup_menu()
        self._setup_main_layout()
        self._setup_status_bar()
        
        # Bind events
        self._bind_events()
        
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Skeleton...", command=self._save_skeleton)
        file_menu.add_command(label="Save Graph...", command=self._save_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self._reset_view)
        view_menu.add_command(label="Fit to Window", command=self._fit_to_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _setup_main_layout(self):
        """Setup the main layout with image viewers and parameter panel."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Original image viewer
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        original_label = ttk.Label(left_frame, text="Original Image")
        original_label.pack(pady=(0, 5))
        
        self.original_viewer = ImageViewer(left_frame, title="Original")
        self.original_viewer.pack(fill=tk.BOTH, expand=True)
        
        # Center panel - Skeleton image viewer
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        skeleton_label = ttk.Label(center_frame, text="Skeleton Image")
        skeleton_label.pack(pady=(0, 5))
        
        self.skeleton_viewer = ImageViewer(center_frame, title="Skeleton")
        self.skeleton_viewer.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Parameters and graph
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Parameter panel
        param_label = ttk.Label(right_frame, text="Parameters")
        param_label.pack(pady=(0, 5))
        
        self.parameter_panel = ParameterPanel(right_frame, callback=self._on_parameters_changed)
        self.parameter_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Graph viewer
        graph_label = ttk.Label(right_frame, text="Graph Analysis")
        graph_label.pack(pady=(0, 5))
        
        self.graph_viewer = GraphViewer(right_frame)
        self.graph_viewer.pack(fill=tk.BOTH, expand=True)
        
    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _bind_events(self):
        """Bind keyboard and mouse events."""
        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-s>', lambda e: self._save_skeleton())
        self.root.bind('<Control-g>', lambda e: self._save_graph())
        
    def _open_image(self):
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self._load_image(file_path)
            
    def _load_image(self, file_path: str):
        """Load and display an image."""
        try:
            self.status_bar.config(text=f"Loading image: {Path(file_path).name}")
            self.root.update()
            
            # Load image using existing AmiImage functionality and convert to grayscale
            image = AmiImage.create_grayscale_from_file(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
                
            # Display in original viewer
            self.original_viewer.set_image(image)
            
            # Store current image path
            self.current_image_path = file_path
            
            # Update status
            self.status_bar.config(text=f"Loaded: {Path(file_path).name}")
            
            # Enable parameter controls
            self.parameter_panel.set_enabled(True)
            
            # Process with default parameters
            self._process_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_bar.config(text="Error loading image")
            
    def _process_image(self):
        """Process the current image with current parameters."""
        if not hasattr(self, 'current_image_path'):
            return
            
        try:
            self.status_bar.config(text="Processing image...")
            self.root.update()
            
            # Get current parameters
            params = self.parameter_panel.get_parameters()
            
            # Use existing AmiImage skeletonization with enhanced parameters
            image = self.original_viewer.get_image()
            
            # Apply preprocessing if specified
            if params['gaussian_blur'] > 1 or params['median_filter'] > 1:
                # Apply preprocessing filters
                if params['gaussian_blur'] > 1:
                    import cv2
                    image = cv2.GaussianBlur(image, (params['gaussian_blur'], params['gaussian_blur']), 0)
                if params['median_filter'] > 1:
                    import cv2
                    image = cv2.medianBlur(image, params['median_filter'])
            
            # Apply thresholding if manual method is selected
            if params['threshold_method'] == 'manual':
                import cv2
                _, image = cv2.threshold(image, params['manual_threshold'], 255, cv2.THRESH_BINARY)
            
            # Create skeleton
            skeleton = AmiImage.create_white_skeleton_from_image(
                image,
                method=params['skeleton_method']
            )
            
            if skeleton is not None:
                self.current_skeleton = skeleton
                self.skeleton_viewer.set_image(skeleton)
                
                # Extract graph
                self._extract_graph(skeleton)
                
                # Update viewport linking
                self._update_viewport_linking()
                
                self.status_bar.config(text="Processing complete")
            else:
                messagebox.showerror("Error", "Failed to create skeleton")
                self.status_bar.config(text="Skeletonization failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_bar.config(text="Processing failed")
            
    def _extract_graph(self, skeleton: np.ndarray):
        """Extract graph from skeleton using existing AmiGraph functionality."""
        try:
            # Use existing AmiGraph functionality
            self.current_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton)
            
            if self.current_graph is not None:
                # Color the graph components
                colored_graph = self._color_graph_components(self.current_graph)
                
                # Update graph viewer with colored graph
                self.graph_viewer.set_graph(colored_graph)
                
                # Update status with graph info
                num_nodes = len(self.current_graph.nodes())
                num_edges = len(self.current_graph.edges())
                num_components = len(list(nx.connected_components(self.current_graph)))
                self.status_bar.config(text=f"Graph extracted: {num_nodes} nodes, {num_edges} edges, {num_components} components")
            else:
                self.status_bar.config(text="Graph extraction failed")
                
        except Exception as e:
            print(f"Graph extraction error: {e}")
            self.status_bar.config(text="Graph extraction failed")
            
    def _color_graph_components(self, graph):
        """Color each connected component of the graph with different colors."""
        import networkx as nx
        
        # Create a copy of the graph to avoid modifying the original
        colored_graph = graph.copy()
        
        # Find all connected components
        components = list(nx.connected_components(colored_graph))
        
        # Create a color palette
        color_palette = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Mint
            '#F7DC6F',  # Gold
            '#BB8FCE',  # Purple
            '#85C1E9'   # Light Blue
        ]
        
        # Color each component
        for i, component in enumerate(components):
            color = color_palette[i % len(color_palette)]
            
            # Set node attributes for coloring
            for node in component:
                colored_graph.nodes[node]['color'] = color
                colored_graph.nodes[node]['component'] = i
                colored_graph.nodes[node]['component_size'] = len(component)
            
            # Set edge attributes for coloring
            for node in component:
                for neighbor in colored_graph.neighbors(node):
                    if neighbor in component:  # Only color edges within the same component
                        colored_graph.edges[node, neighbor]['color'] = color
                        colored_graph.edges[node, neighbor]['component'] = i
        
        # Find the largest component
        largest_component = max(components, key=len)
        largest_component_id = components.index(largest_component)
        
        print(f"Graph colored: {len(components)} components found")
        print(f"Largest component: {len(largest_component)} nodes (component {largest_component_id})")
        print(f"Component sizes: {[len(c) for c in components]}")
        
        return colored_graph
            
    def _update_viewport_linking(self):
        """Update viewport linking between original and skeleton views."""
        # Get current skeleton viewport
        skeleton_viewport = self.skeleton_viewer.get_viewport()
        if skeleton_viewport:
            # Update original viewer with viewport overlay
            self.original_viewer.set_viewport_overlay(skeleton_viewport)
            
    def _on_parameters_changed(self):
        """Callback when parameters change."""
        self._process_image()
        
    def _save_skeleton(self):
        """Save the current skeleton image."""
        if self.current_skeleton is None:
            messagebox.showwarning("Warning", "No skeleton to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Skeleton",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Save using PIL
                from PIL import Image
                Image.fromarray(self.current_skeleton).save(file_path)
                self.status_bar.config(text=f"Skeleton saved: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save skeleton: {str(e)}")
                
    def _save_graph(self):
        """Save the current graph."""
        if self.current_graph is None:
            messagebox.showwarning("Warning", "No graph to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Graph",
            defaultextension=".gml",
            filetypes=[
                ("GML files", "*.gml"),
                ("GraphML files", "*.xml"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Save using NetworkX
                if file_path.endswith('.gml'):
                    import networkx as nx
                    nx.write_gml(self.current_graph, file_path)
                elif file_path.endswith('.xml'):
                    import networkx as nx
                    nx.write_graphml(self.current_graph, file_path)
                    
                self.status_bar.config(text=f"Graph saved: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save graph: {str(e)}")
                
    def _reset_view(self):
        """Reset all views to default."""
        self.original_viewer.reset_view()
        self.skeleton_viewer.reset_view()
        
    def _fit_to_window(self):
        """Fit images to window."""
        self.original_viewer.fit_to_window()
        self.skeleton_viewer.fit_to_window()
        
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Interactive",
            "Interactive - Interactive Skeletonization Dashboard\n\n"
            "Interactive skeletonization and graph analysis dashboard\n"
            "Built on pyamiimage framework\n\n"
            "Version: 0.1.0"
        )
        
    def run(self):
        """Run the dashboard."""
        self.root.mainloop()
        
    def destroy(self):
        """Clean up resources."""
        if hasattr(self, 'root'):
            self.root.destroy()
