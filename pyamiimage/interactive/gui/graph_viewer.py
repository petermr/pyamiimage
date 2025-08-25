"""
Graph viewer component for displaying NetworkX graph information and metrics.
"""

import tkinter as tk
from tkinter import ttk
import networkx as nx
from typing import Optional, Dict, Any


class GraphViewer(ttk.Frame):
    """Component for displaying graph information and metrics."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.current_graph = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Graph info frame
        info_frame = ttk.LabelFrame(self, text="Graph Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Basic metrics
        self.nodes_label = ttk.Label(info_frame, text="Nodes: 0")
        self.nodes_label.pack(anchor=tk.W, pady=2)
        
        self.edges_label = ttk.Label(info_frame, text="Edges: 0")
        self.edges_label.pack(anchor=tk.W, pady=2)
        
        self.density_label = ttk.Label(info_frame, text="Density: 0.0")
        self.density_label.pack(anchor=tk.W, pady=2)
        
        # Connectivity metrics
        connectivity_frame = ttk.LabelFrame(self, text="Connectivity", padding=10)
        connectivity_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.components_label = ttk.Label(connectivity_frame, text="Components: 0")
        self.components_label.pack(anchor=tk.W, pady=2)
        
        self.largest_component_label = ttk.Label(connectivity_frame, text="Largest Component: 0 nodes")
        self.largest_component_label.pack(anchor=tk.W, pady=2)
        
        self.is_connected_label = ttk.Label(connectivity_frame, text="Connected: No")
        self.is_connected_label.pack(anchor=tk.W, pady=2)
        
        # Topological metrics
        topology_frame = ttk.LabelFrame(self, text="Topology", padding=10)
        topology_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.avg_degree_label = ttk.Label(topology_frame, text="Avg Degree: 0.0")
        self.avg_degree_label.pack(anchor=tk.W, pady=2)
        
        self.max_degree_label = ttk.Label(topology_frame, text="Max Degree: 0")
        self.max_degree_label.pack(anchor=tk.W, pady=2)
        
        self.cycles_label = ttk.Label(topology_frame, text="Cycles: Unknown")
        self.cycles_label.pack(anchor=tk.W, pady=2)
        
        # Node type analysis
        node_analysis_frame = ttk.LabelFrame(self, text="Node Analysis", padding=10)
        node_analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.end_nodes_label = ttk.Label(node_analysis_frame, text="End Nodes: 0")
        self.end_nodes_label.pack(anchor=tk.W, pady=2)
        
        self.branch_nodes_label = ttk.Label(node_analysis_frame, text="Branch Nodes: 0")
        self.branch_nodes_label.pack(anchor=tk.W, pady=2)
        
        self.junction_nodes_label = ttk.Label(node_analysis_frame, text="Junction Nodes: 0")
        self.junction_nodes_label.pack(anchor=tk.W, pady=2)
        
        # Graph visualization frame
        viz_frame = ttk.LabelFrame(self, text="Graph Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add matplotlib canvas for graph display
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initial empty plot
            self.ax.text(0.5, 0.5, 'Load an image to see graph visualization', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.canvas.draw()
            
        except ImportError:
            # Fallback if matplotlib is not available
            self.canvas = None
            self.figure = None
            self.ax = None
            ttk.Label(viz_frame, text="Matplotlib not available for graph visualization").pack()
        
        # Actions frame
        actions_frame = ttk.Frame(self)
        actions_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.export_button = ttk.Button(
            actions_frame, 
            text="Export Graph", 
            command=self._export_graph
        )
        self.export_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.analyze_button = ttk.Button(
            actions_frame, 
            text="Analyze", 
            command=self._analyze_graph
        )
        self.analyze_button.pack(side=tk.LEFT)
        
        # Initially disabled
        self.export_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)
        
    def set_graph(self, graph: nx.Graph):
        """
        Set the graph to display.
        
        Args:
            graph: NetworkX graph object
        """
        self.current_graph = graph
        self._update_display()
        
        # Enable buttons
        self.export_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)
        
    def _update_display(self):
        """Update the graph information display."""
        if self.current_graph is None:
            self._clear_display()
            return
            
        try:
            # Basic metrics
            num_nodes = len(self.current_graph.nodes())
            num_edges = len(self.current_graph.edges())
            density = nx.density(self.current_graph)
            
            self.nodes_label.config(text=f"Nodes: {num_nodes}")
            self.edges_label.config(text=f"Edges: {num_edges}")
            self.density_label.config(text=f"Density: {density:.4f}")
            
            # Connectivity metrics
            components = list(nx.connected_components(self.current_graph))
            num_components = len(components)
            largest_component_size = max(len(comp) for comp in components) if components else 0
            is_connected = nx.is_connected(self.current_graph)
            
            self.components_label.config(text=f"Components: {num_components}")
            self.largest_component_label.config(text=f"Largest Component: {largest_component_size} nodes")
            self.is_connected_label.config(text=f"Connected: {'Yes' if is_connected else 'No'}")
            
            # Topological metrics
            degrees = [d for n, d in self.current_graph.degree()]
            avg_degree = sum(degrees) / len(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            
            self.avg_degree_label.config(text=f"Avg Degree: {avg_degree:.2f}")
            self.max_degree_label.config(text=f"Max Degree: {max_degree}")
            
            # Check for cycles (simplified)
            try:
                has_cycles = len(list(nx.simple_cycles(self.current_graph))) > 0
                self.cycles_label.config(text=f"Cycles: {'Yes' if has_cycles else 'No'}")
            except:
                self.cycles_label.config(text="Cycles: Unknown")
                
            # Node type analysis
            end_nodes = sum(1 for n, d in self.current_graph.degree() if d == 1)
            branch_nodes = sum(1 for n, d in self.current_graph.degree() if d > 2)
            junction_nodes = sum(1 for n, d in self.current_graph.degree() if d == 2)
            
            self.end_nodes_label.config(text=f"End Nodes: {end_nodes}")
            self.branch_nodes_label.config(text=f"Branch Nodes: {branch_nodes}")
            self.junction_nodes_label.config(text=f"Junction Nodes: {junction_nodes}")
            
        except Exception as e:
            print(f"Error updating graph display: {e}")
            self._clear_display()
            
    def _clear_display(self):
        """Clear all display labels."""
        labels = [
            self.nodes_label, self.edges_label, self.density_label,
            self.components_label, self.largest_component_label, self.is_connected_label,
            self.avg_degree_label, self.max_degree_label, self.cycles_label,
            self.end_nodes_label, self.branch_nodes_label, self.junction_nodes_label
        ]
        
        for label in labels:
            label.config(text=label.cget("text").split(":")[0] + ": 0")
            
        # Disable buttons
        self.export_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)
        
    def _export_graph(self):
        """Export the current graph."""
        if self.current_graph is None:
            return
            
        # This would typically open a file dialog
        # For now, just print graph info
        print(f"Graph exported: {len(self.current_graph.nodes())} nodes, {len(self.current_graph.edges())} edges")
        
    def _analyze_graph(self):
        """Perform detailed graph analysis."""
        if self.current_graph is None:
            return
            
        try:
            # Perform additional analysis
            analysis = self._perform_detailed_analysis()
            
            # Display results in a new window
            self._show_analysis_results(analysis)
            
        except Exception as e:
            print(f"Error analyzing graph: {e}")
            
    def _perform_detailed_analysis(self) -> Dict[str, Any]:
        """Perform detailed graph analysis."""
        analysis = {}
        
        if self.current_graph is None:
            return analysis
            
        try:
            # Centrality measures
            if len(self.current_graph.nodes()) > 1:
                analysis['betweenness'] = nx.betweenness_centrality(self.current_graph)
                analysis['closeness'] = nx.closeness_centrality(self.current_graph)
                analysis['eigenvector'] = nx.eigenvector_centrality_numpy(self.current_graph)
                
            # Path analysis
            if nx.is_connected(self.current_graph):
                analysis['diameter'] = nx.diameter(self.current_graph)
                analysis['radius'] = nx.radius(self.current_graph)
                analysis['average_shortest_path'] = nx.average_shortest_path_length(self.current_graph)
                
            # Clustering
            analysis['clustering'] = nx.average_clustering(self.current_graph)
            
            # Community detection (if networkx-community is available)
            try:
                import community
                partition = community.best_partition(self.current_graph)
                analysis['communities'] = len(set(partition.values()))
            except ImportError:
                analysis['communities'] = "Not available (install python-louvain)"
                
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            
        return analysis
        
    def _show_analysis_results(self, analysis: Dict[str, Any]):
        """Show analysis results in a new window."""
        # Create a simple text display
        result_window = tk.Toplevel(self)
        result_window.title("Graph Analysis Results")
        result_window.geometry("600x400")
        
        # Text widget for results
        text_widget = tk.Text(result_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Format and display results
        text_widget.insert(tk.END, "Detailed Graph Analysis Results\n")
        text_widget.insert(tk.END, "=" * 40 + "\n\n")
        
        for key, value in analysis.items():
            if isinstance(value, dict):
                text_widget.insert(tk.END, f"{key}:\n")
                for subkey, subvalue in list(value.items())[:10]:  # Limit to first 10
                    text_widget.insert(tk.END, f"  {subkey}: {subvalue:.4f}\n")
                if len(value) > 10:
                    text_widget.insert(tk.END, f"  ... and {len(value) - 10} more\n")
                text_widget.insert(tk.END, "\n")
            else:
                text_widget.insert(tk.END, f"{key}: {value}\n\n")
                
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_button = ttk.Button(
            result_window, 
            text="Close", 
            command=result_window.destroy
        )
        close_button.pack(pady=10)
        
    def get_graph(self) -> Optional[nx.Graph]:
        """Get the current graph."""
        return self.current_graph
        
    def clear(self):
        """Clear the current graph and display."""
        self.current_graph = None
        self._clear_display()
        
    def _update_graph_visualization(self):
        """Update the graph visualization with colored components."""
        if self.canvas is None or self.current_graph is None:
            return
            
        try:
            # Clear the previous plot
            self.ax.clear()
            
            # Get node positions using spring layout
            pos = nx.spring_layout(self.current_graph, k=1, iterations=50)
            
            # Draw edges with colors
            for edge in self.current_graph.edges():
                edge_color = self.current_graph.edges[edge].get('color', '#888888')
                nx.draw_networkx_edges(
                    self.current_graph, pos, 
                    edgelist=[edge], 
                    edge_color=edge_color,
                    width=2,
                    alpha=0.7
                )
            
            # Draw nodes with colors
            for node in self.current_graph.nodes():
                node_color = self.current_graph.nodes[node].get('color', '#888888')
                nx.draw_networkx_nodes(
                    self.current_graph, pos,
                    nodelist=[node],
                    node_color=node_color,
                    node_size=100,
                    alpha=0.8
                )
            
            # Add node labels
            nx.draw_networkx_labels(self.current_graph, pos, font_size=8)
            
            # Set title and remove axes
            self.ax.set_title(f"Graph: {len(self.current_graph.nodes())} nodes, {len(self.current_graph.edges())} edges")
            self.ax.axis('off')
            
            # Redraw the canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating graph visualization: {e}")
            # Show error message on plot
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error visualizing graph: {str(e)}', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.canvas.draw()




