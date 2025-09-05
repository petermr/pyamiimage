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
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add navigation toolbar for zoom/pan
            self.toolbar = NavigationToolbar2Tk(self.canvas, viz_frame)
            self.toolbar.update()

            # Bind click events for node inspection
            self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

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
            self.toolbar = None
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

        self.refresh_button = ttk.Button(
            actions_frame,
            text="Refresh View",
            command=self._refresh_visualization
        )
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        self.analyze_button = ttk.Button(
            actions_frame,
            text="Analyze",
            command=self._analyze_graph
        )
        self.analyze_button.pack(side=tk.LEFT)

        # Initially disabled
        self.export_button.config(state=tk.DISABLED)
        self.refresh_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)

    def set_graph(self, graph: nx.Graph):
        """
        Set the graph to display.
        
        Args:
            graph: NetworkX graph object
        """
        self.current_graph = graph
        self._update_display()
        self._update_graph_visualization()

        # Enable buttons
        self.export_button.config(state=tk.NORMAL)
        self.refresh_button.config(state=tk.NORMAL)
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
        self.refresh_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)

        # Clear matplotlib plot if available
        if hasattr(self, 'ax') and self.ax is not None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No graph loaded',
                         ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.draw()

    def _export_graph(self):
        """Export the current graph."""
        if self.current_graph is None:
            return

        try:
            from tkinter import filedialog
            import os

            # Ask for export format and location
            file_path = filedialog.asksaveasfilename(
                title="Export Graph",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )

            if file_path:
                # Export graph data
                if file_path.endswith('.gml'):
                    nx.write_gml(self.current_graph, file_path)
                elif file_path.endswith('.xml'):
                    nx.write_graphml(self.current_graph, file_path)
                elif file_path.endswith('.pkl'):
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(self.current_graph, f)
                else:
                    # Export visualization as image
                    self._export_visualization(file_path)

                print(
                    f"Graph exported: {len(self.current_graph.nodes())} nodes, {len(self.current_graph.edges())} edges")
                print(f"Saved to: {file_path}")

        except Exception as e:
            print(f"Error exporting graph: {e}")

    def _refresh_visualization(self):
        """Refresh the graph visualization with a new layout."""
        if self.current_graph is None:
            return

        try:
            # Clear stored positions to force new layout
            if hasattr(self.current_graph, 'nodes'):
                for node in self.current_graph.nodes():
                    if 'pos' in self.current_graph.nodes[node]:
                        del self.current_graph.nodes[node]['pos']

            # Update visualization with new layout
            self._update_graph_visualization()

        except Exception as e:
            print(f"Error refreshing visualization: {e}")

    def _export_visualization(self, file_path: str):
        """Export the current graph visualization as an image."""
        try:
            if hasattr(self, 'figure') and self.figure is not None:
                # Save the current figure
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight',
                                    facecolor='white', edgecolor='none')
                print(f"Visualization exported to: {file_path}")
            else:
                print("No visualization available to export")

        except Exception as e:
            print(f"Error exporting visualization: {e}")

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
        """Show analysis results inline instead of in a popup window."""
        try:
            # Create a simple text display in the main window
            # This avoids creating additional windows
            result_text = "Analysis Results:\n"
            result_text += "=" * 20 + "\n"

            for key, value in analysis.items():
                if isinstance(value, dict):
                    result_text += f"{key}: {len(value)} items\n"
                else:
                    result_text += f"{key}: {value}\n"

            # Display in status or print to console
            if hasattr(self, 'status_label'):
                self.status_label.config(text=result_text)
        except Exception as e:
            print(f"Error showing analysis results: {e}")

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
                print(result_text)
                text_widget.insert(tk.END, f"{key}: {value}\n\n")

        # except Exception as e:
        #     print(f"Error showing analysis results: {e}")

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

    def clear_graph(self):
        """Clear the current graph and reset all displays."""
        self.current_graph = None

        # Reset all labels
        self.nodes_label.config(text="Nodes: 0")
        self.edges_label.config(text="Edges: 0")
        self.density_label.config(text="Density: 0.0")
        self.components_label.config(text="Components: 0")
        self.largest_component_label.config(text="Largest Component: 0 nodes")
        self.is_connected_label.config(text="Connected: No")
        self.avg_degree_label.config(text="Avg Degree: 0.0")
        self.max_degree_label.config(text="Max Degree: 0")
        self.cycles_label.config(text="Cycles: Unknown")
        self.end_nodes_label.config(text="End Nodes: 0")
        self.branch_nodes_label.config(text="Branch Nodes: 0")
        self.junction_nodes_label.config(text="Junction Nodes: 0")

        # Show processing state in visualization
        self.show_processing("Processing...")

    def _update_graph_visualization(self):
        """Update the graph visualization with colored components."""
        if self.canvas is None or self.current_graph is None:
            return

        try:
            # Clear the previous plot
            self.ax.clear()

            # Choose layout algorithm based on graph size
            num_nodes = len(self.current_graph.nodes())
            if num_nodes < 50:
                # Small graphs: use spring layout for better visualization
                pos = nx.spring_layout(self.current_graph, k=2, iterations=100, seed=42)
            elif num_nodes < 200:
                # Medium graphs: use spring layout with more iterations
                pos = nx.spring_layout(self.current_graph, k=1.5, iterations=200, seed=42)
            else:
                # Large graphs: use kamada_kawai for better spacing
                try:
                    pos = nx.kamada_kawai_layout(self.current_graph)
                except:
                    pos = nx.spring_layout(self.current_graph, k=1, iterations=50, seed=42)

            # Store positions as node attributes for interactive features
            nx.set_node_attributes(self.current_graph, pos, 'pos')

            # Create color maps for nodes and edges
            node_colors = self._get_node_colors()
            edge_colors = self._get_edge_colors()

            # Draw edges
            nx.draw_networkx_edges(
                self.current_graph, pos,
                edge_color=edge_colors,
                width=1.5,
                alpha=0.6,
                arrowsize=10
            )

            # Draw nodes
            nx.draw_networkx_nodes(
                self.current_graph, pos,
                node_color=node_colors,
                node_size=200,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

            # Add node labels (only for smaller graphs)
            if num_nodes <= 100:
                nx.draw_networkx_labels(
                    self.current_graph, pos,
                    font_size=8,
                    font_weight='bold'
                )

            # Set title and remove axes
            title = f"Graph Visualization: {num_nodes} nodes, {len(self.current_graph.edges())} edges"
            if num_nodes > 100:
                title += " (node labels hidden for clarity)"
            self.ax.set_title(title, fontsize=10, fontweight='bold')
            self.ax.axis('off')

            # Add legend for node types
            self._add_node_legend()

            # Redraw the canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Error updating graph visualization: {e}")
            # Show error message on plot
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error visualizing graph: {str(e)}',
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='red')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.canvas.draw()

    def _get_node_colors(self):
        """Get colors for nodes based on their properties."""
        colors = []
        for node in self.current_graph.nodes():
            degree = self.current_graph.degree(node)
            if degree == 1:
                colors.append('#ff6b6b')  # Red for end nodes
            elif degree == 2:
                colors.append('#4ecdc4')  # Teal for junction nodes
            elif degree > 2:
                colors.append('#45b7d1')  # Blue for branch nodes
            else:
                colors.append('#96ceb4')  # Green for other nodes
        return colors

    def _get_edge_colors(self):
        """Get colors for edges based on their properties."""
        colors = []
        for edge in self.current_graph.edges():
            # Color edges based on whether they're part of cycles
            try:
                # Check if edge is part of a cycle
                temp_graph = self.current_graph.copy()
                temp_graph.remove_edge(*edge)
                if nx.has_path(temp_graph, edge[0], edge[1]):
                    colors.append('#ffa726')  # Orange for cycle edges
                else:
                    colors.append('#66bb6a')  # Green for tree edges
            except:
                colors.append('#888888')  # Gray for unknown
        return colors

    def _add_node_legend(self):
        """Add a legend showing node types and colors."""
        try:
            from matplotlib.patches import Patch

            # Create legend patches
            legend_elements = [
                Patch(facecolor='#ff6b6b', label='End Nodes (degree=1)'),
                Patch(facecolor='#4ecdc4', label='Junction Nodes (degree=2)'),
                Patch(facecolor='#45b7d1', label='Branch Nodes (degree>2)'),
                Patch(facecolor='#96ceb4', label='Other Nodes')
            ]

            # Add legend
            self.ax.legend(handles=legend_elements, loc='upper right',
                           fontsize=8, framealpha=0.8)
        except Exception as e:
            print(f"Error adding legend: {e}")

    def _on_canvas_click(self, event):
        """Handle clicks on the graph canvas for node inspection."""
        if self.current_graph is None or event.inaxes != self.ax:
            return

        try:
            # Find the closest node to the click
            click_pos = (event.xdata, event.ydata)
            if click_pos[0] is None or click_pos[1] is None:
                return

            # Get node positions
            pos = nx.get_node_attributes(self.current_graph, 'pos')
            if not pos:
                # If no stored positions, recalculate layout
                num_nodes = len(self.current_graph.nodes())
                if num_nodes < 50:
                    pos = nx.spring_layout(self.current_graph, k=2, iterations=100, seed=42)
                else:
                    pos = nx.spring_layout(self.current_graph, k=1, iterations=50, seed=42)

            # Find closest node
            min_dist = float('inf')
            closest_node = None

            for node, (x, y) in pos.items():
                dist = ((x - click_pos[0]) ** 2 + (y - click_pos[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node

            # If click is close enough to a node, show node info
            if closest_node and min_dist < 0.1:  # Threshold for node selection
                self._show_node_info(closest_node)

        except Exception as e:
            print(f"Error handling canvas click: {e}")

    def _show_node_info(self, node):
        """Show detailed information about a specific node."""
        try:
            # Instead of creating a popup window, update the status display
            # This keeps everything in one window
            info_text = f"Node {node}: Degree={self.current_graph.degree(node)}, "
            info_text += f"Neighbors={len(list(self.current_graph.neighbors(node)))}"

            # Update the main status or create a simple inline display
            if hasattr(self, 'status_label'):
                self.status_label.config(text=info_text)
            else:
                print(info_text)

        except Exception as e:
            print(f"Error showing node info: {e}")

    def show_processing(self, message: str = "Processing graph..."):
        """Show processing state in the graph visualization."""
        if hasattr(self, 'ax') and self.ax is not None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, message,
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=12, color='blue', fontweight='bold')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.draw()
