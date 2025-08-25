"""
Parameter panel for controlling skeletonization and graph extraction parameters.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


class ParameterPanel(ttk.Frame):
    """Panel for controlling skeletonization and graph extraction parameters."""
    
    def __init__(self, parent, callback: Callable[[], None] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.callback = callback
        self.parameters = {}
        
        self._setup_ui()
        self._setup_defaults()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Skeletonization parameters
        self._create_skeletonization_section()
        
        # Graph extraction parameters
        self._create_graph_extraction_section()
        
        # Live preview checkbox
        live_preview_frame = ttk.Frame(self)
        live_preview_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.live_preview_var = tk.BooleanVar(value=False)
        live_preview_check = ttk.Checkbutton(
            live_preview_frame,
            text="Live Preview (Auto-apply changes)",
            variable=self.live_preview_var,
            command=self._on_live_preview_change
        )
        live_preview_check.pack(side=tk.LEFT)
        
        # Apply button
        apply_frame = ttk.Frame(self)
        apply_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.apply_button = ttk.Button(
            apply_frame, 
            text="Apply Parameters", 
            command=self._apply_parameters
        )
        self.apply_button.pack(fill=tk.X)
        
        # Initially disabled until image is loaded
        self.apply_button.config(state=tk.DISABLED)
        
    def _create_skeletonization_section(self):
        """Create the skeletonization parameters section."""
        # Skeletonization frame
        skeleton_frame = ttk.LabelFrame(self, text="Skeletonization", padding=10)
        skeleton_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Method selection
        method_frame = ttk.Frame(skeleton_frame)
        method_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="medial_axis")
        method_combo = ttk.Combobox(
            method_frame, 
            textvariable=self.method_var,
            values=["medial_axis", "skeletonize", "thin"],
            state="readonly",
            width=15
        )
        method_combo.pack(side=tk.RIGHT)
        method_combo.bind('<<ComboboxSelected>>', lambda e: self._on_parameter_change())
        
        # Thinning parameters (for 'thin' method)
        thin_params_frame = ttk.LabelFrame(skeleton_frame, text="Thinning Parameters", padding=5)
        thin_params_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Max iterations for thinning
        max_iter_frame = ttk.Frame(thin_params_frame)
        max_iter_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(max_iter_frame, text="Max Iterations:").pack(side=tk.LEFT)
        self.max_iter_var = tk.IntVar(value=100)
        max_iter_scale = ttk.Scale(
            max_iter_frame, 
            from_=1, 
            to=500, 
            orient=tk.HORIZONTAL,
            variable=self.max_iter_var,
            command=lambda v: self._on_parameter_change()
        )
        max_iter_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Preprocessing parameters
        preprocess_frame = ttk.LabelFrame(skeleton_frame, text="Preprocessing", padding=5)
        preprocess_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Gaussian blur
        blur_frame = ttk.Frame(preprocess_frame)
        blur_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(blur_frame, text="Gaussian Blur:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=1)
        blur_scale = ttk.Scale(
            blur_frame, 
            from_=1, 
            to=21, 
            orient=tk.HORIZONTAL,
            variable=self.blur_var,
            command=lambda v: self._on_parameter_change()
        )
        blur_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Median filter
        median_frame = ttk.Frame(preprocess_frame)
        median_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(median_frame, text="Median Filter:").pack(side=tk.LEFT)
        self.median_var = tk.IntVar(value=1)
        median_scale = ttk.Scale(
            median_frame, 
            from_=1, 
            to=21, 
            orient=tk.HORIZONTAL,
            variable=self.median_var,
            command=lambda v: self._on_parameter_change()
        )
        median_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Threshold parameters
        threshold_frame = ttk.LabelFrame(skeleton_frame, text="Thresholding", padding=5)
        threshold_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Threshold method
        thresh_method_frame = ttk.Frame(threshold_frame)
        thresh_method_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(thresh_method_frame, text="Method:").pack(side=tk.LEFT)
        self.thresh_method_var = tk.StringVar(value="otsu")
        thresh_method_combo = ttk.Combobox(
            thresh_method_frame, 
            textvariable=self.thresh_method_var,
            values=["otsu", "triangle", "yen", "manual"],
            state="readonly",
            width=15
        )
        thresh_method_combo.pack(side=tk.RIGHT)
        thresh_method_combo.bind('<<ComboboxSelected>>', lambda e: self._on_parameter_change())
        
        # Manual threshold value
        manual_thresh_frame = ttk.Frame(threshold_frame)
        manual_thresh_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(manual_thresh_frame, text="Manual Threshold:").pack(side=tk.LEFT)
        self.manual_thresh_var = tk.IntVar(value=128)
        manual_thresh_scale = ttk.Scale(
            manual_thresh_frame, 
            from_=0, 
            to=255, 
            orient=tk.HORIZONTAL,
            variable=self.manual_thresh_var,
            command=lambda v: self._on_parameter_change()
        )
        manual_thresh_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Threshold preview value
        thresh_preview_frame = ttk.Frame(threshold_frame)
        thresh_preview_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(thresh_preview_frame, text="Preview Value:").pack(side=tk.LEFT)
        self.thresh_preview_label = ttk.Label(thresh_preview_frame, text="128")
        self.thresh_preview_label.pack(side=tk.RIGHT)
        
        # Update preview when threshold changes
        manual_thresh_scale.configure(command=self._update_threshold_preview)
        
    def _create_graph_extraction_section(self):
        """Create the graph extraction parameters section."""
        # Graph extraction frame
        graph_frame = ttk.LabelFrame(self, text="Graph Extraction", padding=10)
        graph_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Branch threshold
        branch_frame = ttk.Frame(graph_frame)
        branch_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(branch_frame, text="Branch Threshold:").pack(side=tk.LEFT)
        self.branch_thresh_var = tk.IntVar(value=1)
        branch_scale = ttk.Scale(
            branch_frame, 
            from_=1, 
            to=50, 
            orient=tk.HORIZONTAL,
            variable=self.branch_thresh_var,
            command=lambda v: self._on_parameter_change()
        )
        branch_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Min path length
        min_path_frame = ttk.Frame(graph_frame)
        min_path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(min_path_frame, text="Min Path Length:").pack(side=tk.LEFT)
        self.min_path_var = tk.IntVar(value=1)
        min_path_scale = ttk.Scale(
            min_path_frame, 
            from_=1, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.min_path_var,
            command=lambda v: self._on_parameter_change()
        )
        min_path_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Node filtering
        node_filter_frame = ttk.Frame(graph_frame)
        node_filter_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(node_filter_frame, text="Min Node Size:").pack(side=tk.LEFT)
        self.min_node_var = tk.IntVar(value=1)
        min_node_scale = ttk.Scale(
            node_filter_frame, 
            from_=1, 
            to=20, 
            orient=tk.HORIZONTAL,
            variable=self.min_node_var,
            command=lambda v: self._on_parameter_change()
        )
        min_node_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
    def _setup_defaults(self):
        """Setup default parameter values."""
        self.parameters = {
            'skeleton_method': 'medial_axis',
            'gaussian_blur': 1,  # No blur by default
            'median_filter': 1,   # No median filter by default
            'threshold_method': 'otsu',
            'manual_threshold': 128,
            'max_iterations': 100,
            'branch_threshold': 1,  # More permissive branch detection
            'min_path_length': 1,   # Keep all paths
            'min_node_size': 1      # Keep all nodes
        }
        
    def _on_parameter_change(self):
        """Handle parameter changes."""
        # Update parameters dictionary
        self.parameters.update({
            'skeleton_method': self.method_var.get(),
            'gaussian_blur': self.blur_var.get(),
            'median_filter': self.median_var.get(),
            'threshold_method': self.thresh_method_var.get(),
            'manual_threshold': self.manual_thresh_var.get(),
            'max_iterations': self.max_iter_var.get(),
            'branch_threshold': self.branch_thresh_var.get(),
            'min_path_length': self.min_path_var.get(),
            'min_node_size': self.min_node_var.get()
        })
        
        # Enable apply button or trigger live preview
        if self.live_preview_var.get():
            # Live preview enabled - apply immediately
            if self.callback:
                print(f"Live preview triggered with parameters: {self.parameters}")  # Debug
                self.callback()
        else:
            # Live preview disabled - enable manual apply
            self.apply_button.config(state=tk.NORMAL)
        
    def _apply_parameters(self):
        """Apply the current parameters."""
        if self.callback:
            self.callback()
            
        # Disable apply button after applying
        self.apply_button.config(state=tk.DISABLED)
        
    def _on_live_preview_change(self):
        """Handle live preview checkbox change."""
        if self.live_preview_var.get():
            # Enable live preview - apply parameters immediately
            self.apply_button.config(state=tk.DISABLED)
            self.apply_button.config(text="Live Preview Active")
        else:
            # Disable live preview - manual apply required
            self.apply_button.config(state=tk.NORMAL)
            self.apply_button.config(text="Apply Parameters")
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return self.parameters.copy()
        
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameter values from dictionary."""
        for key, value in params.items():
            if hasattr(self, f"{key}_var"):
                getattr(self, f"{key}_var").set(value)
                
        # Update parameters dictionary
        self.parameters.update(params)
        
    def set_enabled(self, enabled: bool):
        """Enable or disable all parameter controls."""
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Disable all child widgets
        for child in self.winfo_children():
            if hasattr(child, 'winfo_children'):
                for grandchild in child.winfo_children():
                    if hasattr(grandchild, 'config'):
                        try:
                            grandchild.config(state=state)
                        except tk.TclError:
                            pass  # Some widgets don't support state
                            
        # Apply button state
        if enabled:
            self.apply_button.config(state=tk.NORMAL)
        else:
            self.apply_button.config(state=tk.DISABLED)
            
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self._setup_defaults()
        
        # Update UI variables
        self.method_var.set(self.parameters['skeleton_method'])
        self.blur_var.set(self.parameters['gaussian_blur'])
        self.median_var.set(self.parameters['median_filter'])
        self.thresh_method_var.set(self.parameters['threshold_method'])
        self.manual_thresh_var.set(self.parameters['manual_threshold'])
        self.max_iter_var.set(self.parameters['max_iterations'])
        self.branch_thresh_var.set(self.parameters['branch_threshold'])
        self.min_path_var.set(self.parameters['min_path_length'])
        self.min_node_var.set(self.parameters['min_node_size'])
        
        # Trigger parameter change
        self._on_parameter_change()
        
    def _update_threshold_preview(self, value):
        """Update the threshold preview label."""
        self.thresh_preview_label.config(text=str(int(float(value))))
        self._on_parameter_change()




