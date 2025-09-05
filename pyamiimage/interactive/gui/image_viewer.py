"""
Image viewer component with zoom, pan, and viewport overlay capabilities.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Tuple, Dict, Any


class ImageViewer(ttk.Frame):
    """Interactive image viewer with zoom, pan, and viewport overlay."""
    
    def __init__(self, parent, title: str = "Image Viewer", **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.image = None
        self.photo = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.viewport_overlay = None
        
        # Performance optimization: debounce display updates
        self._update_pending = False
        self._last_update_time = 0
        
        self._setup_ui()
        self._bind_events()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Title and controls
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(title_frame, text=self.title).pack(side=tk.LEFT)
        
        # Zoom controls
        zoom_frame = ttk.Frame(title_frame)
        zoom_frame.pack(side=tk.RIGHT)
        
        ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=self._zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self._reset_view).pack(side=tk.LEFT, padx=2)
        
        # Canvas for image display with scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.canvas = tk.Canvas(canvas_frame, bg='white', relief=tk.SUNKEN, bd=1)
        
        # Create scrollbars
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        # Configure canvas scrolling
        self.canvas.configure(
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        
        # Grid layout for canvas and scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Set minimum size to ensure scrollbars are visible
        canvas_frame.configure(width=400, height=300)
        
        # Configure the frame to expand properly
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_label = ttk.Label(self, text="No image loaded", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
    def _bind_events(self):
        """Bind mouse and keyboard events."""
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)  # Windows/Linux
        self.canvas.bind('<Button-4>', lambda e: self._zoom_in())  # macOS scroll up
        self.canvas.bind('<Button-5>', lambda e: self._zoom_out())  # macOS scroll down
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        
        # Keyboard shortcuts
        self.canvas.bind('<Control-plus>', lambda e: self._zoom_in())
        self.canvas.bind('<Control-minus>', lambda e: self._zoom_out())
        self.canvas.bind('<Control-0>', lambda e: self._reset_view())
        
        # Bind resize event to update scroll region
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        
    def set_image(self, image: np.ndarray):
        """
        Set the image to display.
        
        Args:
            image: NumPy array representing the image
        """
        self.image = image
        self._update_display()
        
    def get_image(self) -> Optional[np.ndarray]:
        """Get the current image."""
        return self.image
        
    def _update_display(self, force=False):
        """Update the image display with debouncing for performance."""
        if self.image is None:
            self.canvas.delete("all")
            self.status_label.config(text="No image loaded")
            return
            
        # Performance optimization: debounce rapid updates
        if not force:
            import time
            current_time = time.time()
            if current_time - self._last_update_time < 0.1:  # 100ms debounce
                if not self._update_pending:
                    self._update_pending = True
                    self.after(100, self._debounced_update)
                return
            self._last_update_time = current_time
            self._update_pending = False
            
        try:
            # Convert numpy array to PIL Image
            if len(self.image.shape) == 3:
                # Color image
                pil_image = Image.fromarray(self.image)
            else:
                # Grayscale image
                pil_image = Image.fromarray(self.image, mode='L')
                
            # Apply zoom and pan
            width = int(pil_image.width * self.zoom_factor)
            height = int(pil_image.height * self.zoom_factor)
            
            # Resize image
            resized_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(resized_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas.create_image(
                self.pan_x, self.pan_y,
                anchor=tk.NW,
                image=self.photo
            )
            
            # Keep reference to prevent garbage collection issues
            self._photo_ref = self.photo
            
            # Update scroll region to include the entire image area
            # Calculate the total scrollable area
            total_width = max(width, self.canvas.winfo_width())
            total_height = max(height, self.canvas.winfo_height())
            
            # Set scroll region to cover the entire image area plus pan offset
            scroll_left = min(0, self.pan_x)
            scroll_top = min(0, self.pan_y)
            scroll_right = max(total_width, self.pan_x + width)
            scroll_bottom = max(total_height, self.pan_y + height)
            
            self.canvas.configure(scrollregion=(scroll_left, scroll_top, scroll_right, scroll_bottom))
            
            # Update status
            self.status_label.config(
                text=f"Image: {self.image.shape[1]}x{self.image.shape[0]} "
                     f"Zoom: {self.zoom_factor:.2f}x"
            )
            
            # Redraw viewport overlay if exists
            if self.viewport_overlay:
                self._draw_viewport_overlay()
                
        except Exception as e:
            self.status_label.config(text=f"Error displaying image: {str(e)}")
            
    def _debounced_update(self):
        """Debounced update method for performance optimization."""
        self._update_pending = False
        self._update_display(force=True)
            
    def _zoom_in(self):
        """Zoom in by 20%."""
        self.zoom_factor *= 1.2
        self._update_display()  # Uses debounced update
        
    def _zoom_out(self):
        """Zoom out by 20%."""
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
        self._update_display()  # Uses debounced update
        
    def _reset_view(self):
        """Reset zoom and pan to default."""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._update_display(force=True)  # Force immediate update for reset
        
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        self.canvas.scan_mark(event.x, event.y)
        
    def _on_mouse_drag(self, event):
        """Handle mouse drag for panning."""
        # Use canvas scan_dragto for smooth panning with scrollbars
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        # Update pan coordinates for viewport calculations
        self.pan_x = self.canvas.canvasx(0)
        self.pan_y = self.canvas.canvasy(0)
        
        # Update pan coordinates for viewport overlay
        self.pan_x = self.canvas.canvasx(0)
        self.pan_y = self.canvas.canvasy(0)
        
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        pass
        
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        # Handle different mouse wheel event formats
        if hasattr(event, 'delta'):
            # Windows/Linux
            if event.delta > 0:
                self._zoom_in()
            else:
                self._zoom_out()
        elif hasattr(event, 'num'):
            # macOS
            if event.num == 4:  # Scroll up
                self._zoom_in()
            elif event.num == 5:  # Scroll down
                self._zoom_out()
            
    def _on_mouse_move(self, event):
        """Handle mouse movement for coordinate display."""
        if self.image is not None:
            # Convert canvas coordinates to image coordinates
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Convert to image coordinates
            img_x = int((canvas_x - self.pan_x) / self.zoom_factor)
            img_y = int((canvas_y - self.pan_y) / self.zoom_factor)
            
            # Check bounds
            if 0 <= img_x < self.image.shape[1] and 0 <= img_y < self.image.shape[0]:
                # Get pixel value
                if len(self.image.shape) == 3:
                    pixel_value = self.image[img_y, img_x]
                    pixel_str = f"RGB({pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]})"
                else:
                    pixel_value = self.image[img_y, img_x]
                    pixel_str = f"Gray({pixel_value})"
                    
                self.status_label.config(
                    text=f"Position: ({img_x}, {img_y}) {pixel_str} "
                         f"Zoom: {self.zoom_factor:.2f}x"
                )
                
    def set_viewport_overlay(self, viewport: Dict[str, Any]):
        """
        Set viewport overlay to show current view area.
        
        Args:
            viewport: Dictionary with 'x', 'y', 'width', 'height' keys
        """
        self.viewport_overlay = viewport
        if self.image is not None:
            self._draw_viewport_overlay()
            
    def _draw_viewport_overlay(self):
        """Draw the viewport overlay on the image."""
        if self.viewport_overlay is None or self.image is None:
            return
            
        # Clear previous overlay
        self.canvas.delete("viewport_overlay")
        
        # Convert viewport coordinates to canvas coordinates
        x = self.viewport_overlay['x'] * self.zoom_factor + self.pan_x
        y = self.viewport_overlay['y'] * self.zoom_factor + self.pan_y
        width = self.viewport_overlay['width'] * self.zoom_factor
        height = self.viewport_overlay['height'] * self.zoom_factor
        
        # Draw rectangle overlay
        self.canvas.create_rectangle(
            x, y, x + width, y + height,
            outline='red',
            width=2,
            tags="viewport_overlay"
        )
        
    def get_viewport(self) -> Optional[Dict[str, Any]]:
        """Get current viewport information."""
        if self.image is None:
            return None
            
        # Get visible area in canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Convert to image coordinates
        img_width = canvas_width / self.zoom_factor
        img_height = canvas_height / self.zoom_factor
        
        return {
            'x': -self.pan_x / self.zoom_factor,
            'y': -self.pan_y / self.zoom_factor,
            'width': img_width,
            'height': img_height
        }
        
    def fit_to_window(self):
        """Fit image to window size."""
        if self.image is None:
            return
            
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet sized, schedule for later
            self.after(100, self.fit_to_window)
            return
            
        # Calculate zoom factor to fit image
        img_width = self.image.shape[1]
        img_height = self.image.shape[0]
        
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        
        # Use smaller zoom factor to fit completely
        self.zoom_factor = min(zoom_x, zoom_y) * 0.9  # 90% to leave some margin
        
        # Center the image
        self.pan_x = (canvas_width - img_width * self.zoom_factor) / 2
        self.pan_y = (canvas_height - img_height * self.zoom_factor) / 2
        
        self._update_display()
        
    def _on_canvas_resize(self, event):
        """Handle canvas resize events to update scroll region."""
        if self.image is not None:
            # Update scroll region when canvas is resized
            self._update_display()



