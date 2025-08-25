# ATPOE Dashboard User Guide

## Overview

ATPOE (Advanced Topological Processing and Optimization Engine) is an interactive desktop application for skeletonization and graph analysis of images. It provides a user-friendly interface to explore different skeletonization parameters and analyze the resulting graph structures.

## Features

- **Interactive Image Processing**: Load and process images with real-time parameter adjustment
- **Multiple Skeletonization Methods**: Choose between medial_axis, skeletonize, and thin algorithms
- **Parameter Control**: Adjust preprocessing, thresholding, and graph extraction parameters
- **Linked Viewports**: See the relationship between original and skeleton images
- **Graph Analysis**: Extract and analyze NetworkX graphs with topological metrics
- **Export Capabilities**: Save skeletons, graphs, and analysis results

## Installation

### Prerequisites

- Python 3.8 or higher
- Tkinter (usually included with Python)
- Required Python packages (see requirements below)

### Dependencies

Install the required packages:

```bash
pip install opencv-python scikit-image skan networkx pillow numpy matplotlib
```

### Running the Application

```bash
# From the pyamiimage directory
python pyamiimage/atpoe.py

# Or as a module
python -m pyamiimage.atpoe
```

## User Interface

### Main Window Layout

The dashboard is divided into three main sections:

1. **Left Panel**: Image viewers for original and skeleton images
2. **Right Panel**: Parameter controls and graph analysis
3. **Menu Bar**: File operations, view controls, and help

### Menu Options

#### File Menu
- **Open Image**: Load an image file (PNG, JPEG, BMP, TIFF)
- **Save Skeleton**: Save the current skeleton image
- **Save Graph**: Export the current graph (GML, GraphML)
- **Exit**: Close the application

#### View Menu
- **Reset View**: Reset all image views to default
- **Fit to Window**: Automatically size images to fit the window

#### Help Menu
- **About**: Application information and version

### Keyboard Shortcuts

- `Ctrl+O`: Open image
- `Ctrl+S`: Save skeleton
- `Ctrl+G`: Save graph
- `Ctrl++`: Zoom in
- `Ctrl+-`: Zoom out
- `Ctrl+0`: Reset zoom

## Using the Dashboard

### Step 1: Load an Image

1. Click **File → Open Image** or use `Ctrl+O`
2. Select an image file (PNG, JPEG, BMP, or TIFF)
3. The image will appear in the "Original Image" viewer
4. Parameter controls will be automatically enabled

### Step 2: Adjust Parameters

The parameter panel contains several sections:

#### Skeletonization Parameters
- **Method**: Choose skeletonization algorithm
  - `medial_axis`: Most detailed, preserves topology
  - `skeletonize`: Standard skeletonization
  - `thin`: Thinning-based approach
- **Preprocessing**:
  - **Gaussian Blur**: Smoothing filter (1-21)
  - **Median Filter**: Noise reduction (1-21)

#### Thresholding Parameters
- **Method**: Threshold selection algorithm
  - `otsu`: Automatic thresholding
  - `triangle`: Triangle method
  - `yen`: Yen's method
  - `manual`: User-defined threshold
- **Manual Threshold**: Value from 0-255 (when using manual method)

#### Graph Extraction Parameters
- **Branch Threshold**: Minimum branch length (1-50)
- **Min Path Length**: Minimum path length (1-100)
- **Min Node Size**: Minimum node size (1-20)

### Step 3: Process the Image

1. Adjust parameters as needed
2. Click **Apply Parameters** to process the image
3. The skeleton will appear in the "Skeleton Image" viewer
4. Graph analysis will be updated automatically

### Step 4: Analyze Results

#### Image Comparison
- **Original Image**: Shows the input image with viewport overlay
- **Skeleton Image**: Shows the processed skeleton
- **Viewport Linking**: Red rectangle on original shows current skeleton view area
- **Zoom and Pan**: Use mouse wheel to zoom, drag to pan

#### Graph Analysis
The graph viewer displays:

**Basic Metrics**
- Number of nodes and edges
- Graph density

**Connectivity**
- Number of connected components
- Largest component size
- Whether the graph is connected

**Topology**
- Average and maximum node degrees
- Presence of cycles

**Node Analysis**
- End nodes (degree 1)
- Branch nodes (degree > 2)
- Junction nodes (degree 2)

### Step 5: Export Results

#### Save Skeleton
1. Click **File → Save Skeleton**
2. Choose format (PNG, JPEG)
3. Select save location

#### Save Graph
1. Click **File → Save Graph**
2. Choose format (GML, GraphML)
3. Select save location

#### Advanced Analysis
1. Click **Analyze** button in the Graph Analysis panel
2. View detailed metrics in a new window
3. Includes centrality measures, path analysis, and clustering

## Parameter Optimization

### Finding the Right Parameters

The optimal parameters depend on your image characteristics:

#### For Noisy Images
- Increase **Gaussian Blur** (7-11)
- Increase **Median Filter** (7-11)
- Use **Triangle** or **Yen** thresholding

#### For Low-Contrast Images
- Try **Manual Threshold** with lower values
- Increase **Gaussian Blur** for smoothing

#### For Complex Structures
- Use **medial_axis** method for best topology preservation
- Increase **Branch Threshold** to filter small branches
- Adjust **Min Path Length** based on desired detail level

### Common Issues and Solutions

#### Broken Skeletons
- **Problem**: Skeleton has unnecessary breaks
- **Solution**: Increase **Gaussian Blur**, use **medial_axis** method

#### Too Many Branches
- **Problem**: Skeleton has too many small branches
- **Solution**: Increase **Branch Threshold**, increase **Min Path Length**

#### Bridges and Cycles
- **Problem**: Skeleton connects unrelated structures
- **Solution**: Decrease **Gaussian Blur**, use **skeletonize** method

## Tips and Best Practices

### Image Preparation
- Use high-contrast images when possible
- Ensure images are properly binarized
- Avoid very small or very large images

### Parameter Adjustment
- Start with default parameters
- Make small adjustments (one parameter at a time)
- Use the real-time preview to see effects immediately

### Performance
- Large images may take longer to process
- Complex graphs may slow down analysis
- Close unused analysis windows to free memory

### Troubleshooting
- If the application crashes, check image format and size
- Ensure all dependencies are properly installed
- Check console output for error messages

## Examples

### Example 1: Simple Binary Image
1. Load a black and white image
2. Use default parameters
3. Apply **medial_axis** method
4. Adjust **Branch Threshold** to 15-20

### Example 2: Noisy Image
1. Load a noisy image
2. Set **Gaussian Blur** to 9
3. Set **Median Filter** to 7
4. Use **Triangle** thresholding
5. Apply **skeletonize** method

### Example 3: Complex Structure
1. Load a complex diagram
2. Use **medial_axis** method
3. Set **Branch Threshold** to 25
4. Set **Min Path Length** to 10
5. Adjust **Min Node Size** to 5

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure images are in supported formats
4. Check that images are not corrupted

## Version Information

- **Current Version**: 0.1.0
- **Python Compatibility**: 3.8+
- **Dependencies**: OpenCV, scikit-image, skan, NetworkX, PIL, NumPy, Matplotlib




