# ATPOE Dashboard Technical Design

## Architecture Overview

ATPOE (Advanced Topological Processing and Optimization Engine) is built as a modular desktop application that reuses existing pyamiimage skeletonization code while providing an interactive Tkinter interface for parameter adjustment and visualization.

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    ATPOE Dashboard                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Main      │  │  Parameter  │  │    Graph Viewer     │ │
│  │  Window     │  │   Panel     │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │  Original   │  │  Skeleton   │                         │
│  │  Viewer     │  │   Viewer    │                         │
│  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                pyamiimage Core Modules                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  AmiImage   │  │ AmiSkeleton │  │     AmiGraph        │ │
│  │             │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Relationships

- **Main Window**: Orchestrates all components and manages application state
- **Image Viewers**: Display original and skeleton images with zoom/pan capabilities
- **Parameter Panel**: Provides interactive controls for all processing parameters
- **Graph Viewer**: Displays graph analysis and metrics
- **Core Modules**: Reuse existing pyamiimage functionality

## Core Components

### 1. Main Window (`main_window.py`)

**Responsibilities:**
- Application lifecycle management
- Component coordination
- File I/O operations
- Menu and toolbar management

**Key Methods:**
- `_load_image()`: Load and display images
- `_process_image()`: Apply skeletonization with current parameters
- `_extract_graph()`: Extract NetworkX graph from skeleton
- `_update_viewport_linking()`: Synchronize viewports between viewers

**Dependencies:**
- Tkinter for GUI framework
- pyamiimage modules for core functionality
- PIL for image saving

### 2. Image Viewer (`image_viewer.py`)

**Responsibilities:**
- Image display and manipulation
- Zoom and pan functionality
- Viewport overlay management
- Coordinate conversion

**Key Features:**
- **Zoom Controls**: Mouse wheel, buttons, keyboard shortcuts
- **Pan Controls**: Drag and drop navigation
- **Viewport Overlay**: Rectangle showing current view area
- **Coordinate Display**: Real-time pixel coordinates and values

**Technical Implementation:**
- Uses PIL for image conversion and resizing
- Canvas-based rendering for performance
- Scrollbar integration for large images
- Event binding for mouse and keyboard interactions

### 3. Parameter Panel (`parameter_panel.py`)

**Responsibilities:**
- Parameter input and validation
- Real-time parameter updates
- Parameter preset management
- UI state management

**Parameter Categories:**

#### Skeletonization Parameters
- **Method**: Algorithm selection (medial_axis, skeletonize, thin)
- **Preprocessing**: Gaussian blur, median filter kernels
- **Thresholding**: Method selection and manual values

#### Graph Extraction Parameters
- **Branch Threshold**: Minimum branch length for skan
- **Min Path Length**: Minimum path length filtering
- **Node Filtering**: Minimum node size requirements

**Technical Implementation:**
- Tkinter widgets (Combobox, Scale, Button)
- Variable binding for real-time updates
- Parameter validation and bounds checking
- Callback system for parameter changes

### 4. Graph Viewer (`graph_viewer.py`)

**Responsibilities:**
- Graph metrics display
- Topological analysis
- Advanced graph analysis
- Export functionality

**Metrics Display:**
- **Basic Metrics**: Nodes, edges, density
- **Connectivity**: Components, largest component, connectivity
- **Topology**: Degree distribution, cycles
- **Node Analysis**: End nodes, branch nodes, junctions

**Technical Implementation:**
- NetworkX for graph analysis
- Real-time metric updates
- Detailed analysis in separate windows
- Export to GML/GraphML formats

## Data Flow

### Image Processing Pipeline

```
1. Image Loading
   ┌─────────┐    ┌─────────────┐    ┌─────────────┐
   │ Image   │───▶│  AmiImage   │───▶│  Original   │
   │ File    │    │  Reader     │    │  Viewer     │
   └─────────┘    └─────────────┘    └─────────────┘

2. Parameter Application
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Parameter   │───▶│  AmiImage   │───▶│  Skeleton   │
   │ Panel       │    │ Skeletonizer│    │  Viewer     │
   └─────────────┘    └─────────────┘    └─────────────┘

3. Graph Extraction
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │  Skeleton   │───▶│   AmiGraph  │───▶│   Graph     │
   │  Image      │    │  Extractor  │    │   Viewer    │
   └─────────────┘    └─────────────┘    └─────────────┘
```

### Viewport Linking

```
┌─────────────────────────────────────────────────────────────┐
│                    Viewport Synchronization                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                    ┌─────────────┐        │
│  │  Original   │◄─── Overlay ──────▶│  Skeleton  │        │
│  │  Viewer     │                    │   Viewer   │        │
│  │             │                    │            │        │
│  │ [Image +    │                    │ [Skeleton  │        │
│  │  Red Rect]  │                    │  Image]    │        │
│  └─────────────┘                    └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Technical Implementation Details

### Image Processing

#### Skeletonization Methods
- **medial_axis**: Uses scikit-image's medial_axis transform
- **skeletonize**: Standard morphological skeletonization
- **thin**: Thinning-based approach

#### Preprocessing Pipeline
1. **Grayscale Conversion**: RGB to grayscale if needed
2. **Gaussian Blur**: Noise reduction and smoothing
3. **Median Filter**: Additional noise reduction
4. **Thresholding**: Binarization with multiple methods

#### Graph Extraction
- **skan Library**: Primary skeleton-to-graph conversion
- **NetworkX**: Graph manipulation and analysis
- **Parameter Filtering**: Branch and path length filtering

### GUI Implementation

#### Tkinter Widgets
- **Frame**: Container organization
- **Canvas**: Image display and overlay
- **Scale**: Parameter sliders
- **Combobox**: Method selection
- **Button**: Action triggers
- **Label**: Information display

#### Event Handling
- **Mouse Events**: Click, drag, wheel
- **Keyboard Events**: Shortcuts and navigation
- **Parameter Events**: Real-time updates
- **File Events**: Open, save operations

### Performance Considerations

#### Image Handling
- **Lazy Loading**: Images loaded only when needed
- **Efficient Resizing**: PIL-based resizing with LANCZOS
- **Memory Management**: Proper cleanup of large images

#### Real-time Updates
- **Parameter Validation**: Immediate feedback on invalid values
- **Efficient Rendering**: Canvas-based display updates
- **Background Processing**: Non-blocking parameter application

## Error Handling

### Input Validation
- **Image Format**: Supported format checking
- **Image Size**: Reasonable size limits
- **Parameter Bounds**: Range validation for all parameters

### Exception Handling
- **File I/O**: Graceful handling of file errors
- **Processing Errors**: User-friendly error messages
- **Memory Issues**: Large image handling

### User Feedback
- **Status Bar**: Real-time operation status
- **Error Dialogs**: Clear error messages
- **Progress Indicators**: Processing feedback

## Testing Strategy

### Unit Tests
- **Component Testing**: Individual GUI component tests
- **Parameter Validation**: Parameter range and type testing
- **Image Processing**: Core functionality testing

### Integration Tests
- **Component Integration**: Component interaction testing
- **End-to-End**: Complete workflow testing
- **Parameter Persistence**: Parameter state management

### Test Coverage
- **GUI Components**: All interactive elements
- **Parameter Logic**: All parameter combinations
- **Error Conditions**: Exception handling paths

## Dependencies

### Core Dependencies
```
opencv-python>=4.8.0      # Image I/O and processing
scikit-image>=0.21.0      # Advanced image processing
skan>=0.11.0              # Skeleton-to-graph conversion
networkx>=3.0             # Graph analysis and manipulation
pillow>=10.0.0            # Image manipulation and saving
numpy>=1.24.0             # Numerical operations
matplotlib>=3.7.0         # Advanced visualization
```

### Built-in Dependencies
```
tkinter                   # GUI framework (Python standard library)
unittest                  # Testing framework (Python standard library)
pathlib                   # Path manipulation (Python standard library)
typing                    # Type hints (Python standard library)
```

## Future Enhancements

### Planned Features
- **Batch Processing**: Multiple image processing
- **Parameter Presets**: Save/load parameter combinations
- **Advanced Visualization**: 3D graph visualization
- **Plugin System**: Extensible parameter modules

### Performance Improvements
- **GPU Acceleration**: CUDA/OpenCL support
- **Parallel Processing**: Multi-threaded image processing
- **Caching**: Parameter result caching
- **Lazy Evaluation**: Deferred computation

### User Experience
- **Dark Mode**: Theme support
- **Customizable Layout**: User-defined interface arrangement
- **Keyboard Shortcuts**: Extended shortcut support
- **Help System**: Context-sensitive help

## Deployment

### Distribution
- **Standalone Executable**: PyInstaller packaging
- **Python Package**: pip installable package
- **Source Distribution**: GitHub releases

### Platform Support
- **Windows**: Full support with native look
- **macOS**: Full support with native look
- **Linux**: Full support with native look

### Installation
- **Dependencies**: Automatic dependency resolution
- **Configuration**: User preference persistence
- **Updates**: Automatic update checking

## Conclusion

The ATPOE dashboard provides a comprehensive, user-friendly interface for interactive skeletonization and graph analysis while maintaining the robust functionality of the existing pyamiimage framework. The modular architecture ensures maintainability and extensibility for future enhancements.




