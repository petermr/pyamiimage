# pyamiimage
`pyamiimage` is a set of tools to extract semantic information from scientific diagrams. 

The current goal is to extract terpene synthase pathway diagrams. 
'Extraction' means that we will go from pixel values in an image to a 'smart diagram'. The output of `pyamiimage` is an image with annotations of substrate, products and enzymes.

We are working to add more support for open formats that encode chemical/pathway information such as [CML](https://www.xml-cml.org/) and [GPML](https://github.com/PathVisio/GPML).

## Requirements

- **Python**: 3.8 or higher
- **Tesseract**: Required for OCR functionality

## Installation

### Prerequisites

#### Tesseract Installation
To run `pyamiimage` on your local system you need to have `Tesseract` installed. If you don't have `Tesseract` installed, install it from [here](https://tesseract-ocr.github.io/tessdoc/).

#### Python Environment
We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install pyamiimage

```bash
# Install from PyPI
pip install pyamiimage

# Or install in development mode from source
git clone https://github.com/petermr/pyamiimage.git
cd pyamiimage
pip install -e .
```

## Usage

`pyamiimage` is a command-line tool and can be accessed via the terminal or command prompt. To bring up the help run:
```bash
pyamiimage --help
```

You can also include pyamiimage in your program using the provided classes.

### AmiImage
AmiImage class provides methods for image manipulation. 
```python
from pyamiimage.ami_image import AmiImage

gray = AmiImage.create_grayscale_from_file(image_file_path)
```

### AmiGraph
AmiGraph class generate a graph from arrows in a diagram.

### AmiOCR
AmiOCR class provides methods to extract words from the image. Uses Tesseract.

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"
pytest -m "not gui"
```

### Building from Source
```bash
# Install build dependencies
pip install build

# Build package
python -m build

# Install built package
pip install dist/*.whl
```

## Timeline
merged main into nodes_and_pixels and re-branched 

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. 
