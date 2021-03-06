# pyamiimage
`pyamiimage` is a set of tools to extract semantic information from scientific diagrams. 

The current goal is to extract terpene synthase pathway diagrams. 
'Extraction' means that we will go from pixel values in an image to a 'smart diagram'. The output of `pyamiimage` is an image with annotations of substrate, products and enzymes.

We are working to add more support for open formats that encode chemical/pathway information such as [CML](https://www.xml-cml.org/) and [GPML](https://github.com/PathVisio/GPML).

## Installation

### Tesseract
To run `pyamiimage` on your local system you need to have `Tesseract` installed. If you don't have `Tesseract` installed, install it from [here](https://tesseract-ocr.github.io/tessdoc/).

```
pip install pyamiimage
```
## Usage

`pyamiimage` is a command-line tool and can be accessed via the terminal or command prompt. To bring up the help run:
```
pyamiimage --help
```

You can also include pyamiimage in your program using the provided classes.

### AmiImage
AmiImage class provides methods for image manipulation. 
```
from pyamiimage.ami_image import AmiImage

gray = AmiImage.create_grayscale_from_file(image_file_path)
```

### AmiGraph
AmiGraph class generate a graph from arrows in a diagram.

### AmiOCR
AmiOCR class provides methods to extract words from the iamge. Uses Tesseract.

## Timeline
merged main into nodes_and_pixels and re-branched 
