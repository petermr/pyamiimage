# pyamiimage
Image analysis in Python

See also the Java tools in https://github.com/ContentMine/imageanalysis. This is not a fork, but completely rewritten.

Code is *copied* from https://github.com/petermr/opendiagram, mainly from https://github.com/petermr/openDiagram/tree/master/physchem/python/image

Test-Driven

Documentation on Wiki.

Demos on ipynb Notebooks.

# components

## AmiImage

Tools for image reading, conversion and writing. Most are based on skimage

## branches

* tess_anuv - tesseract to create words and phrases
* nodes_and_islands. analysis of components using sknw and nx_graph. Able to extract arrows and merge with textboxes
Now merged back into main.

Provides a wrapper for `sknw` and `networkX`.

Can extract island primitives such as arrows.


* adding_pipeline - stitching together parts of pyamiimage to form cohesive outputs from images

