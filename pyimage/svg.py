"""Common SVG elements in lightweight code"""
from lxml.etree import ElementTree, Element
import lxml.etree

"""Seems that it's hard to subclass lxml so this is based on delegation.py
None of the SVG libraries (svgwrite, Cairo) are good for creating subclassed
elements. This is only the common object classes ... at the moment
"""
SVG_NS = "http://www.w3.org/2000/svg"
class AbsSVG():
    lxml.etree.register_namespace('svg', SVG_NS)

    def __init__(self, tag):
        self.element = Element("{"+SVG_NS+"}"+tag)

    def tostring(self):
        s = lxml.etree.tostring(self.element, pretty_print=True, encoding="UTF-8").decode()
        return s

    def append(self, elem):
        self.element.append(elem.element)

    def set_attribute(self, name, value):
        if name is not None and value is not None:
            if type(name) is str and name[0].isalpha():
                self.element.attrib[name] = value

    def set_int_attribute(self, name, val):
        self.set_attribute(name, str(int(val)))

    def set_float_attribute(self, name, val):
        self.set_attribute(name, str(float(val)))


class SVGSVG(AbsSVG):
    TAG = "svg"
    def __init__(self):
        super().__init__(self.TAG)


class SVGG(AbsSVG):
    TAG = "g"
    def __init__(self):
        super().__init__(self.TAG)



class SVGShape(AbsSVG):
    TAG = "shape"
    def __init__(self):
        super().__init__(self.TAG)

    def get_bbox(self):
        pass


class SVGRect(AbsSVG):
    TAG = "rect"

    def __init__(self):
        super().__init__(self.TAG)

    def set_height(self, h):
        self.set_float_attribute("height", h)

    def set_width(self, w):
        self.set_float_attribute("width", w)

    def set_xy(self, tuple2):
        assert len(tuple2) == 2
        self.set_float_attribute("x", tuple2[0])
        self.set_float_attribute("y", tuple2[1])


class SVGPath(AbsSVG):
    TAG = "path"
    def __init__(self):
        super().__init__(self.TAG)


class SVGLine(AbsSVG):
    TAG = "line"
    def __init__(self):
        super().__init__(self.TAG)


class SVGCircle(AbsSVG):
    TAG = "circle"
    def __init__(self):
        super().__init__(self.TAG)


class SVGText(AbsSVG):
    TAG = "text"
    def __init__(self):
        super().__init__(self.TAG)


class SVGTitle(AbsSVG):
    TAG = "title"

    def __init__(self, titl=None):
        super().__init__(self.TAG)
        self.set_title(titl)

    def set_title(self, titl):
        if titl is not None:
            self.set_attribute("title", titl)


class SVGTextBox(SVGG):
    """This will contain text and a shape (for drawing)"""

    def __init__(self, svg_text=None, svg_shape=None):
        super().__init__()
        self.svgg = SVGG()
        self.set_text(svg_text)
        self.set_shape(svg_shape)

    def set_text(self, svg_text):
        """
        Need to check for uniqueness
        :param svg_text:
        :return:
        """
        if svg_text is not None:
            assert type(svg_text) is SVGText
            self.svgg.append(svg_text)

    def set_shape(self, svg_shape):
        if svg_shape is not None:
            assert type(svg_shape) is SVGShape
            self.svgg.append(svg_shape)


class AmiArrow(SVGG):
    """
    <g>
      <line ... marker=arrowhead"/>
    </g>
    """
    def __init__(self):
        super().__init__()
        self.line = None



