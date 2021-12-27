"""Common SVG elements in lightweight code"""
from lxml.etree import ElementTree, Element
import lxml.etree
from abc import abstractmethod, ABC

from ..pyimage.bbox import BBox

"""Seems that it's hard to subclass lxml so this is based on delegation.py
None of the SVG libraries (svgwrite, Cairo) are good for creating subclassed
elements. This is only the common object classes ... at the moment
"""
SVG_NS = "http://www.w3.org/2000/svg"


class AbsSVG(ABC):
    lxml.etree.register_namespace('svg', SVG_NS)

    def __init__(self, tag):
        """
        if bbox is None never calculate it
        if bbox is invalid (e.g. for rect, circle) recalculate when requested
        if parameters are changed, reset bbox to invalid (BBox())
        changes
        :param tag:
        """
        self.element = Element("{"+SVG_NS+"}"+tag)
        # if self.bbox is None this element cannot have a bbox
        self.bbox = None
        # if set use width, height, etc. to calculate bbox (a common default)
        self.calculate_bbox_from_values = True
        # indexes every class object by its lxml delegate
        self.svg_by_lxml = {}

        self.set_fill("none")
        self.set_stroke("red")
        self.set_stroke_width("2")

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

    @abstractmethod
    def calculate_bbox(self):
        """
        only applies to elements with bbox not None
        :return:
        """

    def get_or_create_bbox(self):
        if self.bbox is None or (not self.bbox.is_valid() and self.calculate_bbox_from_values):
            self.calculate_bbox()
        return self.bbox

    def set_fill(self, fill):
        self.fill = fill
        self.set_attribute("file", fill)

    def set_stroke(self, stroke):
        self.stroke = stroke
        self.set_attribute("stroke", stroke)

    def set_stroke_width(self, stroke_width):
        self.stroke_width = stroke_width
        self.set_attribute("stroke-width", stroke_width)


class SVGSVG(AbsSVG):
    TAG = "svg"

    def __init__(self):
        super().__init__(self.TAG)
        self.wrapper_by_lxml = {}  # dictionary of SVG class indexed by wrapped lxml
        self.nxml = 0  # counter of lxml elements
        self.bbox = BBox()
        self.set_width(1200)
        self.set_height(1200)

    def set_width(self, value):
        self.set_float_attribute("width", value)

    def set_height(self, value):
        self.set_float_attribute("height", value)

    def calculate_bbox(self):
        raise NotImplementedError("code not written, BBox should recurse through descendants")


class SVGG(AbsSVG):
    TAG = "g"

    def __init__(self):
        super().__init__(self.TAG)

    def calculate_bbox(self):
        raise NotImplementedError("code not written, BBox should recurse through descendants")


class SVGRect(AbsSVG):
    TAG = "rect"
    HEIGHT = "height"
    WIDTH = "width"
    X = "x"
    Y = "y"

    def __init__(self, bbox=None):
        super().__init__(self.TAG)
        self.xy = None
        self.width = None
        self.height = None
        # BBox is formally independent of the xy,w,h i.e. could be set directly
        self.bbox = bbox

    def set_height(self, h):
        """
        sets height by adjusting existing bbox
        :param h:
        :return:
        """
        if h >= 0:
            self.height = float(h)
            self.set_float_attribute(self.HEIGHT, self.height)
        self.set_invalid()

    def set_width(self, w):
        """
        sets width by adjusting existing bbox
        :param w:
        :return:
        """
        if w >= 0:
            self.width = float(w)
            self.set_float_attribute(self.WIDTH, self.width)
        self.set_invalid()

    def set_xy(self, xy):
        assert len(xy) == 2
        self.xy = [float(xy[0]), float(xy[1])]
        self.set_invalid()
        self.set_float_attribute(self.X, self.xy[0])
        self.set_float_attribute(self.Y, self.xy[1])

    def is_valid(self):
        try:
            float(self.width) >= 0 and float(self.height) >= 0 and \
                float(self.xy[0]) and float(self.xy[1])
            return True
        except ValueError:
            return False

    def set_invalid(self):
        self.bbox = BBox()

    def calculate_bbox(self):
        if self.xy and self.width and self.height and not self.bbox.is_valid():
            self.bbox.set_xrange([self.xy[0], self.xy[0] + self.width])
            self.bbox.set_yrange([self.xy[1], self.xy[1] + self.height])

    def to_xml(self):
        if self.is_valid():
            pass


class SVGCircle(AbsSVG):
    TAG = "circle"

    def __init__(self, xy=None, rad=None):
        super().__init__(self.TAG)
        self.bbox = BBox()
        self.set_xy(xy)
        self.set_rad(rad)

    def calculate_bbox(self):
        try:
            xrange = [self.xy[0] - self.rad, self.xy[0] + self.rad]  
            yrange = [self.xy[1] - self.rad, self.xy[1] + self.rad]  
        except ValueError:
            xrange = None
            yrange = None
        self.bbox = BBox.create_from_ranges(xrange, yrange)

    def is_valid(self):
        try:
            float(self.rad) >= 0 and float(self.xy[0]) and float(self.xy[1])
            return True
        except Exception:
            return False

    def set_rad(self, rad):
        if rad is not None:
            self.rad = rad
            self.set_attribute("r", str(rad))

    def set_xy(self, xy):
        if xy is not None:
            self.xy = xy
            self.set_attribute("cx", str(xy[0]))
            self.set_attribute("cy", str(xy[1]))


class SVGPath(AbsSVG):
    TAG = "path"

    def __init__(self):
        super().__init__(self.TAG)
        self.box = BBox()

    def calculate_bbox(self):
        raise NotImplementedError("code not written, BBox should recurse through descendants")


class SVGLine(AbsSVG):
    TAG = "line"

    def __init__(self, xy1, xy2):
        super().__init__(self.TAG)
        self.box = BBox()
        self.set_xy1(xy1)
        self.set_xy2(xy2)

    def set_xy1(self, xy):
        if xy is not None:
            self.set_attribute("x1", str(xy[0]))
            self.set_attribute("y1", str(xy[1]))

    def set_xy2(self, xy):
        if xy is not None:
            self.set_attribute("x2", str(xy[0]))
            self.set_attribute("y2", str(xy[1]))

    def calculate_bbox(self):
        raise NotImplementedError("code not written")


class SVGText(AbsSVG):
    TAG = "text"

    def __init__(self, xy=None, text=None):
        super().__init__(self.TAG)
        self.box = BBox()
        self.xy = xy
        self.text = text

    def calculate_bbox(self):
        raise NotImplementedError("code not written")


class SVGTitle(AbsSVG):
    TAG = "title"

    def __init__(self, titl=None):
        super().__init__(self.TAG)
        self.set_title(titl)

    def calculate_bbox(self):
        return None

    def set_title(self, titl):
        if titl is not None:
            self.set_attribute("title", titl)

    def get_bounding_box(self):
        """overrides"""
        foo = self
        foo = None
        return foo


class SVGTextBox(SVGG):
    """This will contain text and a shape (for drawing)
    Not an SVG primitive"""

    def __init__(self, svg_text=None, svg_shape=None):
        super().__init__()
        self.box = BBox()
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

    def calculate_bbox(self):
        raise NotImplementedError("code not written, ")


class AmiArrow(SVGG):
    """
    <g>
      <line ... marker=arrowhead"/>
    </g>
    non-standard SVG component
    """
    def __init__(self):
        super().__init__()
        self.line = None

    def calculate_bbox(self):
        raise NotImplementedError("code not written")


class SVGShape(AbsSVG):
    """suoperclass of shapes"""
    TAG = "shape"

    def __init__(self):
        super().__init__(self.TAG)

    def calculate_bbox(self):
        raise NotImplementedError("SVGShape needs a bbox routine")


class XMLNamespaces:
    svg = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"


def is_valid_xy(xy):
    try:
        len(xy) == 2 and float(xy[0]) and float(xy[1])
        return True
    except Exception:
        return False
