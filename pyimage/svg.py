"""Common SVG elements in lightweight code"""
from lxml.etree import ElementTree, Element
import lxml.etree
from abc import abstractmethod, ABC

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
        if self.bbox is not None and not self.bbox.is_valid() and self.calculate_bbox_from_values:
            self.calculate_bbox()
        return self.bbox


class SVGSVG(AbsSVG):
    TAG = "svg"

    def __init__(self):
        super().__init__(self.TAG)
        self.wrapper_by_lxml = {}  # dictionary of SVG class indexed by wrapped lxml
        self.nxml = 0  # counter of lxml elements
        self.bbox = BBox()

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
        self.bbox = BBox()

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
        self.xy = xy
        self.rad = rad

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
        self.rad = rad

    def set_xy(self, xy):
        self.xy = xy


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
        self.xy1 = xy1
        self.xy2 = xy2

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
        return None


class SVGTextBox(SVGG):
    """This will contain text and a shape (for drawing)"""

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


class BBox:
    """bounding box tuple2 of tuple2s
    """
    def __init__(self, xy_ranges=None):
        """
        Must have a valid bbox
        :param xy_ranges:
        """
        self.xy_ranges = [[], []]
        if xy_ranges is not None:
            self.set_ranges(xy_ranges)

    def set_ranges(self, xy_ranges):
        if xy_ranges is None:
            raise ValueError("no lists given")
        if len(xy_ranges) != 2:
            raise ValueError("must be 2 lists of lists")
        if len(xy_ranges[0]) != 2 or len(xy_ranges[1]) != 2:
            raise ValueError("each child list must be a 2-list")
        self.set_xrange(xy_ranges[0])
        self.set_yrange(xy_ranges[1])

    def set_xrange(self, rrange):
        self.set_range(0, rrange)

    def get_xrange(self):
        return self.xy_ranges[0]

    def get_width(self):
        return self.get_xrange()[1] - self.get_xrange()[0] if len(self.get_xrange()) == 2 else None

    def set_yrange(self, rrange):
        self.set_range(1, rrange)

    def get_yrange(self):
        return self.xy_ranges[1]

    def get_height(self):
        return self.get_yrange()[1] - self.get_yrange()[0] if len(self.get_yrange()) == 2 else None

    def set_range(self, index, rrange):
        if index != 0 and index != 1:
            raise ValueError(f"bad tuple index {index}")
        val0 = float(rrange[0])
        val1 = float(rrange[1])
        if val1 < val0:
            raise ValueError(f"ranges must be increasing {val0} !<= {val1}")
        self.xy_ranges[index] = [val0, val1]

    def __str__(self):
        return str(self.xy_ranges)

    def intersect(self, bbox):
        """
        inclusive intersection of boxes (AND)
        if any fields are empty returns None
        :param bbox:
        :return: new Bbox (max(min) ... (min(max)) or None if any  errors
        """
        bbox1 = None
        if bbox is not None:
            xrange = self.intersect_range(self.get_xrange(), bbox.get_xrange())
            yrange = self.intersect_range(self.get_yrange(), bbox.get_yrange())
            bbox1 = BBox((xrange, yrange))
        return bbox1

    def union(self, bbox):
        """
        inclusive merging of boxes (OR)
        if any fields are empty returns None
        :param bbox:
        :return: new Bbox (min(min) ... (max(max)) or None if any  errors
        """
        bbox1 = None
        if bbox is not None:
            xrange = self.union_range(self.get_xrange(), bbox.get_xrange())
            yrange = self.union_range(self.get_yrange(), bbox.get_yrange())
            bbox1 = BBox((xrange, yrange))
        return bbox1

    @classmethod
    def intersect_range(cls, range0, range1):
        """intersects 2 range tuples"""
        rrange = ()
        print(range0, range1)
        if len(range0) == 2 and len(range1) == 2:
            rrange = (max(range0[0], range1[0]), min(range0[1], range1[1]))
        return rrange

    @classmethod
    def union_range(cls, range0, range1):
        """intersects 2 range tuples"""
        rrange = []
        if len(range0) == 2 and len(range1) == 2:
            rrange = [min(range0[0], range1[0]), max(range0[1], range1[1])]
        return rrange

    def add_coordinate(self, xy_tuple):
        self.add_to_range(0, self.get_xrange(), xy_tuple[0])
        self.add_to_range(1, self.get_yrange(), xy_tuple[1])

    def add_to_range(self, index, rrange, coord):
        """if coord outside range , expand range
        :param index:
        :param rrange: x or y range
        :param coord: x or y coord
        :return: None (changes range)
        """
        if index != 0 and index != 1:
            raise ValueError(f"bad index {index}")
        if len(rrange) != 2:
            rrange = [None, None]
        if rrange[0] is None or coord < rrange[0]:
            rrange[0] = coord
        if rrange[1] is None or coord > rrange[1]:
            rrange[1] = coord
        self.xy_ranges[index] = rrange
        return rrange

    @classmethod
    def create_box(cls, xy, width, height):
        if xy is None or width is None or height is None:
            raise ValueError("All params must be not None")
        width = float(width)
        height = float(height)
        if len(xy) != 2:
            raise ValueError("xy must be an array of 2 values")
        if width < 0 or height < 0:
            raise ValueError("width and height must be non negative")
        xrange = float(xy[0]) + float(width)
        yrange = float(xy[1]) + float(height)
        bbox = BBox.create_from_ranges(xrange, yrange)
        return bbox

    @classmethod
    def create_from_ranges(cls, xr, yr):
        """
        create from 2 2-arrays
        :param xr:
        :param yr:
        :return:
        """
        bbox = BBox()
        bbox.set_xrange(xr)
        bbox.set_yrange(yr)
        return bbox

    def is_valid(self):
        """
        both ranges must be present and non-negative
        :return:
        """
        if self.xy_ranges is None or len(self.xy_ranges) != 2:
            return False
        try:
            ok = self.get_width() >= 0 or self.get_height() >= 0
            return ok
        except Exception:
            return False

    def set_invalid(self):
        """set xy_ranges to None"""
        self.xy_ranges = None


"""If you looking for the overlap between two real-valued bounded intervals, then this is quite nice:

def overlap(start1, end1, start2, end2):
    how much does the range (start1, end1) overlap with (start2, end2)
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)
I couldn't find this online anywhere so I came up with this and I'm posting here."""


class AmiArrow(SVGG):
    """
    <g>
      <line ... marker=arrowhead"/>
    </g>
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


def is_valid_xy(xy):
    try:
        len(xy) == 2 and float(xy[0]) and float(xy[1])
        return True
    except Exception:
        return False
