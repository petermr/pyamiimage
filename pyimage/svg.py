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

    def __init__(self, bbox=None):
        super().__init__(self.TAG)
        if bbox is not None:
            self.set_bbox(bbox)

    def set_height(self, h):
        self.set_float_attribute("height", h)

    def set_width(self, w):
        self.set_float_attribute("width", w)

    def set_xy(self, tuple2):
        assert len(tuple2) == 2
        self.set_float_attribute("x", tuple2[0])
        self.set_float_attribute("y", tuple2[1])

    def set_bbox(self, bbox_tuple):
        assert bbox_tuple is not None
        assert len(bbox_tuple) == 2
        self.set_float_attribute("x", bbox_tuple[0])
        self.set_float_attribute("y", bbox_tuple[1])


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

class Bbox:
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

    def set_xrange(self, range):
        self.set_range(0, range)

    def get_xrange(self):
        return self.xy_ranges[0]

    def get_width(self):
        return self.get_xrange()[1] - self.get_xrange()[0] if len(self.get_xrange()) == 2 else None

    def set_yrange(self, tuple2):
        self.set_range(1, tuple2)

    def get_yrange(self):
        return self.xy_ranges[1]

    def get_height(self):
        return self.get_yrange()[1] - self.get_yrange()[0] if len(self.get_yrange()) == 2 else None

    def set_range(self, index, range):
        if index != 0 and index != 1:
            raise ValueError(f"bad tuple index {index}")
        val0 = float(range[0])
        val1 = float(range[1])
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
            bbox1 = Bbox((xrange, yrange))
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
            bbox1 = Bbox((xrange, yrange))
        return bbox1

    def intersect_range(self, range0, range1):
        """intersects 2 range tuples"""
        range = ()
        print(range0, range1)
        if len(range0) == 2 and len(range1) == 2:
            range = (max(range0[0], range1[0]), min(range0[1], range1[1]))
        return range

    def add_coordinate(self, xy_tuple):
        self.add_to_range(0, self.get_xrange(), xy_tuple[0])
        self.add_to_range(1, self.get_yrange(), xy_tuple[1])

    def add_to_range(self, index, range, coord):
        """if coord outside range , expand range
        :param range: x or y range
        :param coord: x or y coord
        :return: None (changes range)
        """
        if index != 0 and index != 1:
            raise ValueError(f"bad index {index}")
        if len(range) != 2:
            range = [None, None]
        if range[0] is None or coord < range[0]:
            range[0] = coord
        if range[1] is None or coord > range[1]:
            range[1] = coord
        self.xy_ranges[index] = range
        return range


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



