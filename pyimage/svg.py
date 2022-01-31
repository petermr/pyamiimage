"""Common SVG elements in lightweight code"""
from lxml.etree import ElementTree, Element, XPathSyntaxError
import lxml.etree
from abc import ABC
from pathlib import Path
import logging
# local
try:
    from ..pyimage.bbox import BBox
except:
    from pyimage.bbox import BBox

logger = logging.getLogger(__name__)

"""Seems that it's hard to subclass lxml so this is based on delegation
None of the SVG libraries (svgwrite, Cairo) are good for creating subclassed
elements. We may add functionality to the actual lxml elements
This is only the common object classes ... at the moment

"""
"""
https://lxml.de/api/lxml.builder.ElementMaker-class.html
"""

FILL = "fill"
NONE = "none"
RED = "red"

STROKE = "stroke"
STROKE_WIDTH = "stroke-width"
SVG_NS = "http://www.w3.org/2000/svg"
SVG_NS_PREF = 'svg'

GPML_NS = "http://pathvisio.org/GPML/2013a"

class AbsSVG(ABC):
    lxml.etree.register_namespace(SVG_NS_PREF, SVG_NS)

    def __init__(self, tag):
        """
        if bbox is None never calculate it
        if bbox is invalid (e.g. for rect, circle) recalculate when requested
        if parameters are changed, reset bbox to invalid (BBox())
        changes
        :param tag:
        """
        self.fill = None
        self.stroke = None
        self.stroke_width = None

        # ns_tag = f"{{{SVG_NS}}}{tag}"

        ns_tag = "{" + SVG_NS + "}" + tag
        self.element = Element(ns_tag)
        # if self.bbox is None this element cannot have a bbox
        self.bbox = None
        # if set use width, height, etc. to calculate bbox (a common default)
        self.calculate_bbox_from_values = True
        # indexes every class object by its lxml delegate
        self.svg_by_lxml = {}

    def tostring(self, pretty_print=False):
        s = lxml.etree.tostring(self.element, pretty_print=pretty_print, encoding="UTF-8").decode()
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

    @classmethod
    def get_float_attribute(cls, elem, attname):
        """
        gets attribute value from elem@attname as float
        if elem is list of length 1 takes that
        if it cannot, raises ValueError
        :param elem:
        :param attname:
        :return:
        """
        # att_float = None
        if type(elem) is list and len(elem) == 1:
            elem = elem[0]

        try:
            attval = elem.get(attname)
            att_float = float(attval)
        except Exception as e:
            print("raises", e)
            print(f"type elem = {type(elem)}")
            logger.warning(f"{lxml.etree.tostring(elem)}")
            raise ValueError(f"cannot get float from @{attname}")
        return att_float

    # @abstractmethod
    def calculate_bbox(self):
        """
        only applies to elements with bbox not None
        :return:
        """
        pass

    def get_or_create_bbox(self):
        if self.bbox is None or (not self.bbox.is_valid() and self.calculate_bbox_from_values):
            self.calculate_bbox()
        return self.bbox

    def set_fill(self, fill):
        self.fill = fill
        self.set_attribute(FILL, fill)

    def set_stroke(self, stroke):
        self.stroke = stroke
        self.set_attribute(STROKE, stroke)

    def set_stroke_width(self, stroke_width):
        self.stroke_width = stroke_width
        self.set_attribute(STROKE_WIDTH, stroke_width)


class SVGSVG(AbsSVG):
    TAG = "svg"
    SVG_WIDTH = "width"
    SVG_HEIGHT = "height"

    def __init__(self):
        super().__init__(self.TAG)
        self.wrapper_by_lxml = {}  # dictionary of SVG class indexed by wrapped lxml
        self.nxml = 0  # counter of lxml elements
        self.bbox = BBox()
        self.set_width(1200)
        self.set_height(1200)

    def set_width(self, value):
        self.set_float_attribute(self.SVG_WIDTH, value)

    def set_height(self, value):
        self.set_float_attribute(self.SVG_HEIGHT, value)

    def calculate_bbox(self):
        raise NotImplementedError("code not written, BBox should recurse through descendants")

    def add_arrowhead(self):
        """
        add simple triangular arrowhead
        need to check it hasn't been added
        :return:
        """
        defs_element = self.get_or_create_defs()
        assert defs_element is not None

        id = "arrowhead"
        clark_xpath = f"{{{SVG_NS}}}{SVGMarker.TAG}[@id='{id}']"
        assert clark_xpath == f"{{http://www.w3.org/2000/svg}}marker[@id='arrowhead']", f"should be {clark_xpath}"

        marker_element = ns_xpath(defs_element, clark_xpath, xpath_type="clark")
        if not marker_element:
            marker_element = SVGMarker(id=id, marker_width=10, marker_height=7, refx=0, refy=3.5, orient="auto")
            defs_element.append(marker_element.element)
            polygon = SVGPolygon(points="0 0, 10 3.5, 0 7")
            marker_element.append(polygon)

    def get_or_create_defs(self):

        """
        ensure a single <defs> in <svg>
        :return: the lxml Element (not SVGDefs)
        """
        # svg_defs_element = None

        # local_name_xpath = f"*[local-name()='{SVGDefs.TAG}' and namespace-uri()='{SVG_NS}']"
        clark_xpath = f"{{{SVG_NS}}}{SVGDefs.TAG}"
        assert clark_xpath == "{http://www.w3.org/2000/svg}defs"

        defs_search = ns_xpath(self.element, clark_xpath, xpath_type="clark")
        assert defs_search is not None
        if type(defs_search) is not list:
            defs_search = []

        if len(defs_search) == 0:
            svg_defs = SVGDefs()
            self.append(svg_defs)
            logger.warning("adding single <defs> element")
            svg_defs_element = svg_defs.element
        elif len(defs_search) == 1:
            svg_defs_element = defs_search[0]
        else:
            svg_defs_element = None
            logger.warning(f"more than one <defs> in <svg>")

        if type(svg_defs_element) is list and len(svg_defs_element) == 1:
            svg_defs_element = svg_defs_element[0]
        return svg_defs_element


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

    def __init__(self, bbox=None, xy_ranges=None):
        super().__init__(self.TAG)
        self.xy = None
        self.width = None
        self.height = None
        # BBox is formally independent of the xy,w,h i.e. could be set directly
        self.bbox = bbox

        if self.bbox is not None:
            self.set_xy([bbox.get_xrange()[0], bbox.get_yrange()[0]])
            self.set_height(bbox.get_height())
            self.set_width(bbox.get_width())
        elif xy_ranges is not None:
            try:
                self.set_xy([xy_ranges[0][0], xy_ranges[1][0]])
                self.set_height(xy_ranges[1][1] - xy_ranges[1][0])
                self.set_width(xy_ranges[0][1] - xy_ranges[0][0])
            except Exception as e:
                logger.error(f"Bad xyranges {xy_ranges} should be two pairs of numbers")
                raise ValueError(e)


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

    # @classmethod
    # def create_svg(cls, bbox):
    #     """
    #     create SVGRect from bbox
    #     :param bbox:
    #     :return: svg_rect or None
    #     """
    #     if bbox is None:
    #         return None
    #     SVGRect()


class SVGCircle(AbsSVG):
    TAG = "circle"

    RAD = "r"

    def __init__(self, xy=None, rad=None):
        super().__init__(self.TAG)
        self.bbox = BBox()
        self.rad = None
        self.xy = None
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
            self.set_attribute(self.RAD, str(rad))

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
    X1 = "x1"
    X2 = "x2"
    Y1 = "y1"
    Y2 = "y2"

    def __init__(self, xy1, xy2):
        super().__init__(self.TAG)
        self.box = BBox()
        self.set_xy1(xy1)
        self.set_xy2(xy2)
        set_default_styles(self)

    def __repr__(self):
        """[x1, y1], [x2, y2]"""
        s = f"[{self.get_float_attribute(self, self.X1)}, {self.get_float_attribute(self, self.Y1)}], " \
            f"[{self.get_float_attribute(self, self.Y1)}, {self.get_float_attribute(self, self.Y2)}]"
        return s

    def set_xy1(self, xy):
        if xy is not None:
            self.set_attribute(self.X1, str(xy[0]))
            self.set_attribute(self.Y1, str(xy[1]))

    def set_xy2(self, xy):
        if xy is not None:
            self.set_attribute(self.X2, str(xy[0]))
            self.set_attribute(self.Y2, str(xy[1]))

    def calculate_bbox(self):
        raise NotImplementedError("code not written")


class SVGText(AbsSVG):
    TAG = "text"

    X = "x"
    Y = "y"

    def __init__(self, xy=None, text=None):
        super().__init__(self.TAG)
        self.box = BBox()
        self.xy = xy
        self.text = text

    def set_xy(self, xy):
        if xy is not None:
            self.set_attribute(self.X, str(xy[0]))
            self.set_attribute(self.Y, str(xy[1]))

    def calculate_bbox(self):
        raise NotImplementedError("code not written")


class SVGTitle(AbsSVG):
    TAG = "title"

    TITLE = "title"

    def __init__(self, titl=None):
        super().__init__(self.TAG)
        self.set_title(titl)

    def calculate_bbox(self):
        return None

    def set_title(self, titl):
        if titl is not None:
            self.set_attribute(self.TITLE, titl)

    def get_bounding_box(self):
        """overrides"""
        foo = None
        return foo


class SVGTextBox(SVGG):
    """This will contain text and a shape (for drawing)
    Not an SVG primitive"""

    def __init__(self, svg_text=None, svg_shape=None):
        super().__init__()
        self.box = BBox()
        self.svgg = SVGG()
        self.shape = None
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

    def set_shape(self, shape):
        if type(shape) is SVGRect or type(shape) is SVGCircle:
            self.shape = shape

    def calculate_bbox(self):
        raise NotImplementedError("code not written, ")


class SVGDefs(AbsSVG):
    TAG = "defs"

    def __init__(self):
        super().__init__(self.TAG)


class SVGMarker(AbsSVG):
    """
 <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7"
    refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" />
    </marker>
  </defs>
      """
    TAG = "marker"

    ID = "id"
    MARKER_WIDTH = "markerWidth"
    MARKER_HEIGHT = "markerHeight"
    MARKER_END = "marker-end"
    MARKER_START = "marker-start"
    REFX = "refX"
    REFY = "refY"
    ORIENT = "orient"
    AUTO = "auto"
    ORIENT_VALUES = [AUTO]

    def __init__(self, id, marker_width=None, marker_height=None, refx=None, refy=None, orient=AUTO):
        super().__init__(self.TAG)
        self.id = None
        self.set_id(id)
        self.set_marker_width(marker_width)
        self.set_marker_height(marker_height)
        self.set_refx(refx)
        self.set_refy(refy)
        self.set_orient(orient)

    def set_id(self, value):
        assert value is not None
        self.id = value
        self.set_attribute(self.ID, value)

    def set_marker_width(self, value):
        if value is not None:
            self.set_float_attribute(self.MARKER_WIDTH, float(value))

    def set_marker_height(self, value):
        if value is not None:
            self.set_float_attribute(self.MARKER_HEIGHT, float(value))

    def set_refx(self, value):
        if value is not None:
            self.set_float_attribute(self.REFX, float(value))

    def set_refy(self, value):
        if value is not None:
            self.set_float_attribute(self.REFY, float(value))

    def set_orient(self, value):
        if value in self.ORIENT_VALUES:
            self.set_attribute(self.ORIENT, value)

    def set_marker_start(self, svg_element):
        svg_element.set_attribute(self.MARKER_START, "#" + self.id)


class SVGPolygon(AbsSVG):
    TAG = "polygon"

    POINTS = "points"

    def __init__(self, points=None):
        super().__init__(self.TAG)
        self.set_points(points)
        set_default_styles(self)

    # need to check/convert to float_array
    def set_points(self, points):
        if points is not None:
            self.set_attribute(self.POINTS, points)


class SVGArrow(SVGG):
    X1 = "x1"
    X2 = "x2"
    Y1 = "y1"
    Y2 = "y2"

    MIN_WIDTH = 20

    def __init__(self, head_xy=None, tail_xy=None):
        """
        SVGArrow requires the <svg> element to have a <defs> elememt

        Typically:
        svg = SVGSVG()
        SVGArrow.create_arrowhead(svg)
        g = SVGG()
        svg.append(g)
        for i, island in enumerate(big_islands):
            ami_arrow = AmiArrow.create_simple_arrow(island)
            g.append(ami_arrow.get_svg())

        :param head_xy:
        :param tail_xy:
        """
        super().__init__()
        self.tail_xy = tail_xy
        self.head_xy = head_xy
        self.line = None
        if self.head_xy and self.tail_xy:
            self.line = SVGLine(xy1=tail_xy, xy2=head_xy)
            self.line.set_attribute(SVGMarker.MARKER_END, "url(#arrowhead)")
            self.append(self.line)

    def __str__(self):
        if self.tail_xy is None or self.head_xy is None:
            return "None"
        s0 = "tail: " + str(self.tail_xy[0]) + "," + str(self.tail_xy[1])  \
            if self.tail_xy[0] is not None and self.tail_xy[1] is not None else "None"
        s1 = " head: " + str(self.head_xy[0]) + "," + str(self.head_xy[1]) \
            if self.head_xy[0] is not None and self.head_xy[1] is not None else "None"
        return s0 + s1

    def set_head_xy(self, xy):
        self.head_xy = xy

    def set_tail_xy(self, xy):
        self.tail_xy = xy

    @classmethod
    def create_arrowhead(cls, svgsvg):
        svgsvg.add_arrowhead()

    def calculate_bbox(self):
        """
        assume that width has a minimum value if aligned with axes
        :return:
        """
        bbox = BBox.create_from_corners(self.tail_xy, self.head_xy)


    @classmethod
    def create_from_svgg(cls, svgg):
        """
        expects an <svg:g role="arrow"><svg:line x1,y1 x2,y2></svg:g> object
        x1,y1 is the tail and x2,y2 is the head
        svg:marker is not required
        Example:
        <svg:g role="arrow">
            <svg:line x1="241" y1="669" x2="243" y2="757" ... />
        </svg:g>
        maybe mainly for testing
        :param svgg:
        :return: SVGArraw object or None if svgg is None or Errors
        """
        if svgg is None:
            return None
        assert type(svgg) is lxml.etree._Element, "svgg must be namespaced element"
        assert svgg.get("role") == "arrow", "must have role='arrow'"
        xpath = f"{{{SVG_NS}}}line"
        svg_line = ns_xpath(svgg, xpath=xpath)
        svg_arrow = SVGArrow()
        x1 = cls.get_float_attribute(svg_line, cls.X1)
        x2 = cls.get_float_attribute(svg_line, cls.X2)
        y1 = cls.get_float_attribute(svg_line, cls.Y1)
        y2 = cls.get_float_attribute(svg_line, cls.Y2)
        svg_arrow.tail_xy = [x1, y1]
        svg_arrow.head_xy = [x2, y2]

        return svg_arrow

class SVGUtil:
    """mainly classmethods"""
    @classmethod
    def write_to_path(cls, svg_object, path, pretty_print=True):
        """write to path"""
        with open(str(Path(path)), "w") as f:
            f.write(svg_object.tostring(pretty_print=pretty_print))


def set_default_styles(svg_element):
    """
    sets fill, stroke, stroke-width to default so that at least the elements
    is visibly displayed
    :param svg_element:
    :return:
    """
    assert svg_element is not None
    svg_element.set_fill(NONE)
    svg_element.set_stroke(RED)
    svg_element.set_stroke_width("1")


def ns_xpath(element, xpath, xpath_type="Clark", unpack_1_elem_lists=True):
    """
    searches an element with possibly namespaced Xpath
    here we use SVG namespace as an f-string substitution
    Two styles:
    * local_name_xpath = f"*[local-name()='defs' and namespace-uri()='{SVG_NS}']"
    * clark_xpath = f"{{{SVG_NS}}}defs"

    They use different tools (xpath and ETXPath) and different syntaxes

    :param element: element to search
    :param xpath: xpath expression
    :param xpath_type: "Clark" or "local"
    :param unpack_1_elem_lists: convert returned list [Foo] to Foo (default True)
    :return: result of xpath (bool or number or string, or list)
    """
    assert element is not None
    assert xpath is not None
    assert xpath_type is not None

    return_val = None
    if xpath_type is None:
        raise ValueError("xpath_type cannot be None")
    elif xpath_type.lower() == "local":
        try:
            return_val = element.xpath(xpath)
        except XPathSyntaxError as e:
            logger.error(f"XPATH error {e} in {xpath}")
            raise e
    elif xpath_type.lower() == "clark":
        try:
            return_val = lxml.etree.ETXPath(xpath)(element)
        except XPathSyntaxError as e:
            logger.error(f"XPATH error {e} in {xpath}")
            raise e
    else:
        logger.error(f"xpath_type must be Clark or local")

    # unpack 1-element lists?
    if unpack_1_elem_lists and type(return_val) is list and len(return_val) == 1:
        return_val = return_val[0]
    return return_val


class XMLNamespaces:
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"


# --- utils ---

def is_valid_xy(xy):
    try:
        len(xy) == 2 and float(xy[0]) and float(xy[1])
        return True
    except Exception:
        return False
