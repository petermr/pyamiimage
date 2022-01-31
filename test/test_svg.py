import os
from pathlib import Path
from lxml import etree as ET


from ..pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG, SVGCircle, \
    SVGPath, BBox, SVGDefs, SVGMarker, SVGPolygon, SVGArrow


class TestSVG():

    def test_good_attribute(self):
        rect = SVGRect()
        rect.element.attrib["foo"] = "bar"
        assert rect.tostring(pretty_print=False) == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg" foo="bar"/>"""

    def test_bad_attribute(self):
        rect = SVGRect()
        try:
            rect.element.attrib["123"] = "bar"
            assert True, "should throw bad att name"
        except ValueError as e:
            assert str(e) == """Invalid attribute name '123'"""

    def test_create_empty_rect(self):
        svg_rect = SVGRect()
        assert type(svg_rect) is SVGRect
        assert svg_rect.tostring(pretty_print=False) == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg"/>"""

    def test_create_empty_rect_title(self):
        rect = SVGRect()
        title = SVGTitle("title")
        rect.append(title)
        assert rect.tostring(pretty_print=False) == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg"><svg:title title="title"/></svg:rect>"""

    def test_create_rect_w_h(self):
        rect = SVGRect()
        rect.set_height(50)
        rect.set_width(100)
        rect.set_xy((200, 300))
        assert rect.tostring(pretty_print=False) == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg" height="50.0" width="100.0" x="200.0" y="300.0"/>"""

    def test_create_svg_rect_w_h(self):
        svg = SVGSVG()
        rect = SVGRect()
        svg.append(rect)
        rect.set_height(50)
        rect.set_width(100)
        rect.set_xy((200, 300))
        assert svg.tostring(pretty_print=False) == """<svg:svg xmlns:svg="http://www.w3.org/2000/svg" width="1200.0" height="1200.0"><svg:rect height="50.0" width="100.0" x="200.0" y="300.0"/></svg:svg>"""

# Circle
    def test_circle(self):
        circle = SVGCircle(xy=[10, 20], rad=5)
        bbox = circle.get_or_create_bbox()
        assert bbox is not None
        assert circle.is_valid()
        assert bbox.xy_ranges == [[5,15],[15,25]]
        assert circle.tostring(pretty_print=False) == """<svg:circle xmlns:svg="http://www.w3.org/2000/svg" cx="10" cy="20" r="5"/>"""

    def test_write_svg(self):
        svg = SVGSVG()
        circle = SVGCircle(xy=[10, 20], rad=5)
        svg.append(circle)
        with open(Path(os.path.expanduser("~"), "junk.svg"), "wb") as f:
            f.write(ET.tostring(svg.element))

    def test_arrowhead(self):
        svg = SVGSVG()
        arrow = SVGArrow.create_arrowhead(svg, head=[100, 200], tail=[50, 150])
        svg.append(arrow)

        path = Path(Path(__file__).parent.parent, "temp/arrow.svg")
        with open(path, "w") as f:
            f.write(svg.tostring(pretty_print=True))

    def test_get_defs(self):
        svg = SVGSVG()
        defs = svg.get_or_create_defs()
        assert type(defs) is ET._Element
        # check that only one is added
        defs1 = svg.get_or_create_defs()
        assert type(defs1) is ET._Element

    def test_arrowhead(self):
        svg = SVGSVG()
        SVGArrow.create_arrowhead(svg)
        arrow = SVGArrow(head_xy=[50, 75], tail_xy=[150, 200])
        svg.append(arrow)
        # print("SVGXX\n", ET.tostring(svg.element, pretty_print=True).decode())

 #        print("SXX\n", ET.tostring(svg.element).decode())
 #        assert """'<svg:svg xmlns:svg="http://www.w3.org/2000/svg" width="1200.0" '
 # 'height="1200.0"><svg:defs><svg:marker id="arrowhead" markerWidth="10.0" '
 # 'markerHeight="7.0" refX="0.0" refY="3.5" orient="auto"><svg:polygon '
 # 'points="0 0, 10 3.5, 0 7" fill="none" stroke="red" '
 # 'stroke-width="1"/></svg:marker></svg:defs><svg:g><svg:line x1="150" y1="200" '
 # 'x2="50" y2="75" fill="none" stroke="red" stroke-width="1" '
 # 'marker-end="url(#arrowhead)"/></svg:g></svg:svg>'
 #  """ == ET.tostring(svg.element).decode()

    def test_svg_arrow_str(self):
        """
        must not return None as this cannot be stringified later (??)
        :return:
        """
        head_xy = [20, 30]
        tail_xy = [40, 50]
        svg_arrow = SVGArrow(head_xy=head_xy, tail_xy=tail_xy)
        ss = str(svg_arrow)
        assert ss == "tail: 40,50 head: 20,30"

        svg_arrow = SVGArrow()
        ss = str(svg_arrow)
        assert ss == "None"
