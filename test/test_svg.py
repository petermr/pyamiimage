import os
from pathlib import Path
from lxml import etree as ET


from ..pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG, SVGCircle, SVGPath, BBox


class TestSVG():

    def test_good_attribute(self):
        rect = SVGRect()
        rect.element.attrib["foo"] = "bar"
        assert rect.tostring() == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg" foo="bar"/>
"""

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
        assert svg_rect.tostring() == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg"/>
"""

    def test_create_empty_rect_title(self):
        rect = SVGRect()
        title = SVGTitle("title")
        rect.append(title)
        assert rect.tostring() == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg">
  <svg:title title="title"/>
</svg:rect>
"""

    def test_create_rect_w_h(self):
        rect = SVGRect()
        rect.set_height(50)
        rect.set_width(100)
        rect.set_xy((200, 300))
        assert rect.tostring() == """<svg:rect xmlns:svg="http://www.w3.org/2000/svg" height="50.0" width="100.0" x="200.0" y="300.0"/>
"""

    def test_create_svg_rect_w_h(self):
        svg = SVGSVG()
        rect = SVGRect()
        svg.append(rect)
        rect.set_height(50)
        rect.set_width(100)
        rect.set_xy((200, 300))
        assert svg.tostring() == \
"""<svg:svg xmlns:svg="http://www.w3.org/2000/svg">
  <svg:rect height="50.0" width="100.0" x="200.0" y="300.0"/>
</svg:svg>
"""
# Circle
    def test_circle(self):
        circle = SVGCircle(xy=[10, 20], rad=5)
        bbox = circle.get_or_create_bbox()
        print("bbox ", bbox)
        assert bbox is not None
        assert circle.is_valid()
        assert bbox.xy_ranges == [[5,15],[15,25]]

    def test_write_svg(self):
        svg = SVGSVG()
        circle = SVGCircle(xy=[10, 20], rad=5)
        svg.append(circle)
        with open(Path(os.path.expanduser("~"), "junk.svg"), "wb") as f:
            f.write(ET.tostring(svg.element))


