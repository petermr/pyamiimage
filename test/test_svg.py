
from pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG

class TestSVG():

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


