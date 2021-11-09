
from pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG, Bbox

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

# Bounding box Tests

    def test_create_empty_bbox(self):
        bbox = Bbox()
        assert bbox.__str__() == "[(), ()]"

    def test_create_bbox(self):
        bbox = Bbox(((100,200), (300,400)))
        assert bbox.__str__() == "[(100.0, 200.0), (300.0, 400.0)]"

    def test_create_bad_bbox(self):
        try:
            Bbox(((100,50), (300,400)))
        except ValueError as e:
            assert str(e) == "ranges must be increasing 100.0 !<= 50.0"

    def test_update_bbox(self):
        bbox = Bbox(((100,200), (300,400)))
        bbox.set_xrange((10,20))
        assert bbox.__str__() == "[(10.0, 20.0), (300.0, 400.0)]"

    def test_update_bbox(self):
        bbox = Bbox()
        assert bbox.__str__() == "[(), ()]"
        bbox.set_xrange((10,20))
        assert bbox.__str__() == "[(10.0, 20.0), ()]"
        bbox.set_yrange((30,40))
        assert bbox.__str__() == "[(10.0, 20.0), (30.0, 40.0)]"

    def test_get_values(self):
        bbox = Bbox()
        assert bbox.get_width() is None
        assert bbox.get_height() is None
        assert bbox.get_xrange() is ()
        assert bbox.get_yrange() is ()
        bbox.set_xrange((10, 20))
        assert type(bbox.get_xrange()) is tuple
        assert bbox.get_xrange() == (10., 20.)
        assert bbox.get_width() == 10.
        assert bbox.get_yrange() == ()
        assert bbox.get_height() is None
        bbox.set_yrange((30, 70))
        assert bbox.get_xrange() == (10., 20.)
        assert bbox.get_width() == 10.
        assert bbox.get_yrange() == (30., 70.)
        assert bbox.get_height() == 40.

    def test_get_intersections(self):
        bbox0 = Bbox(((10,20), (30,40)))
        bbox1 = Bbox(((13,28), (27,38)))
        bbox01 = bbox0.intersect(bbox1)
        assert bbox01.tuple22 == [(13.,20.), (30.,38.)]


