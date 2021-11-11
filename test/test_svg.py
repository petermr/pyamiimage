
from pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG, SVGCircle, SVGPath, BBox

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
# Bounding box Tests

    def test_create_empty_bbox(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[],[]]

    def test_create_bbox(self):
        bbox = BBox([[100, 200], [300, 400]])
        assert bbox.xy_ranges == [[100.0, 200.0], [300.0, 400.0]]

    def test_create_bad_bbox(self):
        try:
            BBox([[100, 50], [300, 400]])
        except ValueError as e:
            assert str(e) == "ranges must be increasing 100.0 !<= 50.0"

    def test_update_bbox(self):
        bbox = BBox([[100, 200], [300, 400]])
        bbox.set_xrange([10,20])
        assert bbox.xy_ranges == [[10.0, 20.0], [300.0, 400.0]]

    def test_update_bbox1(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[],[]]
        bbox.set_xrange((10,20))
        assert bbox.xy_ranges == [[10.0, 20.0], []]
        bbox.set_yrange((30,40))
        assert bbox.xy_ranges == [[10.0, 20.0], [30.0, 40.0]]

    def test_get_values(self):
        bbox = BBox()
        assert bbox.get_width() is None
        assert bbox.get_height() is None
        assert bbox.get_xrange() == []
        assert bbox.get_yrange() == []
        bbox.set_xrange([10, 20])
        assert type(bbox.get_xrange()) is list
        assert bbox.get_xrange() == [10., 20.]
        assert bbox.get_width() == 10.
        assert bbox.get_yrange() == []
        assert bbox.get_height() is None
        bbox.set_yrange([30, 70])
        assert bbox.get_xrange() == [10., 20.]
        assert bbox.get_width() == 10.
        assert bbox.get_yrange() == [30., 70.]
        assert bbox.get_height() == 40.

    def test_get_intersections(self):
        bbox0 = BBox([[10, 20], [30, 40]])
        bbox1 = BBox([[13, 28], [27, 38]])
        bbox01 = bbox0.intersect(bbox1)
        assert bbox01.xy_ranges == [[13., 20.], [30., 38.]]

    def test_add_points(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[], []]
        bbox.add_coordinate([1.,2.])
        assert bbox.xy_ranges == [[1., 1.], [2., 2.]]
        bbox.add_coordinate([3., 4.])
        assert bbox.xy_ranges == [[1., 3.], [2., 4.]]
        bbox.add_coordinate([5., 3.])
        assert bbox.xy_ranges == [[1., 5.], [2., 4.]]

    def test_bbox_update(self):
        rect = SVGRect()
        bbox = rect.get_or_create_bbox()
        assert not bbox.is_valid()
        rect.set_xy([1,2])
        rect.set_width(5)
        rect.set_height(10)
        bbox = rect.get_or_create_bbox()
        assert bbox is not None
        assert bbox.is_valid()
        assert bbox.xy_ranges == [[1,6], [2,12]]
        rect.set_width(20)
        bbox = rect.get_or_create_bbox()
        assert bbox.is_valid()
        assert bbox.xy_ranges == [[1,21], [2,12]]

    def test_bbox_bad_values(self):
        rect = SVGRect()
        rect.set_xy([1,2])
        rect.set_height(30)
        rect.set_width(-20)
        bbox = rect.get_or_create_bbox()
        assert not bbox.is_valid()
        rect.set_width(20)
        bbox = rect.get_or_create_bbox()
        assert bbox.is_valid()
        assert bbox.xy_ranges == [[1,21], [2, 32]]

