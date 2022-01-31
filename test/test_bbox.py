
from ..pyimage.svg import SVGRect, SVGTitle, SVGText, SVGTextBox, SVGG, SVGSVG, SVGCircle, SVGPath, BBox


class TestBBox():

# Bounding box Tests

    def test_create_empty_bbox(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[],[]]

    def test_create_bbox(self):
        bbox = BBox([[100, 200], [300, 400]])
        assert bbox.xy_ranges == [[100, 200], [300, 400]]

    def test_create_bad_bbox(self):
        try:
            BBox([[100, 50], [300, 400]])
        except ValueError as e:
            assert str(e) == "ranges must be increasing 100 !<= 50"

    def test_create_from_xy_w_h(self):
        xy =[100, 200]
        width = 150
        height = 27
        bbox = BBox.create_from_xy_w_h(xy, width, height)
        assert bbox.xy_ranges[0] == [100, 250]
        assert bbox.xy_ranges[1] == [200, 227]

    def test_update_bbox(self):
        bbox = BBox([[100, 200], [300, 400]])
        bbox.set_xrange([10,20])
        assert bbox.xy_ranges == [[10, 20], [300, 400]]

    def test_update_bbox1(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[],[]]
        bbox.set_xrange((10,20))
        assert bbox.xy_ranges == [[10, 20], []]
        bbox.set_yrange((30,40))
        assert bbox.xy_ranges == [[10, 20], [30, 40]]

    def test_get_values(self):
        bbox = BBox()
        assert bbox.get_width() is None
        assert bbox.get_height() is None
        assert bbox.get_xrange() == []
        assert bbox.get_yrange() == []
        bbox.set_xrange([10, 20])
        assert type(bbox.get_xrange()) is list
        assert bbox.get_xrange() == [10, 20]
        assert bbox.get_width() == 10
        assert bbox.get_yrange() == []
        assert bbox.get_height() is None
        bbox.set_yrange([30, 70])
        assert bbox.get_xrange() == [10, 20]
        assert bbox.get_width() == 10
        assert bbox.get_yrange() == [30, 70]
        assert bbox.get_height() == 40

    def test_get_intersections(self):
        bbox0 = BBox([[10, 20], [30, 40]])
        bbox1 = BBox([[13, 28], [27, 38]])
        bbox01 = bbox0.intersect(bbox1)
        assert bbox01.xy_ranges == [[13, 20], [30, 38]]
        bbox2 = BBox([[21, 24], [15, 58]])
        bbox02 = bbox0.intersect(bbox2)
        assert bbox02.xy_ranges == [None, [30, 40]]

    def test_get_unions(self):
        bbox0 = BBox([[10, 20], [30, 40]])
        bbox1 = BBox([[13, 28], [27, 38]])
        bbox01 = bbox0.union(bbox1)
        assert bbox01.xy_ranges == [[10, 28], [27, 40]]
        bbox2 = BBox([[21, 24], [15, 58]])
        bbox02 = bbox0.union(bbox2)
        assert bbox02.xy_ranges == [[10, 24], [15, 58]]
 
    def test_add_points(self):
        bbox = BBox()
        assert bbox.xy_ranges == [[], []]
        bbox.add_coordinate([1.,2.])
        assert bbox.xy_ranges == [[1, 1], [2, 2]]
        bbox.add_coordinate([3., 4.])
        assert bbox.xy_ranges == [[1, 3], [2, 4]]
        bbox.add_coordinate([5., 3.])
        assert bbox.xy_ranges == [[1, 5], [2, 4]]

    def test_bbox_update(self):
        rect = SVGRect()
        bbox = rect.get_or_create_bbox()
        assert bbox is None
        # assert not bbox.is_valid()
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


