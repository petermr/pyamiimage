"""bounding box"""


class BBox:
    """bounding box 2array of 2arrays, based on integers
    """
    def __init__(self, xy_ranges=None):
        """
        Must have a valid bbox
        :param xy_ranges: [[x1, x2], [y1, y2]] will be set to integers
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
        val0 = int(rrange[0])
        val1 = int(rrange[1])
        if val1 < val0:
            raise ValueError(f"ranges must be increasing {val0} !<= {val1}")
        self.xy_ranges[index] = [val0, val1]

    def __str__(self):
        return str(self.xy_ranges)

    def __repr__(self):
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
        assert xy_tuple is not None, f"xy_tuple must have coordinates"
        self.add_to_range(0, self.get_xrange(), xy_tuple[0])
        self.add_to_range(1, self.get_yrange(), xy_tuple[1])

    def add_to_range(self, index, rrange, coord):
        """if coord outside range , expand range

        :param index: 0 or  =1 for axis
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
        width = int(width)
        height = int(height)
        if len(xy) != 2:
            raise ValueError("xy must be an array of 2 values")
        if width < 0 or height < 0:
            raise ValueError("width and height must be non negative")
        xrange =([xy(0), xy[0] + width])
        yrange = [xy(1), xy[1] + int(height)]
        bbox = BBox.create_from_ranges(xrange, yrange)
        return bbox

    def expand_by_margin(self, margin):
        """
        if margin is scalar, apply to both axes
        if margin is 2-tuple, apply to x and y separately
        if margin is negative applies only if current range is >- 2*margin + 1
        i
        :param margin: scalar dx or tuple (dx, dy)
        :return: None
        """

        if not isinstance(margin, list):
            margin = [margin, margin]
        self.change_range(0, margin[0])
        self.change_range(1, margin[1])

    def change_range(self, index, margin):
        """
        change range by margin
        :param index:
        :param margin:
        :return:
        """
        if index != 0 and index != 1:
            raise ValueError(f"Bad index for range {index}")
        rr = self.xy_ranges[index]
        rr[0] -= margin[index]
        rr[1] += margin[index]
        # range cannot be <= 0
        if rr[0] >= rr[1]:
            mid = (rr[0] + rr[1]) / 2
            rr[0] = int(mid - 0.5)
            rr[1] = int(mid + 0.5)
        self.xy_ranges[index] = rr

    @classmethod
    def create_from_ranges(cls, xr, yr):
        """
        create from 2 2-arrays
        :param xr: 2-list of range
        :param yr: 2-list of range
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

    @classmethod
    def get_width_height(cls, bbox):
        """
        TODO MOVED

        :param bbox: tuple of tuples ((x0,x1), (y0,y1))
        :return: (width, height) tuple
        """
        """
        needs to have its own class
        """
        width = bbox[0][1] - bbox[0][0]
        height = bbox[1][1] - bbox[1][0]
        return width, height

    @classmethod
    def fits_within(cls, bbox, bbox_gauge):
        """
        TODO MOVED

        :param bbox: tuple of tuples ((x0,x1), (y0,y1))
        :param bbox_gauge: tuple of (width, height) that bbox must fit in
        :return: true if firs in rectangle
        """
        """
        needs to have its own class
        """
        width, height = bbox.get_width_height()
        return width < bbox_gauge[0] and height < bbox_gauge[1]


"""If you looking for the overlap between two real-valued bounded intervals, then this is quite nice:

def overlap(start1, end1, start2, end2):
    how much does the range (start1, end1) overlap with (start2, end2)
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)
I couldn't find this online anywhere so I came up with this and I'm posting here."""
