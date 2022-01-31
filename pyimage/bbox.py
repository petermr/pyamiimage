"""bounding box"""
from skimage import draw
# from ..pyimage.svg import SVGG, SVGRect

class BBox:
    """bounding box 2array of 2arrays, based on integers
    """

    X = "x"
    Y = "y"
    WIDTH = "width"
    HEIGHT = "height"

    def __init__(self, xy_ranges=None, swap_minmax=False):
        """
        Must have a valid bbox
        Still haven'tb worked out logic of default boxes (must include None's)
        as [0.0],[0.0] is valid
        :param xy_ranges: [[x1, x2], [y1, y2]] will be set to integers
        """
        self.xy_ranges = [[], []]
        self.swap_minmax = swap_minmax
        if xy_ranges is not None:
            self.set_ranges(xy_ranges)

    @classmethod
    def create_from_xy_w_h(cls, xy, width, height):
        """
        create from xy, width height
        all inputs must be floats
        :param xy: origin a [float, float]
        :param width:
        :param height:
        :return:
        """
        assert type(xy[0]) is float or type(xy[0]) is int, f"found {type(xy[0])}"
        assert type(xy[1]) is float or type(xy[1]) is int, f"found {type(xy[1])}"
        assert type(width) is float or type(width) is int, f"found {type(width)}"
        assert type(height) is float or type(height) is int, f"found {type(height)}"

        try:
            xy_ranges = [[float(xy[0]), float(xy[0]) + float(width)], [float(xy[1]), float(xy[1]) + float(height)]]
        except Exception as e:
            raise ValueError(f"cannot create bbox from {xy},{width},{height}")
        return BBox(xy_ranges=xy_ranges)

    def set_ranges(self, xy_ranges):
        if xy_ranges is None:
            raise ValueError("no lists given")
        if len(xy_ranges) != 2:
            raise ValueError("must be 2 lists of lists")
        if xy_ranges[0] is not None and len(xy_ranges[0]) == 0:
            xy_ranges[0] = None
        if xy_ranges[0] is not None and len(xy_ranges[0]) != 2:
            raise ValueError(f"range {xy_ranges[0]} must be None or 2-tuple")
        if xy_ranges[1] is not None and len(xy_ranges[1]) == 0:
            xy_ranges[1] = None
        if xy_ranges[1] is not None and len(xy_ranges[1]) != 2:
            raise ValueError(f"range {xy_ranges[1]} must be None or 2-tuple")
        self.set_xrange(xy_ranges[0])
        self.set_yrange(xy_ranges[1])

    def get_ranges(self):
        """gets ranges as [xrange, yrange]"""
        return self.xy_ranges

    def set_xrange(self, rrange):
        self.set_range(0, rrange)

    def get_xrange(self):
        return self.xy_ranges[0]

    def get_width(self):
        """get width
        :return: width or None if x range invalid or not set"""
        if self.get_xrange() is None or len(self.get_xrange()) == 0:
            return None;
        assert self.get_xrange() is not None
        assert len(self.get_xrange()) == 2, f"xrange, got {len(self.get_xrange())}"
        return self.get_xrange()[1] - self.get_xrange()[0]

    def set_yrange(self, rrange):
        self.set_range(1, rrange)

    def get_yrange(self):
        return self.xy_ranges[1]

    def get_height(self):
        if self.get_yrange() is None or len(self.get_yrange()) == 0:
            return None
        return self.get_yrange()[1] - self.get_yrange()[0] if len(self.get_yrange()) == 2 else None

    def set_range(self, index, rrange):
        if index != 0 and index != 1:
            raise ValueError(f"bad tuple index {index}")
        if rrange is None:
            self.xy_ranges[index] = None
            return
        val0 = int(rrange[0])
        val1 = int(rrange[1])
        if val1 < val0:
            if self.swap_minmax:
                val1, val0 = val0, val1
            else:
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
        DOES NOT CHANGE self
        :param bbox:
        :return: new Bbox (max(min) ... (min(max)) or None if any  errors
        """
        bbox1 = None
        if bbox is not None:
            # print("XRanges",self.get_xrange(), bbox.get_xrange())
            xrange = self.intersect_range(self.get_xrange(), bbox.get_xrange())
            # print("XRange", xrange)
            # print("YRanges",self.get_yrange(), bbox.get_yrange())
            yrange = self.intersect_range(self.get_yrange(), bbox.get_yrange())
            # print("YRange", yrange)
            bbox1 = BBox([xrange, yrange])
        return bbox1

    def union(self, bbox):
        """
        inclusive merging of boxes (OR)
        if any fields are empty returns None
        DOES NOT CHANGE self

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
        if len(range0) == 2 and len(range1) == 2:
            maxmin = max(range0[0], range1[0])
            minmax = min(range0[1], range1[1])
            rrange = [maxmin, minmax] if minmax >= maxmin else None
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
        xrange = ([xy(0), xy[0] + width])
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
        if self.xy_ranges[0] is None or self.xy_ranges[1] is None:
            return False
        try:
            ok = self.get_width() >= 0 and self.get_height() >= 0
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
        :return: true if fits in rectangle
        """
        """
        needs to have its own class
        """
        width, height = bbox.get_width_height()
        return width < bbox_gauge[0] and height < bbox_gauge[1]

    def min_dimension(self):
        """
        gets minimum of height and width
        :return: min(height, width)
        """
        return min(self.get_width(), self.get_height())

    def max_dimension(self):
        """
        gets maximum of height and width
        :return: max(height, width)
        """
        return max(self.get_width(), self.get_height())

    def get_point_pair(self):
        """
        BBox stores the location as ranges of x and y values:
        [[x1, x2], [y1, y2]]
        sometimes it is necessary to work with points instead:
        [(y1, x1), (y2, x2)]
        :returns: list of 2 tuples
        """
        return [(self.get_yrange()[0], self.get_xrange()[0]),
         (self.get_yrange()[1], self.get_xrange()[1])]
         # remember that the indexing is in terms of rows and columns
         # hence x(columns) y(rows) values are flipped when returning point pair

    @classmethod
    def plot_bbox_on(cls, image, bbox):
        """
        Plots bbox on an image
        :param: image
        :type: numpy array 
        :param: bbox
        :type: BBox or list
        :returns: fig, ax
        """
        # bbox can either be BBox object or in form of [[a, b][c, d]]
        
        # if type(bbox) == BBox:
        #     assert bbox.is_valid()
        # elif type(bbox) == list:
        #     bbox = BBox(bbox)
        #     assert bbox.is_valid()
        # else:
        #     # the bbox passed is not invalid
        #     return None


        point_pair = bbox.get_point_pair()
        if point_pair[0][0] > image.shape[0] or point_pair[0][1] >image.shape[1]:
            # if the starting point is outside the image, ignore bbox
            return image
        
        try:
            row, col = draw.rectangle_perimeter(start=point_pair[0], end=point_pair[1])
            image[row, col] = 0
        except IndexError as e:
            point_pair = BBox.fit_point_pair_within_image(image, point_pair)
            row, col = draw.rectangle_perimeter(start=point_pair[0], end=point_pair[1])
            image[row, col] = 0

        return image

    @classmethod
    def fit_point_pair_within_image(cls, image, point_pair):
        max_row = image.shape[0]
        max_col = image.shape[1]
        bbox_row = point_pair[1][0]
        bbox_col = point_pair[1][1]
        if bbox_row >= max_row-1:
            bbox_row = max_row - 2
        if bbox_col >= max_col-1:
            bbox_col = max_col - 2
        point_pair[1] = (bbox_row, bbox_col)
        return point_pair

    # RECURSIVE imports...
    # def create_svg(self):
    #     """creates SVG (a <g> with a <rect>
    #     :return: <g role="bbox"><rect .../></g>
    #     """
    #     g = SVGG()
    #     g.set_attribute("role", "bbox")
    #     svg_rect = SVGRect(self)
    #     g.append(svg_rect)
    #     return g

    @classmethod
    def create_from_corners(cls, xy1, xy2):
        if xy1 is None or xy2 is None:
            return None
        if len(xy1) != 2 or len(xy2) != 2:
            return None
        xrange = [xy1[0], xy2[0]] if xy2[0] > xy1[0] else [xy2[0], xy1[0]]
        yrange = [xy1[1], xy2[1]] if xy2[1] > xy1[1] else [xy2[1], xy1[1]]
        bbox = BBox(xy_ranges=[xrange, yrange])
        return bbox


"""If you looking for the overlap between two real-valued bounded intervals, then this is quite nice:

def overlap(start1, end1, start2, end2):
    how much does the range (start1, end1) overlap with (start2, end2)
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)
I couldn't find this online anywhere so I came up with this and I'm posting here."""
