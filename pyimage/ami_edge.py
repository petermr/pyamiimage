"""wrapper for edge from sknw/nx, still being developed"""

from pyimage.svg import BBox


class AmiEdge:
    def __init__(self):
        self.points_xy = None
        self.bbox = None

    def read_nx_edge_points_yx(self, points_array_yx):
        """
        convert from nx_points (held as yarray, xarray) to array(x, y)
        :param points_array_yx:
        :return:
        """
        # points are in separate columns (y, x)
        # print("coord", points[:, 1], points[:, 0], 'green')
        # I can't get list comprehension to work, help needed!
        # self.points_xy = [   [point[1], point[0]] for point in points_array_yx]:
        assert points_array_yx is not None and points_array_yx.ndim == 2 and points_array_yx.shape[1] == 2
        self.points_xy = []
        for point in points_array_yx:
            self.points_xy.append([point[1], point[0]])
        # print ("points_xy", self.points_xy)

    def __repr__(self):
        s = ""
        if self.points_xy is not None:
            s = f"ami edge pts: {self.points_xy[0]} .. {len(str(self.points_xy))} .. {self.points_xy[-1]}"
        return s

    def get_or_create_bbox(self):
        if self.bbox is None and self.points_xy is not None:
            self.bbox = BBox()
            for point in self.points_xy:
                self.bbox.add_coordinate(point)

        return self.bbox
