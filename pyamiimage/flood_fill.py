"""bespoke floodfill algorithm as I couldn't find one"""

from collections import deque
import matplotlib.pyplot as plt
import numpy as np


class FloodFill:
    """creates a list of flood_filling pixels given a seed"""

    def __init__(self):
        self.start_pixel = None
        self.binary = None
        self.filling_pixels = None

    def flood_fill(self, binary_image, start_pixel):
        """

        :param binary_image: Not altered
        :param start_pixel:
        :return: (filled image, set of filling pixels)
        """
        # self.binary = self.binary.astype(int)

        self.start_pixel = start_pixel
        self.binary = binary_image
        self.filling_pixels = self.get_filling_pixels()
        return self.filling_pixels

    def get_filling_pixels(self):
        # new_image = np.copy(self.binary)
        xy = self.start_pixel
        xy_deque = deque()
        xy_deque.append(xy)
        filling_pixels = set()
        while xy_deque:
            xy = xy_deque.popleft()
            self.binary[xy[0], xy[1]] = 0  # unset pixel
            neighbours_list = self.get_neighbours(xy)
            for neighbour in neighbours_list:
                neighbour_xy = (neighbour[0], neighbour[1])  # is this necessary??
                if neighbour_xy not in filling_pixels:
                    filling_pixels.add(neighbour_xy)
                    xy_deque.append(neighbour_xy)
                else:
                    pass
        return filling_pixels

    def get_neighbours(self, xy):
        # i = xy[0]
        # j = xy[1]
        w = 3
        h = 3
        neighbours = []
        # I am sure there's a more pythonic way
        for i in range(w):
            ii = xy[0] + i - 1
            if ii < 0 or ii >= self.binary.shape[0]:
                continue
            for j in range(h):
                jj = xy[1] + j - 1
                if jj >= 0 or jj < self.binary.shape[1]:
                    if self.binary[ii][jj] == 1:
                        neighbours.append((ii, jj))
        return neighbours

    def plot_used_pixels(self):
        used_image = self.create_image_of_filled_pixels()
        fig, ax = plt.subplots()
        ax.imshow(used_image)
        plt.show()

    def create_image_of_filled_pixels(self):
        used_image = np.zeros(self.binary.shape, dtype=bool)
        for pixel in self.filling_pixels:
            used_image[pixel[0], pixel[1]] = 1
        return used_image

    def get_raw_box(self):
        """
        gets raw bounding box dimensions as an array of arrays.
        will make this into BoundingBox soon

        :return:
        """
