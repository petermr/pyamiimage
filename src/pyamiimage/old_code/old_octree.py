# !/usr/bin/env python
#-*- coding: utf8 -*-
# from https://github.com/delimitry/octree_color_quantizer CC0, thanks
from pathlib import Path

from pyimage.old_code.ColorModule import Color
from PIL import Image


class OctreeNode(object):
    """
    Octree Node class for color quantization
    """

    def __init__(self, level, parent):
        """
        Init new Octree Node
        """
        self.color = Color(0, 0, 0)
        self.pixel_count = 0
        self.palette_index = 0
        self.children = [None for _ in range(8)]
        # add node to current level
        if level < OctreeQuantizer.MAX_DEPTH - 1:
            parent.add_level_node(level, self)

    def is_leaf(self):
        """
        Check that node is leaf
        """
        return self.pixel_count > 0

    def get_leaf_nodes(self):
        """
        Get all leaf nodes
        """
        leaf_nodes = []
        for i in range(8):
            node = self.children[i]
            if node:
                if node.is_leaf():
                    leaf_nodes.append(node)
                else:
                    leaf_nodes.extend(node.get_leaf_nodes())
        return leaf_nodes

    def get_nodes_pixel_count(self):
        """
        Get a sum of pixel count for node and its children
        """
        sum_count = self.pixel_count
        for i in range(8):
            node = self.children[i]
            if node:
                sum_count += node.pixel_count
        return sum_count

    def add_color(self, color, level, parent):
        """
        Add `color` to the tree
        """
        if level >= OctreeQuantizer.MAX_DEPTH:
            self.color.red += color.red
            self.color.green += color.green
            self.color.blue += color.blue
            self.pixel_count += 1
            return
        index = self.get_color_index_for_level(color, level)
        if not self.children[index]:
            self.children[index] = OctreeNode(level, parent)
        self.children[index].add_color(color, level + 1, parent)

    def get_palette_index(self, color, level):
        """
        Get palette index for `color`
        Uses `level` to go one level deeper if the node is not a leaf
        """
        if self.is_leaf():
            return self.palette_index
        index = self.get_color_index_for_level(color, level)
        if self.children[index]:
            return self.children[index].get_palette_index(color, level + 1)
        else:
            # get palette index for a first found child node
            for i in range(8):
                if self.children[i]:
                    return self.children[i].get_palette_index(color, level + 1)

    def remove_leaves(self):
        """
        Add all children pixels count and color channels to parent node
        Return the number of removed leaves
        """
        result = 0
        for i in range(8):
            node = self.children[i]
            if node:
                self.color.red += node.color.red
                self.color.green += node.color.green
                self.color.blue += node.color.blue
                self.pixel_count += node.pixel_count
                result += 1
        return result - 1

    @classmethod
    def get_color_index_for_level(cls, color, level):
        """
        Get index of `color` for next `level`
        """
        index = 0
        mask = 0x80 >> level
        if color.red & mask:
            index |= 4
        if color.green & mask:
            index |= 2
        if color.blue & mask:
            index |= 1
        return index

    def get_color(self):
        """
        Get average color
        """
        return Color(
            self.color.red // self.pixel_count,
            self.color.green // self.pixel_count,
            self.color.blue // self.pixel_count)


class OctreeQuantizer(object):
    """
    Octree Quantizer class for image color quantization
    Use MAX_DEPTH to limit a number of levels
    """

    MAX_DEPTH = 8

    def __init__(self):
        """
        Init Octree Quantizer
        """
        self.levels = {i: [] for i in range(OctreeQuantizer.MAX_DEPTH)}
        self.root = OctreeNode(0, self)

    def get_leaves(self):
        """
        Get all leaves
        """
        return [node for node in self.root.get_leaf_nodes()]

    def add_level_node(self, level, node):
        """
        Add `node` to the nodes at `level`
        """
        self.levels[level].append(node)

    def add_color(self, color):
        """
        Add `color` to the Octree
        """
        # passes self value as `parent` to save nodes to levels dict
        self.root.add_color(color, 0, self)

    def make_palette(self, color_count):
        """
        Make color palette with `color_count` colors maximum
        """
        palette = []
        palette_index = 0
        leaf_count = len(self.get_leaves())
        # reduce nodes
        # up to 8 leaves can be reduced here and the palette will have
        # only 248 colors (in worst case) instead of expected 256 colors
        for level in range(OctreeQuantizer.MAX_DEPTH - 1, -1, -1):
            if self.levels[level]:
                for node in self.levels[level]:
                    leaf_count -= node.remove_leaves()
                    if leaf_count <= color_count:
                        break
                if leaf_count <= color_count:
                    break
                self.levels[level] = []
        # build palette
        for node in self.get_leaves():
            if palette_index >= color_count:
                break
            if node.is_leaf():
                palette.append(node.get_color())
            node.palette_index = palette_index
            palette_index += 1
        return palette

    def get_palette_index(self, color):
        """
        Get palette index for `color`
        """
        return self.root.get_palette_index(color, 0)


def quantize(image, size=4):
    import time
    from ColorModule import Color
    # from Octree import OctreeQuantizer

    # size = 16 => 256 colors for 8 bits per pixel output image

    # print("pixels shape", image.shape)
    pixels = image.load()
    width, height = image.size
    print(width, height)

    octree = OctreeQuantizer()

    # add colors to the octree
    time0 = time.perf_counter()
    for j in range(height):
        for i in range(width):
            octree.add_color(Color(*pixels[i, j]))
    print("octree time", time.perf_counter()-time0)

    palette, palette_image = create_palette_image(size, octree, width, height)
    out_image = create_output_image(width, height, octree, palette, pixels)
    return out_image, palette, palette_image


def create_palette_image(size, octree, width, height):
    from PIL import Image
    import time
    time0 = time.perf_counter()
    palette = octree.make_palette(size * size)
    palette_image = Image.new('RGB', (size, size))
    palette_pixels = palette_image.load()
    for i, color in enumerate(palette):
        rgb = (color.red, color.green, color.blue)
        palette_pixels[i % size, i // size] = rgb
        print("rgb", rgb)
    print("palette time", time.perf_counter()-time0)
    return palette, palette_image


def create_output_image(width, height, octree, palette, pixels):
    from PIL import Image
    import time
    time0 = time.perf_counter()
    out_image = Image.new('RGB', (width, height))
    out_pixels = out_image.load()
    for j in range(height):
        for i in range(width):
            index = octree.get_palette_index(Color(*pixels[i, j]))
            color = palette[index]
            out_pixels[i, j] = (color.red, color.green, color.blue)

    print("output time", time.perf_counter()-time0)
    return out_image


def main():
    # image_name = "red_black_cv.png"
    image_name = "purple_ocimum_basilicum.png"
    path = Path(Path(__file__).parent.parent, "assets", image_name)
    assert path.exists()
    # img = imageio.imread(path)
    img = Image.open(path)

    # ImageLib.image_show(img)
    # print(img)

    quantize(img, size=4)


if __name__ == "__main__":
    main()
