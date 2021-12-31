from skimage import data
import os
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk  # noqa
import numpy as np

from pyimage.ami_image import AmiImage

class Exploration:

    def sharpen_explore(self, axis=False):
        gray = Exploration.create_gray_network_snippet("snippet_rgba.png")

        result_1_1 = unsharp_mask(gray, radius=1, amount=1)
        result_5_2 = unsharp_mask(gray, radius=5, amount=2)
        result_20_1 = unsharp_mask(gray, radius=20, amount=1)

        plots = [
            {"image": gray, "title": 'Original image'},
            {"image": result_1_1, "title": 'Enhanced image, radius=1, amount=1.0'},
            {"image": result_5_2, "title": 'Enhanced image, radius=5, amount=2.0'},
            {"image": result_20_1, "title": 'Enhanced image, radius=20, amount=1.0'},
        ]
        ax, fig = self.create_subplots(plots, nrows=2, ncols=2, figsize=(10, 10))

        Exploration.axis_layout(ax, axis, fig)
        plt.show()

    def explore_erode_dilate(self):
        from skimage.util import img_as_ubyte
        # orig_phantom = img_as_ubyte(data.shepp_logan_phantom())
        # Exploration.make_numpy_assert(orig_phantom, shape=(400, 400), max=255, dtype=np.uint8)
        # fig, ax = plt.subplots()
        # ax.imshow(orig_phantom, cmap=plt.cm.gray)
        # plt.show()
        #
        # footprint = disk(6)
        # Exploration.make_numpy_assert(footprint, shape=(13, 13), max=1, dtype=np.uint8)
        #
        # eroded = erosion(orig_phantom, footprint)
        # Exploration.make_numpy_assert(eroded, shape=(400, 400), max=255)
        # Exploration.plot_comparison(orig_phantom, eroded, 'erosion')
        # plt.show()
        #

        white = Exploration.create_white_network_snippet("snippet_rgba.png")

        Exploration.make_numpy_assert(white, shape=(341, 796), max=1, dtype=np.bool)

        footprint = disk(1)
        eroded = erosion(white, footprint)
        Exploration.plot_comparison(white, eroded, 'erosion')

        plt.show()

        erode_1 = erosion(white, disk(1))
        erode_2 = erosion(white, disk(2))
        dilate_1 = dilation(white, disk(1))
        dilate_2 = dilation(white, disk(2))
        dilate_erode_1 = dilation(erosion(white, disk(1)), disk(1))

        plots = [
            {"image": white, "title": 'Original image'},
            {"image": erode_1, "title": 'erode disk=1'},
            {"image": erode_2, "title": 'erode disk=2'},
            {"image": dilate_1, "title": 'dilate disk=1'},
            {"image": dilate_2, "title": 'dilate disk=2'},
            {"image": dilate_erode_1, "title": 'dilate_erode disk=1'},
        ]
        ax, fig = self.create_subplots(plots, nrows=3, ncols=2, figsize=(10, 10))

        # Exploration.axis_layout(ax, axis, fig)
        plt.show()

# ================= resources =================
    @classmethod
    def create_gray_network_snippet(cls, png):
        path = Path(Path(__file__).parent.parent, "test", "resources", png)
        assert path.exists(), f"path {path} exists"
        gray = AmiImage.create_grayscale_from_file(path)
        return gray

    @classmethod
    def create_white_network_snippet(cls, png):
        path = Path(Path(__file__).parent.parent, "test", "resources", png)
        assert path.exists(), f"path {path} exists"
        white = AmiImage.create_white_binary_from_file(path)

        Exploration.make_numpy_assert(white, shape=(341, 796), max=1, dtype=np.bool)

        return white

    @classmethod
    def plot_comparison(cls, original, modified, modified_title):
        """
        Plots old/new images side-by-side
        :param original:
        :param modified:
        :param modified_title: title of RH image
        :return:
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(modified, cmap=plt.cm.gray)
        ax2.set_title(modified_title)
        ax2.axis('off')

    @classmethod
    def create_subplots(cls, plots, nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10)):
        """
        Convenience method to create subplots
        :param plots: array of ndarrays to plot
        :param nrows:
        :param ncols:
        :param sharex:
        :param sharey:
        :param figsize: tuple for display (width, height in "cm" I think)
        :return: ax, fig
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 sharex=sharex, sharey=sharey, figsize=figsize)
        ax = axes.ravel()
        for i, plot in enumerate(plots):
            ax[i].imshow(plots[i]["image"], cmap=plt.cm.gray)
            ax[i].set_title(plots[i]["title"])
        return ax, fig

    @classmethod
    def axis_layout(cls, ax, axis, fig):
        """
        Not quite sure what it switches on/off
        :param ax:
        :param axis:
        :param fig:
        :return:
        """
        if axis:
            for a in ax:
                a.axis('off')
        fig.tight_layout()

# =========== palettes ============

"""
From https://stackoverflow.com/questions/45523205/get-rgb-colors-from-color-palette-image-and-apply-to-binary-image

You can use a combination of a reshape and np.unique to extract the unique RGB values from your color palette image:
"""
# Load the color palette
from skimage import io
raise NotImplemented("image explore, needs biosynth3??")
palette = io.imread(os.image.join(os.getcwd(), 'color_palette.png'))

# Use `np.unique` following a reshape to get the RGB values
palette = palette.reshape(palette.shape[0]*palette.shape[1], palette.shape[2])
palette_colors = np.unique(palette, axis=0)
"""
(Note that the axis argument for np.unique was added in numpy version 1.13.0, so you may need to upgrade numpy for this to work.)

Once you have palette_colors, you can pretty much use the code you already have to save the image, except you now add the different RGB values instead of copies of ~img to your img_rgba array.
"""
img = None  # TODO
for p in range(palette_colors.shape[0]):

    # Create an MxNx4 array (RGBA)
    img_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

    # Fill R, G and B with appropriate colors
    for c in range(3):
        img_rgba[:,:,c] = img.astype(np.uint8) * palette_colors[p,c]

    # For alpha just use the image again (makes background transparent)
    img_rgba[:,:,3] = img.astype(np.uint8) * 255

    # Save image
    io.imsave('img_col'+str(p)+'.png', img_rgba)

# (Note that you need to use np.uint8 as datatype for your image, since binary images obviously cannot represent different colors.)
"""

if __name__ == '__main__':
    Exploration().sharpen_explore(axis=True)
    Exploration().explore_erode_dilate()
"""