from skimage import data
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


    @classmethod
    def make_numpy_assert(cls, numpy_array, shape=None, max=None, dtype=None):
        """
        Asserts properties of numpy_array
        :param numpy_array:
        :param shape:
        :param max: max value (e.g. 255, or 1.0 for images)
        :param dtype:
        :return:
        """
        assert numpy_array is not None, f"numpy array should not be None"
        if type(numpy_array) is not np.ndarray:
            print(f"object should be numpy.darray, found {type(numpy_array)} \n {numpy_array}")
        if shape:
            assert numpy_array.shape == shape, f"shape should be {numpy_array.shape}"
        if max:
            assert np.max(numpy_array) == max, f"max should be {np.max(numpy_array)}"
        if dtype:
            assert numpy_array.dtype == dtype, f"dtype should be {numpy_array.dtype}"

if __name__ == '__main__':
    Exploration().sharpen_explore(axis=True)
    Exploration().explore_erode_dilate()