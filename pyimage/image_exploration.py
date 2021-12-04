from skimage import data
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk  # noqa

from pyimage.ami_image import AmiImage
class Exploration:

    def sharpen_explore(self, axis=False):
        gray = Exploration.create_gray_network_snippet("snippet_rgba.png")

        result_1_1 = unsharp_mask(gray, radius=1, amount=1)
        result_5_2 = unsharp_mask(gray, radius=5, amount=2)
        result_20_1 = unsharp_mask(gray, radius=20, amount=1)

        fig, axes = plt.subplots(nrows=2, ncols=2,
                                 sharex=True, sharey=True, figsize=(10, 10))
        ax = axes.ravel()

        ax[0].imshow(gray, cmap=plt.cm.gray)
        ax[0].set_title('Original image')
        ax[1].imshow(result_1_1, cmap=plt.cm.gray)
        ax[1].set_title('Enhanced image, radius=1, amount=1.0')
        ax[2].imshow(result_5_2, cmap=plt.cm.gray)
        ax[2].set_title('Enhanced image, radius=5, amount=2.0')
        ax[3].imshow(result_20_1, cmap=plt.cm.gray)
        ax[3].set_title('Enhanced image, radius=20, amount=1.0')

        Exploration.axis_layout(ax, axis, fig)
        plt.show()

    def axis_layout(self, ax, axis, fig):
        if axis:
            for a in ax:
                a.axis('off')
        fig.tight_layout()

    def explore_erode_dilate(self):
        white = Exploration.create_white_network_snippet("snippet_rgba.png")

        footprint = disk(2)
        eroded = erosion(white, footprint)
        Exploration.plot_comparison(white, eroded, 'erosion')
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
        return white

    @classmethod
    def plot_comparison(original, filtered, filter_name):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                       sharey=True)
        ax1.imshow(original, cmap=plt.cm.gray)
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(filtered, cmap=plt.cm.gray)
        ax2.set_title(filter_name)
        ax2.axis('off')


if __name__ == '__main__':
    # Exploration().sharpen_explore(axis=True)
    Exploration().explore_erode_dilate()