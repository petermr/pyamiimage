import numpy as np
from skimage import io
from skimage.color.colorconv import rgb2gray
import skimage
from skimage import morphology
from skimage import filters
from pathlib import Path
import matplotlib.pyplot as plt

"""
The ImageProcessor class is current in development by PMR and Anuv for preprocessing images
as a part of the project: "Extraction of biosynthetic pathway from images"

Some part of the code has been copied from ImageLib written by PMR for openDiagram
We decided against continuing development on openDiagram library because the size
of the repository exceeded 2 gigabytes

The ImageLib module has been included in this repository for testing and reference 
"""

# setting a sample image for default path
DEFAULT_PATH = "assets/purple_ocimum_basilicum.png"
TEST_RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
# BIOSYNTH_PATH_IMAGE = Path(TEST_RESOURCES_DIR, "biosynth_path_1.png")
BIOSYNTH_PATH_IMAGE = Path(TEST_RESOURCES_DIR, "biosynth_path_1_cropped_text_removed.png")


class ImageProcessor:

    def __init__(self) -> None:
        self.image = None
        self.inverted = None

    def load_image(self, path):
        """
        loads image with io.imread
        resets self.image
        input: path
        returns: None if path is None
        """
        self.image = None
        if path is not None:
            self.image = io.imread(path)
        return self.image
        
    def to_gray(self):
        """convert existing self.image to grayscale
        uses rgb2gray from skimage.color.colorconv
        """
        image_gray = None
        if self.image is not None:
            image_gray = rgb2gray(self.image)
        return image_gray

    def invert(self, image):
        """Inverts the brightness values of the image"""
        self.inverted = skimage.util.invert(image)
        return self.inverted
    
    def skeletonize(self, image):
        """Returns a skeleton of the image"""
        mask = morphology.skeletonize(image)
        skeleton = np.zeros(self.image.shape)
        skeleton[mask] = 1
        # print("Skeleton Image: ", self.skeleton)
        return skeleton

    @classmethod
    def threshold(cls, image):
        """"Returns a binary image using a threshold value"""
        threshold = filters.threshold_otsu(image)
        # self.binary_image = np.zeros(self.image.shape)
        # print(self.image.shape)
        # idx = self.image > threshold
        # print("Threshold Mask: ", idx)
        # self.binary_image[idx] = 1
        # print("Binary Image: ", self.binary_image)
        binary_image = np.where(image >= threshold, 1, 0)
        return binary_image

    def show_image(self, image):
        """
        Shows self.image in a separate window
        :param image:
        """
        if self.image is None:
            self.load_image()
        io.imshow(image)
        io.show()
        return True

    def example1(self):
        print(BIOSYNTH_PATH_IMAGE)
        self.load_image(BIOSYNTH_PATH_IMAGE)
        # print(self.image)
        # print(self.image.shape)
        # from skimage.exposure import histogram
        # hist, hist_centers = histogram(self.threshold(self.image))
        # print(hist)
        # print(hist_centers)

        self.show_image(self.image)
        inverted_image = self.invert(self.image)
        self.show_image(inverted_image)
        binary_image = self.threshold(inverted_image)
        self.show_image(binary_image)
        skeleton = self.skeletonize(binary_image)
        self.show_image(skeleton)

    @classmethod
    def get_maxx_maxy_non_pythonic(cls, node_dict, nodes):
        maxx = -999999
        maxy = -999999
        for node, i in enumerate(nodes):
            # certainly a more pythonic way exists
            x = node_dict[i][0]
            if x > maxx:
                maxx = x
            y = node_dict[i][1]
            if y > maxy:
                maxy = y
        return maxx, maxy

    @classmethod
    def plot_line(cls, node_dict, node0, node1, ymax):
        # print("node", node0, type(node0))
        xy0 = node_dict[node0]
        xy1 = node_dict[node1]
        # print("xy0 xy1", xy0, xy1)
        # x and y are swapped
        plt.plot([xy0[1], xy1[1]], [ymax - xy0[0], ymax - xy1[0]], marker="")
        # print(type(node0))

    def invert_threshold_skeletonize(self, show=False):
        """Inverts Thresholds and Skeletonize a single channel grayscale image
        :show: display images for invert, threshold and skeletonize 
        :return: skeletonized image
        """
        if show:
            self.show_image(self.image)

        inverted_image = self.invert(self.image)
        if show:
            self.show_image(inverted_image)
        
        binary_image = self.threshold(inverted_image)
        binary_image = binary_image.astype(np.uint16)
        if show:
            self.show_image(binary_image)
        
        skeleton = self.skeletonize(binary_image)
        
        if show:
            self.show_image(skeleton)

        return skeleton


def main():
    image_processor = ImageProcessor()
    image_processor.example1()
    #image_processor.example_skeletonize_extract_subgraphs()


if __name__ == '__main__':
    main()
