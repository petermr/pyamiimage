"""Integration of image processing, binarization, skeletonization and netwprk analysis"""
from skan.pre import threshold
# library
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pathlib import Path
import unittest
# local
from ..test.resources import Resources
from ..pyimage.ami_graph_all import AmiGraph
from ..pyimage.ami_skeleton import AmiSkeleton
from ..pyimage.ami_image import AmiImage
from ..pyimage.ami_util import AmiUtil


class TestAmiSkeleton:

    # def __init__(self):
    #     self.plot_plot = True
    # plot_plot = True # plots interactive plots (set false for offline)
    plot_plot = False
    cmap = "YlGnBu"

    # to reduce numbers of tests
    skip_non_essential = True
    skip_non_essential = False

    # markers
    use_ami_graph = True
    # use_ami_graph = False

    # errors to be fixed
    skip_no_create_bbox_error = False  # 1 cases
    skip_found_set_error = False  # 1 cases
    skip_not_subscriptable = False  # 8 cases
    skip_not_iterable = False  # 1 cases

    skip_will_be_refactored = True
    skipfloodfilltest = True  # test elsewhere
    obsolete = True

    interactive = False

    # init seems to disable tests
    # def __init__(self):
    #     self.arrows_skeleton = None

    def setup_method(self):
        self.arrows_skeleton = TestAmiSkeleton.create_biosynth_arrows_skeleton()
        self.arrows1_image = io.imread(Resources.BIOSYNTH1_ARROWS)
        self.arrows1_skeleton = AmiImage.invert_binarize_skeletonize(io.imread(Resources.BIOSYNTH1_ARROWS))
        self.arrows1_graph = AmiGraph.create_nx_graph_from_skeleton(self.arrows1_skeleton)
        # self.arrows1_graph = AmiSkeleton().create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)


    @classmethod

    def create_biosynth_arrows_skeleton(cls):
        skeleton_image = TestAmiSkeleton.create_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        return skeleton_image
        # @unittest.skipIf(skip_OK, "already runs")

    def test_example_basics_biosynth1_no_text(self):
        assert np.count_nonzero(self.arrows_skeleton) == 1377
        """Primarily for validating the image data which will be used elsewhere
        gray image, later binarized and thresholded
        return skeleton_image
        This will interactively plot the various images.
        (I am still learning matplotlib so take this with caution)
        to disable this set plot_plot to False
        to display the plot, set plot_plot to True

        the command
        plt.show()
        will show the latest image submitted to ax.imshow() or plt.imshow()

        """
        cmap = "Greys"
        cmap = "Greens"
        cmap = self.cmap

        file = Resources.BIOSYNTH1_ARROWS
        assert file.exists()
        image = io.imread(file)
        # this is a gray image??
        assert image.shape == (315, 1512)
        npix = image.size
        nwhite = np.sum(image == 255)
        assert nwhite == 469624
        nblack = np.sum(image == 0)
        assert nblack == 1941
        ndark = np.sum(image <= 127)
        assert ndark == 4285
        nlight = np.sum(image > 127)
        assert nlight == 471995
        print(f"\nnpix {npix}, nwhite {nwhite}, nblack {nblack}  nother {npix - nwhite - nblack}, ndark {ndark}, "
              f"nlight {nlight}")
        fig, ax = plt.subplots()
        ax.set_title("greyscale")
        fig.set_title = "FIGURE"
        # gray plot
        cmap = "Greys"
        ax.imshow(image, cmap=cmap)
        plt.title("grayscale")
        if self.plot_plot:
            plt.show()

        binary = threshold(image)
        assert binary.shape == (315, 1512)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 471788
        nblack = npix - nwhite
        assert nblack == 4492
        # print(f"npix {npix}, nwhite {nwhite} nblack {nblack} nother {npix - nwhite - nblack}")
        # print(binary)

        fig, ax = plt.subplots(1, 2)
        fig.title = "FIGURE"
        # binary plot
        cmap = "Reds"
        ax[0].imshow(binary, cmap=cmap)
        ax[0].set_title("ax0 auto-thresholded plot")
        # plt.show()

        binary = np.invert(binary)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 4492
        cmap = "YlOrRd"
        ax[1].imshow(binary, cmap=cmap)
        ax[1].set_title("ax1 binary")
        # cmap = "Greys"
        # plt.imshow(binary, cmap=cmap)
        if self.plot_plot:
            plt.show()

        return

    def test_skeletonize_biosynth1_no_text(self):
        skeleton_image = TestAmiSkeleton.create_biosynth_arrows_skeleton()
        # will be white on gray
        plt.imshow(skeleton_image, cmap="YlGnBu")
        plt.imshow(skeleton_image, cmap="Greys")
        print("\n", skeleton_image)
        if self.plot_plot:
            plt.show()

    def test_skeleton_to_graph_arrows1_WORKS(self):
        """creates nodes and edges for already clipped """
        # ami_skel = AmiSkeleton()
        #
        # skeleton_array = AmiImage.create_white_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        # io.imshow(skeleton_array)
        # Util.check_type_and_existence(skeleton_array, np.ndarray)
        # # build graph from skeleton
        # ami_skel.nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)



        AmiUtil.check_type_and_existence(self.arrows1_graph, nx.MultiGraph)
        print(f" nx {self.arrows1_graph}, {self.arrows1_graph.nodes} {self.arrows1_graph.edges}")
        AmiUtil.check_type_and_existence(self.arrows1_graph.nodes, nx.classes.reportviews.NodeView)
        assert list(self.arrows1_graph.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                 19, 20, 21, 22, 23, 24, 25, 26]
        AmiUtil.check_type_and_existence(self.arrows1_graph.edges, nx.classes.reportviews.MultiEdgeView)
        assert list(self.arrows1_graph.edges) == [(0, 2, 0), (1, 4, 0), (2, 4, 0), (2, 3, 0), (2, 7, 0), (4, 5, 0),
                                                     (4, 6, 0), (8, 19, 0), (9, 19, 0), (10, 12, 0), (11, 13, 0), (12, 13, 0),
                                                     (12, 18, 0), (13, 14,  0), (13, 15, 0), (16, 18, 0), (17, 18, 0), (18, 20, 0),
                                                     (19, 26, 0), (21, 24, 0), (22, 24, 0), (23, 24, 0), (24, 25, 0)]

        if self.plot_plot:
            AmiGraph.plot_nx_graph_NX(self.arrows1_graph)

    def test_skeleton_to_graph_components_with_nodes(self):
        # skeleton_array = AmiImage.create_white_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        # Util.check_type_and_existence(skeleton_array, np.ndarray)
        # nx_graph = AmiSkeleton().create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)
        assert nx.algorithms.components.number_connected_components(self.arrows1_graph) == 4
        connected_components = list(nx.algorithms.components.connected_components(self.arrows1_graph))
        assert connected_components == [{0, 1, 2, 3, 4, 5, 6, 7},
                                        {8, 9, 26, 19},
                                        {10, 11, 12, 13, 14, 15, 16, 17, 18, 20},
                                        {21, 22, 23, 24, 25}]
        assert connected_components[0] == {0, 1, 2, 3, 4, 5, 6, 7}
        assert connected_components[1] == {8, 9, 26, 19}

    def test_remove_pixels_in_bounding_box_arrows1(self):
        image = io.imread(Resources.BIOSYNTH1_ARROWS)
        bbox = ((82, 102), (661, 863))
        image = AmiGraph.set_bbox_pixels_to_color(bbox, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

# the only use so far of AmiGraph

    def test_skeletonize(self):
        skeleton_image = self.binarize_and_skeletonize_arrows()
        skeleton_image = skeleton_image.astype(np.uint16)
        assert skeleton_image is not None
        assert skeleton_image.shape == (315, 1512)
        print("skeleton type: ", type(skeleton_image))
        assert type(skeleton_image) is np.ndarray
        assert type(skeleton_image[0][0]) is np.uint16
        assert skeleton_image[0][0] == 0

# Utils

    @classmethod
    def binarize_and_skeletonize_arrows(cls):
        test_resources_dir = Path(Path(__file__).parent.parent, "test/resources")
        biosynth_path_image = Path(test_resources_dir, "biosynth_path_1_cropped_text_removed.png")
        grayscale = AmiImage.create_grayscale_from_file(biosynth_path_image)
        skeleton = AmiImage.create_white_skeleton_from_image(grayscale)
        return skeleton

    # obsolete?
    def set_bbox_to_color(self, bbox, dd, image):
        margined_bbox = ((bbox[0][0] - dd, bbox[0][1] + dd), (bbox[1][0] - dd, bbox[1][1] + dd))
        AmiGraph.set_bbox_pixels_to_color(margined_bbox, image, color=160)



    @classmethod
    def create_skeleton_from_file(cls, file):
        assert file.exists()
        gray_image = AmiImage.create_grayscale_from_file(file)
        skeleton_image = AmiImage.invert_binarize_skeletonize(gray_image)
        assert type(skeleton_image) is np.ndarray, f"skeleton type shoukd be np.ndarray, is {type(skeleton_image)}"
        return skeleton_image

