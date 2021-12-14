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
from ..pyimage.util import Util


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
        # TODO
        # ami_skel = AmiSkeleton()
        #
        # skeleton_array = AmiImage.create_white_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        # io.imshow(skeleton_array)
        # Util.check_type_and_existence(skeleton_array, np.ndarray)
        # # build graph from skeleton
        # ami_skel.nx_graph = AmiGraph.create_nx_graph_from_skeleton(skeleton_array)



        Util.check_type_and_existence(self.arrows1_graph, nx.classes.graph.Graph)
        print(f" nx {self.arrows1_graph}, {self.arrows1_graph.nodes} {self.arrows1_graph.edges}")
        Util.check_type_and_existence(self.arrows1_graph.nodes, nx.classes.reportviews.NodeView)
        assert list(self.arrows1_graph.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                 19, 20, 21, 22, 23, 24, 25, 26]
        Util.check_type_and_existence(self.arrows1_graph.edges, nx.classes.reportviews.EdgeView)
        assert list(self.arrows1_graph.edges) == [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6), (8, 19),
                                                 (9, 19), (10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15),
                                                 (16, 18), (17, 18), (18, 20), (19, 26), (21, 24), (22, 24), (23, 24),
                                                 (24, 25)]
        if self.plot_plot:
            AmiGraph.plot_nx_graph_NX(self.arrows1_graph)

    @unittest.skipIf(skip_non_essential, "graphs of texts not very useful")
    def test_skeleton_to_graph_text(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot_TEST(Resources.BIOSYNTH1_TEXT, self.plot_plot)

    @unittest.skipIf(skip_non_essential, "graphs of everything not very useful")
    def test_skeleton_to_graph_path1(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot_TEST(Resources.BIOSYNTH1, plot_plot=self.plot_plot)

    @unittest.skipIf(skip_non_essential, "graphs of everything not very useful")
    @unittest.skip("seg faults")
    def test_skeleton_to_graph_path2(self):
        assert Resources.BIOSYNTH2.exists(), f"file should exist {Resources.BIOSYNTH2}"
        try:
            AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot_TEST(Resources.BIOSYNTH2, plot_plot=self.plot_plot)
        except Exception:
            raise Exception("seg fault")

    @unittest.skipIf(skip_non_essential, "graphs of everything not very useful")
    def test_skeleton_to_graph_path3(self):
        """plots all islands in page, including characters"""
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot_TEST(Resources.BIOSYNTH3, plot_plot=self.plot_plot)

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

    @unittest.skipIf(skip_found_set_error or skip_will_be_refactored,
                     "expected <class 'pyimage.graph_lib.AmiIsland'> found <class 'set'>")
    def test_create_bounding_box_from_node_list(self):
        """computes bbox for single 7-node island"""
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)
        node_ids = {0, 1, 2, 3, 4, 5, 6, 7}

        bbox = ami_skeleton.extract_bbox_for_nodes_ISLAND(node_ids)
        assert bbox == ((661.0, 863.0), (82.0, 102.0))

    @unittest.skipIf(skip_no_create_bbox_error or skip_will_be_refactored,
                     "'TestAmiSkeleton' object has no attribute 'create_bbox_for_island'")
    def test_create_bounding_boxes_from_node_list(self):
        """reads plot with 4 islands, extracts islands and calculates their bboxes"""
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)
        bboxes = self.create_bboxes_for_islands(ami_skeleton)

        assert len(bboxes) == 4
        assert bboxes == [((661.0, 863.0), (82.0, 102.0)),
                          ((391.0, 953.0), (117.0, 313.0)),
                          ((991.0, 1064.0), (148.0, 236.0)),
                          ((992.0, 1009.0), (252.0, 294.0))]

    @unittest.skipIf(skip_not_subscriptable or skip_will_be_refactored, "'AmiIsland' object is not subscriptable")
    def test_create_bounding_boxes_from_node_list_with_size_filter_biosynth3(self):
        """filters out small components by bbox_gauge"""

        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH3)
        min_box = (50, 50)
        # ami_skeleton.set_minimum_dimension(min_box)
        assert ami_skeleton.nx_graph is not None
        ami_skeleton.islands = ami_skeleton.get_ami_islands_from_nx_graph_GRAPH()
        bboxes = ami_skeleton.islands
        assert len(bboxes) == 417

        bboxes_small = [bbox for bbox in bboxes if AmiSkeleton.fits_within_BBOX(bbox, min_box)]
        assert len(bboxes_small) == 412
        bboxes_large = [bbox for bbox in bboxes if not AmiSkeleton.fits_within_BBOX(bbox, min_box)]
        assert len(bboxes_large) == 5

        assert bboxes_large == [
             ((194, 217), (188, 242)),
             ((194, 217), (298, 354)),
             ((87, 219), (385, 786)),
             ((193, 216), (410, 465)),
             ((197, 219), (849, 904))]

    def test_remove_pixels_in_bounding_box_arrows1(self):
        image = io.imread(Resources.BIOSYNTH1_ARROWS)
        bbox = ((82, 102), (661, 863))
        image = AmiGraph.set_bbox_pixels_to_color(bbox, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    @unittest.skipIf(obsolete or skip_not_subscriptable, "'NoneType' object is not subscriptable")
    def test_remove_pixels_in_bounding_boxes_from_islands_arrows1(self):
        image = io.imread(Resources.BIOSYNTH1_ARROWS)
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)

        assert ami_skeleton.nx_graph is not None
        ami_skeleton.islands = ami_skeleton.get_ami_islands_from_nx_graph_GRAPH()
        islands = ami_skeleton.islands
        print("island", islands[0])
        margin = 2  # to overcome some of the antialiasing
        for island in islands:
            raw_bbox = island.get_raw_box()  # not used
            sub_image = ((raw_bbox[0][0]-margin, raw_bbox[0][1]+margin), (raw_bbox[1][0]-margin, raw_bbox[1][1]+margin))
            AmiGraph.set_bbox_pixels_to_color(sub_image, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    @unittest.skipIf(skip_not_subscriptable, "'NoneType' object is not subscriptable")
    def test_remove_pixels_in_bounding_boxes_from_islands_arrows1_NEW(self):
        # image = io.imread(Resources.BIOSYNTH1_ARROWS)
        # ami_skeleton = AmiSkeleton()
        # nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)
        ami_graph = AmiGraph(self.arrows1_graph)
        # ami_graph = AmiGraph(nx_graph)

        islands = ami_graph.get_ami_islands_from_nx_graph()
        # print("island", islands[0])
        margin = 2  # to overcome some of the antialiasing
        for island in islands:
            bbox = island.get_or_create_bbox()
            bbox.expand_by_margin((20, 30))
            print(f"bbox {bbox}")
            image = AmiGraph.set_bbox_pixels_to_color(bbox.xy_ranges, self.arrows1_image, colorx=255)
            if self.interactive:
                plt.imshow(image)
                plt.show()

        if self.interactive:
            fig, ax = plt.subplots()
            ax.imshow(self.arrows1_image, cmap='gray')
            plt.show()
        return

    @unittest.skipIf(obsolete or skip_not_subscriptable, "'AmiIsland' object is not subscriptabl")
    def test_remove_all_pixels_in_bounding_boxes_from_islands(self):
        """
        Don't think this is working yet
        :return:
        """
        image = io.imread(Resources.BIOSYNTH1)
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1)
        assert ami_skeleton.nx_graph is not None
        ami_skeleton.islands = ami_skeleton.get_ami_islands_from_nx_graph_GRAPH()
        bboxes = ami_skeleton.islands
        margin = 2   # to overcome some of the antialiasing
        for bbox in bboxes:
            self.set_bbox_to_color(bbox, margin, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    @unittest.skipIf(obsolete or skip_not_subscriptable, "'AmiIsland' object is not subscriptable")
    def test_remove_pixels_in_arrow_bounding_boxes_from_islands_text1(self):
        ami_skeleton = AmiSkeleton()
        # arrows_image = io.imread(Resources.BIOSYNTH1_ARROWS)
        arrows_image = AmiImage.create_grayscale_from_file(Resources.BIOSYNTH1_ARROWS)

        cropped_image = AmiImage.create_grayscale_from_file(Resources.BIOSYNTH1_CROPPED)
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw_NX_GRAPH(Resources.BIOSYNTH1_ARROWS)
        assert nx_graph is not None
        ami_skeleton.islands = ami_skeleton.get_ami_islands_from_nx_graph()
        bboxes_arrows = ami_skeleton.islands
        dd = 2  # to overcome some of the antialiasing
        for bbox in bboxes_arrows:
            bbox = ((bbox[0][0]-dd, bbox[0][1]+dd), (bbox[1][0]-dd, bbox[1][1]+dd))
            AmiGraph.set_bbox_pixels_to_color(bbox, cropped_image, colorx=127)
        fig, ax = plt.subplots()
        ax.imshow(cropped_image, cmap='gray')
        if ami_skeleton.interactive:
            plt.show()
        return

    @unittest.skipIf(skipfloodfilltest or skip_not_iterable, "'AmiIsland' object is not iterable")
    def test_flood_fill_first_component(self):
        ami_skeleton = AmiSkeleton()
        component_index = 0  # as example
        ami_skeleton.read_image_plot_component(component_index, Resources.BIOSYNTH1_ARROWS)
        return

    @unittest.skipIf(skipfloodfilltest or skip_not_subscriptable, "'AmiIsland' object is not subscriptable")
    def test_flood_fill_many_components(self):
        ami_skeleton = AmiSkeleton()
        path = Resources.BIOSYNTH1_ARROWS
        ami_skeleton.create_and_plot_all_components_TEST(path)
        return

    @unittest.skipIf(skipfloodfilltest or skip_not_subscriptable, "'AmiIsland' object is not subscriptable")
    def test_flood_fill_many_components_select(self):
        ami_skeleton = AmiSkeleton()
        path = Resources.BIOSYNTH1_CROPPED
        ami_skeleton.create_and_plot_all_components_TEST(path, min_size=[30, 30])
        return

    @unittest.skipIf(skipfloodfilltest or skip_not_subscriptable, "'AmiIsland' object is not subscriptable")
    def test_flood_fill_many_components_biosynth3(self):
        """EXAMPLE finds the 5 arrows"""
        ami_skeleton = AmiSkeleton(title="biosynth3")
        ami_skeleton.interactive = True
        ami_skeleton.interactive = False
        path = Resources.BIOSYNTH3
        ami_skeleton.create_and_plot_all_components_TEST(path, min_size=[30, 30])
        return

    @unittest.skipIf(skipfloodfilltest or skip_not_subscriptable, "'AmiIsland' object is not subscriptable")
    def test_flood_fill_many_components_1(self):
        AmiSkeleton().create_and_plot_all_components_TEST(Resources.BIOSYNTH1_ARROWS)
        return

    @unittest.skip("not part of skeleton")
    def test_hocr_to_svg_biosynth1(self):
        ami_skeleton = AmiSkeleton()

        biosynth_html = str(Resources.BIOSYNTH1_HOCR)
        ami_skeleton.create_svg_from_hocr(biosynth_html, "biosynth_1.svg")

    @unittest.skip("not part of skeleton")
    def test_hocr_to_svg_biosynth3(self):
        """creates textboxes for HOCR put and writes to temp/textbox"""
        ami_skeleton = AmiSkeleton()

        foo = ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH3_HOCR), "biosynth_3.svg")
        print(foo)

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
        image = AmiImage.create_grayscale_from_file(biosynth_path_image)
        skeleton = AmiImage.create_white_skeleton_from_image(image)
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

