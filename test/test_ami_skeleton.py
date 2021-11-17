"""Integration of image processing, binarization, skeletonization and netwprk analysis"""
from skan.pre import threshold

from test.resources import Resources
from skimage import filters, color, io, data, draw
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sknw
from pyimage.graph_lib import AmiSkeleton, AmiIsland, AmiGraph, FloodFill


class TestAmiSkeleton:

    def test_basics_biosynth1_no_text(self):
        """Primarily for validating the image data which will be used elsewhere
        Uncomment for debug-like printing"""

        file = Resources.BIOSYNTH1_ARROWS
        assert file.exists()
        image = io.imread(file)
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
        # print(image)
        # images are not shown in tests, I think
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')

        binary = threshold(image)
        assert binary.shape == (315, 1512)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 471788
        nblack = npix - nwhite
        # print(f"npix {npix}, nwhite {nwhite} nblack {nblack} nother {npix - nwhite - nblack}")
        # print(binary)

        fig, ax = plt.subplots()
        ax.imshow(binary, cmap="gray")

        binary = np.invert(binary)
        nwhite = np.count_nonzero(binary)
        assert nwhite == 4492
        ax.imshow(binary, cmap="gray")
        return

    def test_skeletonize_biosynth1_no_text(self):
        file = Resources.BIOSYNTH1_ARROWS
        assert file.exists()
        skeleton = AmiSkeleton().create_white_skeleton_from_file(file)
        assert np.count_nonzero(skeleton) == 1378
        # will be white on gray
        plt.imshow(skeleton, cmap="gray")
        print("\n", skeleton)

    def test_skeleton_to_graph_arrows1(self):
        """creates nodes and edges for already clipped """
        ami_skel = AmiSkeleton()
        skeleton = ami_skel.create_white_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        # build graph from skeleton
        ami_skel.nx_graph = sknw.build_sknw(skeleton)
        ami_skel.plot_nx_graph(title="graph_arrows")

    def test_skeleton_to_graph_text(self):
        ami_skel = AmiSkeleton()
        ami_skel.binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH1_TEXT)

    def test_skeleton_to_graph_path1(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH1)

    def test_skeleton_to_graph_path2(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH2)

    def test_skeleton_to_graph_path3(self):
        """plots all islands in page, including characters"""
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH3)

    def test_skeleton_to_graph_components_with_nodes(self):
        nx_graph = AmiSkeleton().create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        assert nx.algorithms.components.number_connected_components(nx_graph) == 4
        connected_components = list(nx.algorithms.components.connected_components(nx_graph))
        assert connected_components == [{0, 1, 2, 3, 4, 5, 6, 7},
                                        {8, 9, 26, 19},
                                        {10, 11, 12, 13, 14, 15, 16, 17, 18, 20},
                                        {21, 22, 23, 24, 25}]
        assert connected_components[0] == {0,1,2,3,4,5,6,7}
        assert connected_components[1] == {8,9,26,19}

    def test_create_bounding_box_from_node_list(self):
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        node_ids = {0, 1, 2, 3, 4, 5, 6, 7}

        bbox = ami_skeleton.extract_bbox_for_nodes(node_ids)
        assert bbox == ( (661.0, 863.0), (82.0, 102.0))

    def test_create_bounding_boxes_from_node_list(self):
        """reads plot with 4 islands, extracts islands and calculates their bboxes"""
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        bboxes = ami_skeleton.create_bboxes_for_connected_components()
        assert len(bboxes) == 4
        assert bboxes == [((661.0, 863.0), (82.0, 102.0)),
                         ((391.0, 953.0), (117.0, 313.0)),
                         ((991.0, 1064.0), (148.0, 236.0)),
                         ((992.0, 1009.0), (252.0, 294.0))]

    def test_create_bounding_boxes_from_node_list_with_size_filter_biosynth3(self):
        """filters out small components by bbox_gauge"""

        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH3)
        min_box = (50, 50)
        # ami_skeleton.set_minimum_dimension(min_box)
        bboxes = ami_skeleton.create_bboxes_for_connected_components()
        assert len(bboxes) == 417

        bboxes_small = [bbox for bbox in bboxes if AmiSkeleton.fits_within(bbox, min_box)]
        assert len(bboxes_small) == 412
        bboxes_large = [bbox for bbox in bboxes if not AmiSkeleton.fits_within(bbox, min_box)]
        assert len(bboxes_large) == 5

        assert bboxes_large == [
             ((194, 217), (188, 242)),
             ((194, 217), (298, 354)),
             ((87, 219), (385, 786)),
             ((193, 216), (410, 465)),
             ((197, 219), (849, 904))]

    def test_create_bounding_box_from_node_list(self):
        """computes bbox for single 7-node island"""
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        node_ids = {0, 1, 2, 3, 4, 5, 6, 7}

        bbox = ami_skeleton.extract_bbox_for_nodes(node_ids)
        assert bbox == ((661.0, 863.0), (82.0, 102.0))

    def test_remove_pixels_in_bounding_box_arrows1(self):
        image = io.imread(Resources.BIOSYNTH1_ARROWS)
        bbox = ((82, 102), (661, 863))
        image = AmiGraph.set_bbox_pixels_to_color(bbox, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    def test_remove_pixels_in_bounding_boxes_from_islands_arrows1(self):
        image = io.imread(Resources.BIOSYNTH1_ARROWS)
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        bboxes = ami_skeleton.create_bboxes_for_connected_components()
        dd = 2  #  to overcome some of the antialiasing
        for bbox in bboxes:
            bbox = ((bbox[0][0]-dd, bbox[0][1]+dd), (bbox[1][0]-dd, bbox[1][1]+dd))
            AmiGraph.set_bbox_pixels_to_color(bbox, image)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    def test_remove_all_pixels_in_bounding_boxes_from_islands(self):
        image = io.imread(Resources.BIOSYNTH1)
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1)
        bboxes = ami_skeleton.create_bboxes_for_connected_components()
        dd = 2  #  to overcome some of the antialiasing
        for bbox in bboxes:
            bbox = ((bbox[0][0]-dd, bbox[0][1]+dd), (bbox[1][0]-dd, bbox[1][1]+dd))
            AmiGraph.set_bbox_pixels_to_color(bbox, image, color=160)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        return

    def test_remove_pixels_in_arrow_bounding_boxes_from_islands_text1(self):
        ami_skeleton = AmiSkeleton()
        # arrows_image = io.imread(Resources.BIOSYNTH1_ARROWS)
        arrows_image = ami_skeleton.create_grayscale_from_file(Resources.BIOSYNTH1_ARROWS)

        cropped_image = ami_skeleton.create_grayscale_from_file(Resources.BIOSYNTH1_CROPPED)
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        bboxes_arrows = ami_skeleton.create_bboxes_for_connected_components()
        dd = 2  #  to overcome some of the antialiasing
        for bbox in bboxes_arrows:
            bbox = ((bbox[0][0]-dd, bbox[0][1]+dd), (bbox[1][0]-dd, bbox[1][1]+dd))
            AmiGraph.set_bbox_pixels_to_color(bbox, cropped_image, color=127)
        fig, ax = plt.subplots()
        ax.imshow(cropped_image, cmap='gray')
        if ami_skeleton.interactive:
            plt.show()
        return

    def test_flood_fill_first_component(self):
        ami_skeleton = AmiSkeleton()
        component_index = 0 # as example
        ami_skeleton.read_image_plot_component(component_index, Resources.BIOSYNTH1_ARROWS)
        return

    def test_flood_fill_many_components(self):
        ami_skeleton = AmiSkeleton()
        path = Resources.BIOSYNTH1_ARROWS
        ami_skeleton.create_and_plot_all_components(path)
        return

    def test_flood_fill_many_components_select(self):
        ami_skeleton = AmiSkeleton()
        path = Resources.BIOSYNTH1_CROPPED
        ami_skeleton.create_and_plot_all_components(path, min_size=[30, 30])
        return

    def test_flood_fill_many_components_biosynth3(self):
        """EXAMPLE finds the 5 arrows"""
        ami_skeleton = AmiSkeleton(title="biosynth3")
        ami_skeleton.interactive = True
        path = Resources.BIOSYNTH3
        ami_skeleton.create_and_plot_all_components(path, min_size=[30, 30])
        return

    def test_flood_fill_many_components_1(self):
        AmiSkeleton().create_and_plot_all_components(Resources.BIOSYNTH1_ARROWS)
        return

    def test_hocr_to_svg_biosynth1(self):
        ami_skeleton = AmiSkeleton()

        biosynth_html = str(Resources.BIOSYNTH1_HOCR)
        ami_skeleton.create_svg_from_hocr(biosynth_html, "biosynth_1.svg")

    def test_hocr_to_svg_biosynth3(self):
        """creates textboxes for HOCR put and writes to temp/textbox"""
        ami_skeleton = AmiSkeleton()

        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH3_HOCR), "biosynth_3.svg")

    def test_hocr_to_svg_biosynth4to8(self):
        """creates textboxes for HOCR put and writes to temp/textbox"""
        ami_skeleton = AmiSkeleton()

        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH4_HOCR), "biosynth_4.svg")
        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH5_HOCR), "biosynth_5.svg") 
        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH6_HOCR), "biosynth_6.svg") 
        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH7_HOCR), "biosynth_7.svg") 
        ami_skeleton.create_svg_from_hocr(str(Resources.BIOSYNTH8_HOCR), "biosynth_8.svg")   

    def test_flood_fill_many_components_biosynth4to8(self):
        ami_skeleton = AmiSkeleton()
        ami_skeleton.create_and_plot_all_components(Resources.BIOSYNTH4, min_size=[30, 30])
        ami_skeleton.create_and_plot_all_components(Resources.BIOSYNTH5, min_size=[30, 30])
        ami_skeleton.create_and_plot_all_components(Resources.BIOSYNTH6, min_size=[30, 30])
        ami_skeleton.create_and_plot_all_components(Resources.BIOSYNTH7, min_size=[30, 30])
        ami_skeleton.create_and_plot_all_components(Resources.BIOSYNTH8, min_size=[30, 30])
        return