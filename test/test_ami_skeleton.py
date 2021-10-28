"""Integration of image processing, binarization, skeletonization and netwprk analysis"""
import matplotlib.pyplot as plt
from skan.pre import threshold

from test.resources import Resources
from skimage import filters, color, io, data, draw
from skimage.segmentation import flood_fill
import numpy as np
import networkx as nx
import sknw
from lxml import etree
from pyimage.graph_lib import AmiSkeleton, AmiIsland, AmiGraph
from lxml.etree import Element, ElementTree
from pathlib import Path


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
        ami_skel = AmiSkeleton()
        skeleton = ami_skel.create_white_skeleton_from_file(Resources.BIOSYNTH1_ARROWS)
        # build graph from skeleton
        ami_skel.nx_graph = sknw.build_sknw(skeleton)
        ami_skel.plot_nx_graph()

    def test_skeleton_to_graph_text(self):
        ami_skel = AmiSkeleton()
        ami_skel.binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH1_TEXT)

    def test_skeleton_to_graph_path1(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH1)

    def test_skeleton_to_graph_path2(self):
        AmiSkeleton().binarize_skeletonize_sknw_nx_graph_plot(Resources.BIOSYNTH2)

    def test_skeleton_to_graph_path3(self):
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
        assert bbox == ((82.0, 102.0), (661.0, 863.0))

    def test_create_bounding_boxes_from_node_list(self):
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        bboxes = AmiSkeleton.create_bboxes_for_connected_components(ami_skeleton)
        assert bboxes == [((82.0, 102.0), (661.0, 863.0)),
                         ((117.0, 313.0), (391.0, 953.0)),
                         ((148.0, 236.0), (991.0, 1064.0)),
                         ((252.0, 294.0), (992.0, 1009.0))]

    def test_create_bounding_box_from_node_list(self):
        ami_skeleton = AmiSkeleton()
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        node_ids = {0, 1, 2, 3, 4, 5, 6, 7}

        bbox = ami_skeleton.extract_bbox_for_nodes(node_ids)
        assert bbox == ((82.0, 102.0), (661.0, 863.0))

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
        plt.show()
        return

    def test_flood_fill(self):
        colors = [0x00ff0000, 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00000000]
        ami_skeleton = AmiSkeleton()
        # arrows_image = io.imread(Resources.BIOSYNTH1_ARROWS)
        arrows_image = ami_skeleton.create_grayscale_from_file(Resources.BIOSYNTH1_ARROWS)

        cropped_image = ami_skeleton.create_grayscale_from_file(Resources.BIOSYNTH1_CROPPED)
        nx_graph = ami_skeleton.create_nx_graph_via_skeleton_sknw(Resources.BIOSYNTH1_ARROWS)
        ami_skeleton.get_nodes_and_edges_from_nx_graph()
        components = ami_skeleton.get_connected_components()
        for i, component in enumerate(components):
            ami_skeleton.flood_fill(component, colors[i % len(colors)])


        fig, ax = plt.subplots()
        ax.imshow(cropped_image, cmap='gray')
        plt.show()
        return

    def test_hocr_to_svg(self):
        """
        Convert HOCR (HTML) to SVG
        :return: svg

        Typical input
        <?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
 <head>
  <title></title>
  <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
  <meta name='ocr-system' content='tesseract 4.1.1' />
  <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf'/>
 </head>

 <body>
  <div class='ocr_page' id='page_1' title='image "biosynth_path_1.png"; bbox 0 0 1515 1167; ppageno 0'>
   <div class='ocr_carea' id='block_1_1' title="bbox 687 43 846 71">
    <p class='ocr_par' id='par_1_1' lang='eng' title="bbox 687 43 846 71">
     <span class='ocr_line' id='line_1_1' title="bbox 687 43 846 71; baseline -0.006 -5; x_size 28; x_descenders 6; x_ascenders 7">
      <span class='ocrx_word' id='word_1_1' title='bbox 687 44 807 66; x_wconf 91'>Isomerase</span>
      <span class='ocrx_word' id='word_1_2' title='bbox 815 43 846 71; x_wconf 90'>(?)</span>
     </span>
    </p>
   </div>
   <div class='ocr_carea' id='block_1_2' title="bbox 336 76 1217 111">
    <p class='ocr_par' id='par_1_2' lang='eng' title="bbox 336 76 1217 111">
     <span class='ocr_header' id='line_1_2' title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7">
      <span class='ocrx_word' id='word_1_3' title='bbox 336 80 474 108; x_wconf 92'>Isopentenyl</span>
      ...
      <span class='ocrx_word' id='word_1_8' title='bbox 1072 76 1217 104; x_wconf 96'>diphosphate</span>
     </span>
    </p>
   </div>

        """
        ami_skeleton = AmiSkeleton()

        biosynth_html = str(Resources.BIOSYNTH1_HOCR)
        html = etree.parse(biosynth_html)
        word_spans = html.findall("//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
        svg = Element("svg")
        svg.attrib["xmlns"] = "http://www.w3.org/2000/svg"

        for word_span in word_spans:
            title = word_span.attrib["title"]
            title_dict = ami_skeleton.parse_hocr_title(title)
            bbox = title_dict["bbox"]
            text = word_span.text
            g = ami_skeleton.create_svg_text_box_from_hocr(bbox, text)
            svg.append(g)

        bb = etree.tostring(svg, encoding='utf-8', method='xml')
        s = bb.decode("utf-8")
        path_svg = Path(Path(__file__).parent.parent, "temp", "textbox.svg")
        with open(path_svg, "w", encoding="UTF-8") as f:
            f.write(s)
            print(f"Wrote textboxes to {path_svg}")


