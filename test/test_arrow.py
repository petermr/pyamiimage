import logging
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from lxml import etree

from ..pyimage.ami_arrow import AmiArrow, AmiNetwork
from ..pyimage.ami_graph_all import AmiGraph, AmiIsland
from ..pyimage.svg import SVGSVG, SVGArrow, SVGG, SVGRect, ns_xpath, SVG_NS
from ..pyimage.bbox import BBox
from ..test.resources import Resources

logger = logging.getLogger(__name__)


class TestArrow:

    def setup_class(self):
        """
        resources are created once only in self.resources.create_ami_graph_objects()
        Make sure you don't corrupt them
        we may need to add a copy() method
        :return:
        """
        self.resources = Resources()
        self.resources.create_ami_graph_objects()

    def setup_method(self, method):
        self.arrows1_ami_graph = self.resources.arrows1_ami_graph
        self.islands1 = self.arrows1_ami_graph.get_or_create_ami_islands()
        assert 4 == len(self.islands1)
        self.double_arrow_island = self.islands1[0]
        self.no_heads = self.islands1[1]
        self.branched_two_heads_island = self.islands1[2]
        self.one_head_island = self.islands1[3]
        assert [21, 22, 23, 24, 25] == list(self.one_head_island.node_ids)
        assert self.one_head_island.ami_graph == self.arrows1_ami_graph
        assert self.one_head_island.island_nx_graph is not None

        # complete image includes arrows and text
        # self.biosynth1_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1)
        # self.biosynth3_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH3)
        self.biosynth1_ami_graph = self.resources.biosynth1_ami_graph
        self.biosynth3_ami_graph = self.resources.biosynth3_dto.ami_graph
        self.biosynth6_compounds_ami_graph = self.resources.biosynth6_compounds_dto.ami_graph

    def test_extract_single_arrow(self):
        ami_graph = self.one_head_island.ami_graph
        assert len(self.one_head_island.node_ids) == 5, \
            f"single arrow should have 5 nodes, found {len(self.one_head_island.node_ids)}"
        list1 = AmiGraph.get_node_ids_from_graph_with_degree(ami_graph.nx_graph, 1)
        assert len(list1) == 20
        list2 = AmiGraph.get_node_ids_from_graph_with_degree(self.one_head_island.island_nx_graph, 1)
        assert list2 == [21, 22, 23, 25], f"{__name__} ligands found {list2} expected {[21, 22, 23, 25]}"
        longest_edge = ami_graph.find_longest_edge(24)
        assert longest_edge[0] == (24, 21)
        assert longest_edge[1] == pytest.approx(30.0)
        node0, central, other_dict = ami_graph.get_angles_round_node(24)
        for idx in other_dict:
            print(f"{node0} - {central} - {idx} = {other_dict[idx]}")

    def test_double_arrow(self):
        assert len(self.double_arrow_island.node_ids) == 8, \
            f"double arrow should have 8 nodes, found {len(self.double_arrow_island.node_ids)}"
        nodes4 = self.double_arrow_island.get_node_ids_of_degree(4)
        assert nodes4 == [2, 4], f"nodes or degree 4 should be {[2, 4]}"
        assert self.double_arrow_island.get_node_ids_of_degree(3) == []
        assert self.double_arrow_island.get_node_ids_of_degree(1) == [0, 1, 3, 5, 6, 7]

    def test_branched_two_heads(self):
        """
        one-tailed arrow that bifurcates into 2 heads
        :return:
        """
        TestArrow.assert_arrows(self.branched_two_heads_island,
                                {1: [10, 11, 14, 15, 16, 17, 20], 2: [], 3: [12], 4: [13, 18]})

    def test_no_heads(self):
        assert len(self.no_heads.node_ids) == 4, \
            f"no heads should have 4 nodes, found {len(self.no_heads.node_ids)}"
        TestArrow.assert_arrows(self.no_heads, {1: [8, 9, 26], 2: [], 3: [19], 4: []})

    def test_get_edges_and_lengths(self):
        node_id = 24
        nx_edges = self.arrows1_ami_graph.get_nx_edge_list_for_node(node_id)
        assert [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)] == nx_edges, \
            "edges should be [(24, 21), (24, 22), (24, 23), (24, 25)], found {nx_edges}"
        edge_length_dict = self.arrows1_ami_graph.get_nx_edge_lengths_by_edge_list_for_node(node_id)
        edge_lengths = [v for v in edge_length_dict.values()]
        assert pytest.approx(edge_lengths, rel=0.001) == [30.00, 8.944, 9.848, 12.041]

    def test_get_interedge_angles(self):
        """test get angles round node 24"""
        node_id = 24
        interactive = False
        nx_edges = self.arrows1_ami_graph.get_nx_edge_list_for_node(node_id)
        if interactive:
            self.arrows1_ami_graph.pre_plot_edges(plt.gca())
            self.arrows1_ami_graph.pre_plot_nodes(plot_ids=True)
            plt.show()

        assert [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)] == nx_edges, \
            "edges should be [(24, 21, 0), (24, 22, 0), (24, 23, 0), (24, 25, 0)], found {nx_edges}"
        angles = []

        for edge0 in nx_edges:
            for edge1 in nx_edges:
                # only do upper triangle
                if (edge0 is not edge1) and edge0[1] < edge1[1]:
                    angle = self.arrows1_ami_graph.get_interedge_tuple_angle(edge0, edge1)
                    angles.append(angle)
        expected = [-1.107, 1.152, 3.058, 2.259, -2.117, 1.906]

        assert expected == pytest.approx(angles, 0.001), \
            f"expected {expected} found {pytest.approx(angles, 0.001)}"

    def test_whole_image_biosynth3(self):
        assert self.biosynth3_ami_graph is not None
        islands = self.biosynth3_ami_graph.get_or_create_ami_islands()
        assert len(islands) == 436
        big_islands = AmiIsland.get_islands_with_max_dimension_greater_than(40, islands)
        assert len(big_islands) == 5

        test_arrows = [
            "tail 293 - head 384 > point 384 barbs [378, 379] tail: 188,205 head: 243,205",
            "tail 476 - head 592 > point 592 barbs [572, 573] tail: 298,205 head: 354,205",
            str(None),
            "tail 628 - head 728 > point 728 barbs [719, 720] tail: 410,205 head: 466,205",
            "tail 1083 - head 1192 > point 1192 barbs [1178, 1179] tail: 849,207 head: 905,207",
        ]
        for i, island in enumerate(big_islands):
            ami_arrow = AmiArrow.create_simple_arrow(island)
            assert str(ami_arrow) == test_arrows[i]

    def test_biosynth1_arrows(self):
        # TODO get interedge angles
        """
        extract all large islands and analyse as simple arrows
        There are several false positives
        :return:
        """
        max_dim = 40
        total_islands = 484
        big_island_count = 18
        expected_arrows = [
            str(None),
            str(None),
            str(None),
            str(None),
            "tail 428 - head 434 > point 456 barbs [429, 430] tail: 258,1003 head: 300,1004",
            str(None),
            str(None),
            "tail 706 - head 718 > point 722 barbs [702, 757] tail: 435,682 head: 440,803",
            "tail 792 - head 952 > point 958 barbs [950, 951] tail: 493,557 head: 627,558",
            "tail 968 - head 932 > point 925 barbs [939, 940] tail: 634,241 head: 569,240",
            "tail 1014 - head 997 > point 1015 barbs [967, 1066] tail: 641,716 head: 641,805",
            "tail 1037 - head 1031 > point 1039 barbs [976, 1085] tail: 647,446 head: 648,336",
            "tail 1115 - head 1312 > point 1340 barbs [1304, 1308] tail: 669,241 head: 757,243",
            "tail 1205 - head 1381 > point 1382 barbs [1379, 1380] tail: 701,559 head: 860,563",
            str(None),
            "tail 1412 - head 1396 > point 1404 barbs [1383, 1445] tail: 916,765 head: 915,878",
            str(None),
            "tail 1594 - head 1702 > point 1703 barbs [1700, 1701] tail: 973,565 head: 1108,566",
        ]
        output_temp = "biosynth1_arrows.svg"

        ami_graph = self.biosynth1_ami_graph

        TestArrow.create_and_test_arrows(ami_graph, max_dim, big_island_count=big_island_count,
                                         expected_arrows=expected_arrows,
                                         output_temp=output_temp, total_islands=total_islands)

    def test_biosynth3_arrows(self):
        """
        extract all large islands and analyse as simple arrows
        full defaults except output
        :return:
        """
        TestArrow.create_and_test_arrows(self.biosynth3_ami_graph, 40, output_temp="biosynth3_arrows.svg")

    def test_biosynth6_compounds_arrows(self):
        # TODO get interedge angles
        """
        extract all large islands and analyse as simple arrows
        There are several false positives
        :return:
        """
        max_dim = 40
        total_islands = 169
        big_island_count = 8

        expected_arrows = [
            str(None),
            str(None),
            str(None),
            str(None),
            "tail 428 - head 434 > point 456 barbs [429, 430]",
            str(None),
            str(None),
            "tail 706 - head 718 > point 722 barbs [702, 757]",
            "tail 792 - head 952 > point 958 barbs [950, 951]",
            "tail 968 - head 932 > point 925 barbs [939, 940]",
            "tail 1014 - head 997 > point 1015 barbs [967, 1066]",
            "tail 1037 - head 1031 > point 1039 barbs [976, 1085]",
            "tail 1115 - head 1312 > point 1340 barbs [1304, 1308]",
            "tail 1205 - head 1381 > point 1382 barbs [1379, 1380]",
            str(None),
            "tail 1412 - head 1396 > point 1404 barbs [1383, 1445]",
            str(None),
            "tail 1594 - head 1702 > point 1703 barbs [1700, 1701]",
        ]
        TestArrow.create_and_test_arrows(self.biosynth6_compounds_ami_graph, max_dim, big_island_count=big_island_count,
                                         expected_arrows=None,
                                         output_temp="biosynth6_compounds_arrows.svg", total_islands=total_islands)

    # @unittest.skip("Not yet implemented")
    def test_several_files(self):
        """
        Iterate over many images, tests the arrows and ouputs
        uses a dictiomary of parameters. This will be close to a commandline

        NYI
        """
        image_dict = {}
        image_dict["biosynth3"] = {'input': None, "ami_graph": self.biosynth3_ami_graph,
                                   "temp_output": "biosynth3_arrows.svg"}
        print(image_dict.keys())
        for key in image_dict.keys():
            param_dict = image_dict[key]
            print(param_dict)
            TestArrow.create_and_test_arrows(param_dict["ami_graph"], 40, output_temp=param_dict["temp_output"])

    def test_arrows_and_text_biosynth6(self):
        """simplest reaction pathway of 8 steps"""
        self.biosynth6_compounds_ami_graph = self.resources.biosynth6_compounds_dto.ami_graph
        image = self.resources.biosynth6_compounds_dto.image

    def test_validate_arrows_text_biosynth1(self):
        """
        validate prepared pathway with up/down/right/left arrows and multiple texts

        :return:
        """
        element = etree.parse(str(self.resources.BIOSYNTH1_ARROWS_TEXT_SVG))
        print("FILE", self.resources.BIOSYNTH1_ARROWS_TEXT_SVG)
        assert element is not None, f"{self.resources.BIOSYNTH1_ARROWS_TEXT_SVG}"
        gs = ns_xpath(element, f"{{{SVG_NS}}}g")
        assert len(gs) == 2, f"2 svg:g children (a and t)  expected"

        # validate arrows input
        g_arrows = ns_xpath(element, f"{{{SVG_NS}}}g[@role='arrows']")
        assert type(g_arrows) is etree._Element, f"expected Element"
        arrows = ns_xpath(g_arrows, f"{{{SVG_NS}}}g[@role='arrow']")
        assert len(arrows) == 10, f"child g_arrows"
        """
        <svg:g role="arrow" orient="up">
            <svg:rect role="bbox" position="core" x="220" width="40" y="385" height="115" stroke-width="1.0"
             stroke="red" fill="blue" opacity="0.3"/>
            <svg:rect role="bbox" position="front" x="220" width="40" y="345" height="40" stroke-width="1.0"
             stroke="red" fill="fuchsia" opacity="0.3"/>
            <svg:rect role="bbox" position="back" x="220" width="40" y="500" height="40" stroke-width="1.0"
             stroke="red" fill="turquoise" opacity="0.3"/>
            <svg:rect role="bbox" position="left" x="180" width="40" y="385" height="115" stroke-width="1.0"
             stroke="red" fill="lime" opacity="0.3"/>
            <svg:rect role="bbox" position="right" x="260" width="40" y="385" height="115" stroke-width="1.0"
             stroke="red" fill="red" opacity="0.3"/>
            <svg:line orient="up" x1="240" y1="500" x2="240" y2="385" fill="none" stroke="black"
             stroke-width="2.0" marker-end="url(#arrowhead)"/>
        </svg:g>        
        """
        # rects
        assert arrows[0].get("role") == 'arrow', "role should be arrow"
        assert arrows[0].get("orient") == 'up', "orient should be up"
        rect = ns_xpath(arrows[0], f"./{{{SVG_NS}}}rect[@position='core']")
        assert rect is not None, f"only one core expected"
        assert rect.get("x") == "220", f"x coord of core"
        # lines
        line = ns_xpath(arrows[0], f"./{{{SVG_NS}}}line")
        assert line is not None, f"only one line expected"
        assert line.get("x1") == "240", f"x1 coord of line"

        # validate texts input
        """
        <svg:g role="text">
            <svg:rect role="bbox" x="195" width="148" y="357" height="28" stroke-width="1.0" stroke="red" fill="none"/>
            <svg:text x="195" y="385" font-size="25.2" stroke="blue" font-family="sans-serif">Phytosterols</svg:text>
        </svg:g>
        """
        g_text_container = ns_xpath(element, f"{{{SVG_NS}}}g[@role='texts']")
        assert type(g_text_container) is etree._Element, f"expected 1 g[@role='texts']"
        assert g_text_container.get("role") == "texts", f"text container"
        assert g_text_container.get("id") == "t", f"text container"
        texts = ns_xpath(g_text_container, f"{{{SVG_NS}}}g[@role='text']")
        assert type(texts) is list, f"expecting <g>"

        assert len(texts) == 15, f"child g_texts"

        text0 = texts[0]
        # rect
        t_rect0 = ns_xpath(text0, f"{{{SVG_NS}}}rect")
        assert t_rect0 is not None, "rect0"
        assert type(t_rect0) is etree._Element, f"element {t_rect0}"
        assert t_rect0.get("role") == "bbox", f"role"
        assert t_rect0.get("x") == "195", f"x"
        # text
        text0_text0 = ns_xpath(text0, f"{{{SVG_NS}}}text")
        assert type(text0_text0) is etree._Element, f"element {text0_text0}"
        assert text0_text0.get("y") == "385", f"y"

    @unittest.skip("obsolete")
    def test_analyze_front_arrows_text_biosynth1(self):
        """
        analyze prepared pathway with points of up/down/right/left arrows and multiple texts
        :return:
        """
        svgsvg = etree.parse(str(self.resources.BIOSYNTH1_ARROWS_TEXT_SVG))
        position = "front"
        self.overlap_arrows_and_text(position, svgsvg)
        return

        front_arrows = ns_xpath(svgsvg,
                                f"{{{SVG_NS}}}g[@role='arrows']/{{{SVG_NS}}}g[@role='arrow']/{{{SVG_NS}}}rect[@position='{position}']")
        assert len(front_arrows) == 10, f"arrows"
        for front_arrow_elem in front_arrows:
            front_arrow_bbox = self.get_bbox(front_arrow_elem)
            print("bbox", front_arrow_bbox)
        texts = ns_xpath(svgsvg, f"{{{SVG_NS}}}g[@role='texts']/{{{SVG_NS}}}g[@role='text']")
        assert len(texts) == 28, f"texts"
        for txt in texts:
            text_bbox_elem = ns_xpath(txt, f"{{{SVG_NS}}}rect[@role='bbox']")[0]
            text_bbox = self.get_bbox(text_bbox_elem)
            text_val = ns_xpath(txt, f"{{{SVG_NS}}}text")[0].text
            for front_arrow_elem in front_arrows:
                front_arrow_bbox = self.get_bbox(front_arrow_elem)
                overlap = text_bbox.intersect(front_arrow_bbox)
                if overlap.is_valid():
                    print("front arrow", front_arrow_bbox)
                    print("textbox", text_bbox, text_val)
                    print("overlap", overlap)

    def test_analyze_arrows_text_biosynth1(self):
        """
        analyze prepared pathway with tails of up/down/right/left arrows and multiple texts
        :return:
        """
        svgsvg = etree.parse(str(self.resources.BIOSYNTH1_ARROWS_TEXT_SVG))
        ami_network = AmiNetwork.create_from_svgsvg(svgsvg)
        ami_network.overlap_arrows_and_text()
        ami_network.write_graph(Path(Resources.TEMP_DIR, "biosynth1_network.gpml"))

    # @unittest.skip("under development")
    # def test_raw_arrows_to_bboxes(self):
    #     """
    #     raw arrows in SVG resulting from pixel analysis
    #     processed to add bounding boxes
    #     :return:
    #     """
    #     element = etree.parse(str(self.resources.BIOSYNTH1_RAW_ARROWS_SVG))
    #     assert element is not None, f"{self.resources.BIOSYNTH1_RAW_ARROWS_SVG}"
    #     arrows = ns_xpath(element, f"{{{SVG_NS}}}g[@role='arrows']/{{{SVG_NS}}}g[@role='arrow']")
    #     assert len(arrows) == 10, f"expected arrow count"
    #     for arrow_svg in arrows:
    #         svg_arrow = SVGArrow.create_from_svgg(arrow_svg)
    #         ami_arrow = AmiArrow.create_from_svg_arrow(svg_arrow)
    #         if ami_arrow is not None:
    #             print("ami arrow str:", str(ami_arrow))
    #         else:
    #             print("cannot create AmiArrow")
    #         print(ami_arrow.ge)

    @unittest.skip("Obsolete?")
    def test_write_gpml(self):
        ami_network = AmiNetwork()
        ami_network.write_graph(Path(Resources.TEMP_DIR, "test.gpml"))

    def test_create_overlap_boxes(self):
        """Create front/back/side overlap boxes
        """
        svg = SVGSVG()
        arrows = [
            [[400, 300], [500, 300]],  # PLUSX horiziontal right
            [[300, 400], [300, 500]],  # PLUSY vertical down
            [[200, 300], [100, 300]],  # MINUSX horiziontal left
            [[300, 200], [300, 100]],  # MINUSY vertical up
        ]
        expected_boxes = [
            # PLUSX right
            [
                [[400, 500], [285, 315]], [[410, 490], [225, 285]], [[410, 490], [315, 375]], [[500, 550], [285, 315]],
                [[350, 400], [285, 315]]

            ],
            # PLUSY down
            [
                [[285, 315], [400, 500]], [[225, 285], [410, 490]], [[315, 375], [410, 490]], [[285, 315], [500, 550]],
                [[285, 315], [350, 500]]

            ],
            # MINUSX left
            [
                [[100, 200], [285, 315]], [[110, 190], [225, 285]], [[110, 190], [315, 375]], [[50, 100], [285, 315]],
                [[200, 250], [285, 315]]
            ],

            # MINUSY up
            [
                [[285, 315], [100, 200]], [[225, 285], [110, 190]], [[315, 375], [110, 190]], [[285, 315], [50, 100]], [[285, 315], [200, 250]]
            ],
        ]

        """
        <svg:svg xmlns:svg="http://www.w3.org/2000/svg" width="1400.0" height="1200.0">
	<svg:defs>
		<svg:marker id="arrowhead" markerWidth="10.0" markerHeight="7.0" refX="10.0" refY="3.5" orient="auto">
			<svg:polygon points="0 0, 10 3.5, 0 7" fill="red" stroke="red" stroke-width="1"/>
		</svg:marker>
	</svg:defs>
	<svg:g role="arrows">
		<svg:g id="a0" role="arrow" orient="up">
			<svg:rect role="bbox" position="core" x="220" width="40" y="385" height="115" stroke-width="1.0" stroke="red" fill="blue" opacity="0.3"/>
			<svg:rect role="bbox" position="front" x="220" width="40" y="345" height="40" stroke-width="1.0" stroke="red" fill="fuchsia" opacity="0.3"/>
			<svg:rect role="bbox" position="back" x="220" width="40" y="500" height="40" stroke-width="1.0" stroke="red" fill="turquoise" opacity="0.3"/>
			<svg:rect role="bbox" position="left" x="180" width="40" y="385" height="115" stroke-width="1.0" stroke="red" fill="lime" opacity="0.3"/>
			<svg:rect role="bbox" position="right" x="260" width="40" y="385" height="115" stroke-width="1.0" stroke="red" fill="red" opacity="0.3"/>
			<svg:line orient="up" x1="240" y1="500" x2="240" y2="385" fill="none" stroke="black" stroke-width="2.0" marker-end="url(#arrowhead)"/>
			<svg:title>a0</svg:title>

        """
        for expected_box in expected_boxes:
            gg = SVGG()
            gg.set_attribute("role", "arrow")
            svg.append(gg)
            for xy_ranges in expected_box:
                bbox = BBox(xy_ranges=xy_ranges, swap_minmax=True)
                ranges = bbox.get_ranges()
                g = SVGG()
                svg_rect = SVGRect(xy_ranges = ranges)
                svg_rect.set_attribute("fill", "none")
                svg_rect.set_attribute("stroke", "red")
                svg_rect.set_attribute("opacity", "0.3")
                g.append(svg_rect)
                gg.append(g)
        print("svg: ", svg.tostring(pretty_print=True))
        path = Path(Resources.TEMP_DIR, "arrow_bboxes.svg")
        with open(path, "w") as f:
            f.write(svg.tostring(pretty_print=True))

        for arrow, exp_boxes in zip(arrows, expected_boxes):
            print("a ", arrow)
            ami_arrow = AmiArrow()
            ami_arrow.svg_arrow = SVGArrow(tail_xy=arrow[0], head_xy=arrow[1])
            box_tuple = ami_arrow.make_overlap_boxes(arrow_width=30, length=50, len_trim=10)
            # print("BOXTUPLE", box_tuple)

            for box, expect in zip(box_tuple, exp_boxes):
                assert str(box) == str(expect), f"expected {expect}"

    # ------------ helpers -------------

    @classmethod
    def create_and_test_arrows(cls, ami_graph, max_dim, total_islands=None, expected_arrows=None, big_island_count=None,
                               output_temp=None):
        islands = ami_graph.get_or_create_ami_islands()
        if total_islands:
            assert len(islands) == total_islands
        big_islands = AmiIsland.get_islands_with_max_dimension_greater_than(max_dim, islands)
        if big_island_count:
            assert len(big_islands) == big_island_count
        svg = SVGSVG()
        SVGArrow.create_arrowhead(svg)
        g = SVGG()
        svg.append(g)
        for i, island in enumerate(big_islands):
            ami_arrow = AmiArrow.create_simple_arrow(island)
            if ami_arrow is not None:
                g.append(ami_arrow.get_svg())
            else:
                bbox = island.get_or_create_bbox()
                svg_box = SVGRect(bbox=bbox)
                svg_box.set_stroke("blue")
                svg_box.set_fill("none")
                g.append(svg_box)
            if expected_arrows is not None:
                assert str(ami_arrow) == expected_arrows[i]

        # output svg
        if output_temp:
            parent = Path(__file__).parent.parent
            path = Path(parent, f"temp/{output_temp}")
            with open(path, "wb") as f:
                f.write(etree.tostring(svg.element))
            assert path.exists(), f"{path} should exist"

    @classmethod
    def assert_arrows(cls, ami_graph, node_id_dict):
        """

        :param ami_graph: ami_graph or island
        :param node_id_dict:
        :return:
        """
        for degree in node_id_dict:
            AmiGraph.assert_nodes_of_degree(ami_graph, degree, node_id_dict[degree])
