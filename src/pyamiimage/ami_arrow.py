import logging
import math
from lxml import etree
from lxml.builder import ElementMaker
# local
from ..pyimage.ami_graph_all import AmiNode, AmiEdge
from ..pyimage.ami_util import AmiUtil
from ..pyimage.svg import SVGArrow, ns_xpath, SVG_NS, GPML_NS
from ..pyimage.bbox import BBox

logger = logging.getLogger(__name__)


class AmiArrow:
    """
    Holds the graphics primitives and the semantics of arrows
    contains an SVGArrow (which conatins basic drawing coords tail_xy and head_xy) and additional fields
    """
    HEAD_WOBBLE = 0.8 * math.pi  # allowed deviation from straightness between shaft and head
    BARB_ANGLE_DIFF = 0.1 * math.pi  # allowed asymmetry in barb angles
    MAX_BARB_TO_TAIL_ANGLE = 0.9 * math.pi / 2  # maximum angle of barbs to tail

    def __init__(self, ami_island=None):
        """
        SVG_Arrow may be created from different sources
        :param ami_island:
        """
        self.svg_arrow = SVGArrow()

        self.ami_island = ami_island
        # nodes in island
        self.hcentre_id = None  # centre of head (often a 4-branched node in skeleton graph
        self.head_id = None  # extreme point (end of SVGLine)
        self.tail_id = None  # extreme point
        self.barb_ids = None  # points on side (connected to hcentre); 0, 1 or 2
        # self.point_xy = None
        # self.tail_xy = None

    def __str__(self):
        """
        can't get this working for None
        :return:
        """
        s = f"tail {self.tail_id} - head {self.head_id} > point {self.hcentre_id} barbs {self.barb_ids}"
        if self.svg_arrow is None:
            s += "None"
        else:
            s1 = str(self.svg_arrow)
            s += " " + s1
        return s

    def set_tail_xy(self, yx):
        tail_xy = AmiUtil.swap_yx_to_xy(yx)
        self.svg_arrow.set_tail_xy(tail_xy)

    def set_head_xy(self, yx):
        head_xy = AmiUtil.swap_yx_to_xy(yx)
        self.svg_arrow.set_head_xy(head_xy)

    @classmethod
    def create_simple_arrow(cls, island):
        """
        create simple arrow from networkX island
        one head, one tail
        :param island:
        :return: AmiArrow or none
        """
        if not 5 >= len(island.node_ids) >= 4:
            logger.warning(f"cannot create simple arrow from {island.node_ids}")
            return None
        node_dict = island.create_node_degree_dict()
        logger.debug(f"\nnode dict {node_dict}")
        neighbour_count = len(island.node_ids) - 1
        try:
            central_node_id = node_dict[neighbour_count][0]
            logger.debug(f"central node {central_node_id}")
        except Exception:
            return None

        edge_dict = island.create_edge_property_dikt(central_node_id)
        logger.debug(f"edge_dict {edge_dict}")

        longest_dict = cls._find_dict_with_longest_edge(edge_dict)
        longest_edge = island.ami_graph.get_or_create_ami_edge_from_ids(central_node_id, longest_dict[AmiNode.REMOTE],
                                                                        branch_id=0)
        logger.debug(f"longest edge {longest_edge}")
        logger.debug(f"longest dict {longest_dict}")

        ami_arrow = AmiArrow(island)
        svg_arrow = ami_arrow.svg_arrow
        ami_arrow.head_id = central_node_id
        ami_arrow.tail_id = longest_dict[AmiNode.REMOTE]

        # find short lines
        short_lines = [value for value in edge_dict.values() if value != longest_dict]
        logger.debug(f"short lines {short_lines} {len(short_lines)}")

        if len(short_lines) == 3:  # 5-node arrow - normally from thinned solid triangle

            # find straight on
            barbs1 = []
            arrow_point_line = None
            for line in short_lines:
                line_tuple = AmiEdge.create_normalized_edge_id_tuple(central_node_id, line[AmiNode.REMOTE], 0)  # assume only one branch
                angle = island.ami_graph.get_interedge_tuple_angle(longest_edge.get_tuple(), line_tuple)
                assert angle is not None
                if abs(angle) > AmiArrow.HEAD_WOBBLE:
                    if arrow_point_line is not None:
                        raise ValueError(f"competing lines for head line {longest_edge} to {short_lines}")
                    arrow_point_line = line
                else:
                    barbs1.append(line)
            if arrow_point_line is None:
                raise ValueError(f"cannot find point line {longest_edge} to {short_lines}")
            ami_arrow.hcentre_id = arrow_point_line[AmiNode.REMOTE]
            if ami_arrow.hcentre_id is None:
                raise ValueError(f"cannot find point {arrow_point_line} , {central_node_id}")
        else:  # 4 node arrow
            barbs1 = short_lines
            ami_arrow.hcentre_id = ami_arrow.head_id
        barbs = barbs1
        svg_arrow.set_head_xy(AmiNode.get_xy_for_node_id(island.ami_graph.nx_graph, ami_arrow.hcentre_id))
        svg_arrow.set_tail_xy(AmiNode.get_xy_for_node_id(island.ami_graph.nx_graph, ami_arrow.tail_id))
        print(f"nodes head {svg_arrow.head_xy} tail {svg_arrow.tail_xy}")
        logger.debug(f"longest {longest_edge} point {ami_arrow.hcentre_id} barbs {barbs}")
        if len(barbs) != 2:
            raise ValueError(f" require exactly 2 barbs on arrow {barbs}")
        ami_arrow.barb_ids = [barb[AmiNode.REMOTE] for barb in barbs]

        barb_angles = [island.ami_graph.get_angle_between_nodes(ami_arrow.tail_id, ami_arrow.head_id, barb_id) for
                       barb_id in ami_arrow.barb_ids]
        logger.debug(f"barb angles {barb_angles}")
        if abs(barb_angles[0] + barb_angles[1]) > AmiArrow.BARB_ANGLE_DIFF:
            raise ValueError(f"barb angles not symmetric {barb_angles}")
        if abs(barb_angles[0] - barb_angles[1]) / 2 > AmiArrow.MAX_BARB_TO_TAIL_ANGLE:
            raise ValueError(f"barb angles not acute {barb_angles}")

        logger.debug(f"AMIARR {ami_arrow}")
        return ami_arrow

    @classmethod
    def _find_dict_with_longest_edge(cls, edge_dict):
        longest_dict = None
        for key, value in edge_dict.items():
            if longest_dict is None:
                longest_dict = value
            else:
                if value[AmiNode.PIXLEN] > longest_dict[AmiNode.PIXLEN]:
                    longest_dict = value
        return longest_dict

    def get_svg(self):
        """
        create line with arrowhead
        :return:
        """
        # svg_arrow = SVGArrow(head_xy=self.point_xy, tail_xy=self.tail_xy)
        return self.svg_arrow

    def get_orient(self, deviation=10, min_length=50):
        """
        orientation of arrow along axes
        PLUSX, PLUSY, MINUSX, MINUSY
        arrow must be horizontal (abs(x1-x2) < deviatiom), or vertical (abs(y1-y2) < deviation
        :param deviation: max difference between aligned coordinates
        :return: None if not aligned else ArrowBBox.PLUSX/PLUSY/MINUSX/MINUSY


        """
        if self.svg_arrow is None or self.svg_arrow.head_xy is None or self.svg_arrow.tail_xy is None:
            return None
        tail_xy = self.svg_arrow.tail_xy
        head_xy = self.svg_arrow.head_xy
        if abs(tail_xy[0] - head_xy[0]) < deviation:
            if abs(tail_xy[1] - head_xy[1]) > min_length:
                return ArrowBBox.PLUSY if tail_xy[1] < head_xy[1] else ArrowBBox.MINUSY
        if abs(tail_xy[1] - head_xy[1]) < deviation:
            if abs(tail_xy[0] - head_xy[0]) > min_length:
                return ArrowBBox.PLUSX if tail_xy[0] < head_xy[0] else ArrowBBox.MINUSX
        return None

    @classmethod
    def create_from_svg_arrow(cls, svg_arrow):
        """
        Creates an AmiArrow from an SVGArrow
        may be missing relation to a graph.
        At present used for development

        :param svg_arrow:
        :return:
        """
        if svg_arrow is None:
            logger.warning("null SVGArrow")
            return None
        ami_arrow = AmiArrow()
        ami_arrow.svg_arrow = svg_arrow

        return ami_arrow

    def create_bbox(self, bbox_type, width=None, length=None):
        """

        boxes:


               +-----------+  ^
               |   LEFT    |  |  width
        -------+-----------+-------|
        | BACK | ------>   | FRONT |
        -------+-----------+-------|
               |  RIGHT    |  --> length
               +-----------+

         LEFT, RIGHT, FRONT, Back are relative to arrow direction
         LEFT, RIGHT have adjustable WIDTH
         FRONT, BACK have adjustable LENGTH


        :param bbox_type: from ArrowBBox
        :param length: only used for FRONT, BACK
        :param width: only used for LEFT, RIGHT
        :return:
        """
        core_bbox = self.svg_arrow.get_bbox() if self.svg_arrow is not None else None

        if core_bbox is None or bbox_type is None:
            return None

        if bbox_type == ArrowBBox.CORE:
            bbox = core_bbox
        elif bbox_type == ArrowBBox.FRONT:
            # = core_bbox
            pass
        elif bbox_type == ArrowBBox.BACK:
            # bbox = translate_and_(bbox, translate=[delta, 0], )
            pass
        elif bbox_type == ArrowBBox.RIGHT:
            # bbox = translate_and_(bbox, translate=[delta, 0], )
            pass
        elif bbox_type == ArrowBBox.LEFT:
            # bbox = translate_and_(bbox, translate=[delta, 0], )
            pass
        else:
            logger.warning("unknown direction {}")

    def make_overlap_boxes(self, length=None, arrow_width=40, len_trim=10, colors=None):
        orient = self.get_orient()
        head_xy = self.svg_arrow.head_xy
        tail_xy = self.svg_arrow.tail_xy
        # core_bbox = BBox(xy_ranges=[
        #     [tail_xy[0], head_xy[0]] - arrow_width / 2],[]])
        xmid = (tail_xy[0] + head_xy[0]) / 2
        ymid = (tail_xy[1] + head_xy[1]) / 2
        half_width = arrow_width / 2
        # horizontal
        if orient == ArrowBBox.PLUSX:
            ymax = ymid + half_width
            ymin = ymid - half_width
            xmin = tail_xy[0]
            xmax = head_xy[0]
            bbox_core = BBox(xy_ranges=[[xmin, xmax], [ymin, ymax]], swap_minmax=True)
            bbox_front = BBox(xy_ranges=[[xmax, xmax + length], [ymin, ymax]], swap_minmax=True)
            bbox_back = BBox(xy_ranges=[[xmin - length, xmin], [ymin, ymax]], swap_minmax=True)
            bbox_right = BBox(xy_ranges=[[xmin + len_trim, xmax - len_trim], [ymax, ymax + 2 * arrow_width]], swap_minmax=True)
            bbox_left = BBox(xy_ranges=[[xmin + len_trim, xmax - len_trim], [ymin - 2 * arrow_width, ymin]], swap_minmax=True)
            if colors is not None:
                bbox_core.fill = colors["core"]

        elif orient == ArrowBBox.MINUSX:
            ymax = ymid + half_width
            ymin = ymid - half_width
            xmin = head_xy[0]
            xmax = tail_xy[0]
            bbox_core = BBox(xy_ranges=[[xmin, xmax], [ymin, ymax]], swap_minmax=True)
            bbox_front = BBox(xy_ranges=[[xmin - length, xmin], [ymin, ymax]], swap_minmax=True)
            bbox_back = BBox(xy_ranges=[[xmax, xmax + length], [ymin, ymax]], swap_minmax=True)
            bbox_right = BBox(xy_ranges=[[xmin + len_trim, xmax - len_trim], [ymax, ymax + 2 * arrow_width]], swap_minmax=True)
            bbox_left = BBox(xy_ranges=[[xmin + len_trim, xmax - len_trim], [ymin - 2 * arrow_width, ymin]], swap_minmax=True)
        # vertical
        elif orient == ArrowBBox.PLUSY:
            xmax = xmid + half_width
            xmin = xmid - half_width
            ymin = tail_xy[1]
            ymax = head_xy[1]
            bbox_core = BBox(xy_ranges=[[xmin, xmax], [ymin, ymax]], swap_minmax=True)
            bbox_front = BBox(xy_ranges=[[xmin, xmax ], [ymax, ymax + length]], swap_minmax=True)
            bbox_back = BBox(xy_ranges=[[xmin, xmax], [ymin - length, ymax]], swap_minmax=True)
            bbox_right = BBox(xy_ranges=[[xmax, xmax + 2 * arrow_width], [ymin + len_trim, ymax - len_trim]], swap_minmax=True)
            bbox_left = BBox(xy_ranges=[[xmin - 2 * arrow_width, xmin], [ymin + len_trim, ymax - len_trim]], swap_minmax=True)
        elif orient == ArrowBBox.MINUSY:
            xmax = xmid + half_width
            xmin = xmid - half_width
            ymin = head_xy[1]
            ymax = tail_xy[1]
            bbox_core = BBox(xy_ranges=[[xmin, xmax], [ymin, ymax]], swap_minmax=True)
            bbox_front = BBox(xy_ranges=[[xmin, xmax ], [ymin-length, ymin]], swap_minmax=True)
            bbox_back = BBox(xy_ranges=[[xmin, xmax], [ymax, ymax + length]], swap_minmax=True)
            bbox_right = BBox(xy_ranges=[[xmax, xmax + 2 * arrow_width], [ymin + len_trim, ymax - len_trim]], swap_minmax=True)
            bbox_left = BBox(xy_ranges=[[xmin - 2 * arrow_width, xmin], [ymin + len_trim, ymax - len_trim]], swap_minmax=True)

        bbox_core.color="blue"
        bbox_front.color="fuchsia"
        bbox_back.color="blue"
        bbox_left.color="lime"
        bbox_right.color="red"

        return bbox_core, bbox_left, bbox_right, bbox_front, bbox_back

class AmiNetwork:
    ARROW = "arrow"
    ARROWS = "arrows"
    BBOX = "bbox"
    DATABASE = "Database"
    ID = "id"
    POSITIONS = "positions"
    TEXT = "text"
    TEXTBOXES = "textboxes"
    TYPE = "type"
    UPPER_ID = "ID"
    VALUE = "value"

    def __init__(self):
        self.arrows_text_dict = dict()
        self.svgsvg = None

    @classmethod
    def create_from_svgsvg(cls, svgsvg):
        """
        process output of pixel analysis
        :param svgsvg:
        :return:
        """
        ami_network = AmiNetwork()
        ami_network.svgsvg = svgsvg
        return ami_network

    def overlap_arrows_and_text(self):
        arrows = ns_xpath(self.svgsvg,
                          f"{{{SVG_NS}}}g[@role='arrows']/{{{SVG_NS}}}g[@role='arrow']")
        text_boxes = ns_xpath(self.svgsvg, f"{{{SVG_NS}}}g[@role='texts']/{{{SVG_NS}}}g[@role='text']")
        self.arrows_text_dict = dict()

        self.arrows_dict = dict()
        self.arrows_text_dict[self.ARROWS] = self.arrows_dict

        self.textboxes_dict = dict()
        self.arrows_text_dict[self.TEXTBOXES] = self.textboxes_dict
        for arrow_elem in arrows:
            arrow_id = arrow_elem.get(self.ID)
            for position in [ArrowBBox.FRONT, ArrowBBox.BACK, ArrowBBox.RIGHT, ArrowBBox.LEFT]:
                arrow_box_elem = ns_xpath(arrow_elem,
                                          f"{{{SVG_NS}}}rect[@position='{position}']")
                arrow_bbox = AmiNetwork.get_bbox(arrow_box_elem)
                for text_box in text_boxes:
                    self.merge_texts_and_arrows(arrow_bbox, arrow_id, position, text_box)

        self.clean_overlap_in_arrows()
        self.create_reactions()

    def merge_texts_and_arrows(self, arrow_bbox, arrow_id, position, textbox):
        """merge text and arrows using position
        :param arrow_bbox: save bbox for arrow, indexed by position
        :param arrow_id: ID of arrow
        :param position: of bbox (e.g. LEFT)
        :param textbox: save textbox coordinates
        """
        textbox_id = textbox.get(self.ID)
        textbox_bbox_elem = ns_xpath(textbox, f"{{{SVG_NS}}}rect[@role='bbox']")
        textbox_bbox = AmiNetwork.get_bbox(textbox_bbox_elem)
        overlap = textbox_bbox.intersect(arrow_bbox)
        if overlap.is_valid():
            self.add_text_fields_to_dict(textbox_bbox, textbox_id, textbox)
            self.add_arrow_fields_to_dict(arrow_bbox, arrow_id, position, textbox_id)

    def add_text_fields_to_dict(self, text_bbox, textbox_id, text_element):
        """

        :param text_bbox:
        :param textbox_id:
        :param text_element: element containing one or more strings
        :return:
        """
        text_val = AmiNetwork.get_text_val(text_element)
        if textbox_id not in self.textboxes_dict:
            self.textboxes_dict[textbox_id] = dict()
        print(f"added {text_val} for {textbox_id}")
        self.textboxes_dict[textbox_id][self.VALUE] = text_val
        self.textboxes_dict[textbox_id][self.BBOX] = text_bbox
        print(f"{textbox_id} gives {self.textboxes_dict}")
        return text_val

    def add_arrow_fields_to_dict(self, arrow_bbox, arrow_id, position, text_id):
        if arrow_id not in self.arrows_dict:
            self.arrows_dict[arrow_id] = dict()
        self.arrows_dict[arrow_id][self.BBOX] = arrow_bbox
        # self.arrows_dict[arrow_id][self.TYPE] = self.ARROW
        if self.POSITIONS not in self.arrows_dict[arrow_id]:
            self.arrows_dict[arrow_id][self.POSITIONS] = dict()
        self.arrows_dict[arrow_id][self.POSITIONS][position] = text_id

    def write_graph(self, path):
        """
        write graph
        output format determined by suffix of path
        :param path: pathname, suffix can be *.gpml, *.dot (NYI), *.svg (NYI)
        :return: None
        """
        if path.suffix == ".gpml":
            self.write_graph_gpml(path)
        elif path.suffix == ".dot":
            self.write_graph_dot(path)
        elif path.suffix == ".svg":
            self.write_graph_svg(path)
        else:
            logger.error(f"unsupported format {path}")

    def write_graph_gpml(self, path):
        """
https://github.com/PathVisio/libGPML/blob/main/org.pathvisio.lib/src/main/resources/GPML2013a.xsd

    <Pathway xmlns="http://pathvisio.org/GPML/2013a" Name="Foo" Version="000" Organism="T. rex">
      <Graphics BoardWidth="484.0" BoardHeight="234.25" />
      <DataNode TextLabel="HDL-C" GraphId="bf9b8" Type="Metabolite">
        <Graphics CenterX="348.5" CenterY="184.25" Width="93.0" Height="35.5" ZOrder="32768" FontSize="12"
            Valign="Middle" Color="0000ff" />
        <Xref Database="ChEBI" ID="CHEBI:47775" />
      </DataNode>
      ...
      <Interaction>
        <Graphics ZOrder="12288" LineThickness="1.0">
          <Point X="151.0" Y="182.625" GraphRef="eb089" RelX="1.0" RelY="0.0" />
          <Point X="302.0" Y="184.25" GraphRef="bf9b8" RelX="-1.0" RelY="0.0" ArrowHead="mim-conversion" />
        </Graphics>
        <Xref Database="" ID="" />
      </Interaction>
      <InfoBox CenterX="0.0" CenterY="0.0" />
      <Biopax />
    </Pathway>

            """
        """
        E = ElementMaker(namespace=GPML_NS)
        BoardWidth = "484.0"
        BoardHeight = "234.25"
        ami_network = "AmiNetwork"
        Version = "000"
        organism = "T. rex"
        gpml_root = self.make_gpml_root(BoardHeight, BoardWidth, Version, ami_network, organism)
        X = "348.5"
        Y = "184.25"
        text_label = "TextLabel"
        text_id = "t0"
        type = "Metabolite"
        width = "93.0"
        label = "35.5"
        data_node1 = self.make_text_data_node(X, Y, text_label, text_id, type, width, label, Color="0000ff")
        data_node2 = self.make_text_data_node(X, Y, text_label, text_id, type, width, label, Color="0000ff")
        point_X1 = "123"
        point_Y1 = "456"
        point_X2 = "654"
        point_Y2 = "321"
        idref1 = "t0"
        idref2 = "t2"
        interact = self.make_interaction(idref1, idref2, point_X1, point_X2, point_Y1, point_Y2)
        gpml_root.append(data_node1)
        gpml_root.append(data_node2)
        gpml_root.append(interact)
        """
        total_bbox = self.get_total_bbox()
        print(f"total box: {total_bbox}")

        gpml_root = self.make_gpml_root(total_bbox.get_xrange()[1], total_bbox.get_yrange()[1], network_name="unknown_network", organism="unknown")
        self.create_and_add_text_nodes(gpml_root)
        self.create_and_add_interactions(gpml_root)
        with open(path, "w") as f:
            f.write(etree.tostring(gpml_root, pretty_print=True).decode(encoding="UTF-8"))

    def create_and_add_interactions(self, gpml_root):
        for arrow_id in self.arrows_dict:
            arrow = self.arrows_dict[arrow_id]
            bbox = self.arrows_dict[arrow_id][self.BBOX]
            print(f"arrow: {arrow_id} {arrow}, {bbox}")
            # arrow: a0 {'bbox': [[220, 260], [500, 540]], 'positions': {'front': 't0', 'back': 't1'}}
            positions_dict = arrow[self.POSITIONS]
            front_id = positions_dict[ArrowBBox.FRONT]
            back_id = positions_dict[ArrowBBox.BACK]
            if front_id is None and back_id is None:
                print(f"need front and back for arrow {positions_dict}")
            else:
                print(f" front {front_id} back {back_id}")
                gpml_interaction = self.make_gpml_interaction(front_id, back_id, bbox.get_xrange()[0],
                                                              bbox.get_xrange()[1], bbox.get_yrange()[0],
                                                              bbox.get_yrange()[1])
                gpml_root.append(gpml_interaction)

    def create_and_add_text_nodes(self, gpml_root):
        for textbox_id in self.textboxes_dict:
            bbox = self.textboxes_dict[textbox_id][self.BBOX]
            text = self.textboxes_dict[textbox_id][self.VALUE]
            Type = "Metabolite"
            gpml_data_node = self.make_text_data_node(bbox.get_xrange()[0], bbox.get_yrange()[0], text[:10],
                                                      textbox_id, Type,
                                                      Width="80", Height="30")
            # gpml_point = self.make_gpml_point(bbox.get_xrange()[0], bbox.get_yrange()[0], textbox_id)
            gpml_root.append(gpml_data_node)

    def make_gpml_interaction(self, idref1, idref2, point_X1, point_Y1, point_X2, point_Y2, ZOrder="32768", LineThickness="1.0"):
        E = ElementMaker(namespace=GPML_NS)
        return E("Interaction",
                 E("Graphics",
                   self.make_gpml_point(X=point_X1, Y=point_Y1, GraphRef=idref1),
                   self.make_gpml_point(X=point_X2, Y=point_Y2, GraphRef=idref2, ArrowHead="mim-conversion"),
                   ),
                 ZOrder=ZOrder, LineThickness=LineThickness,
                 )

    def make_gpml_root(self, BoardHeight, BoardWidth, Version="0.0.1", network_name="unknown_network", organism="unknown_organism"):
        E = ElementMaker(namespace=GPML_NS)
        return (
            E("Pathway",
              E("Graphics", BoardWidth=str(BoardWidth), BoardHeight=str(BoardHeight)),
              Name=network_name, Version=Version, Organism=organism)
        )

    def make_gpml_point(self, X, Y, GraphRef, RelX="1.0", RelY="0.0", ArrowHead=None):
        """
        make GPML point
        :param X: xcoord of point
        :param Y: ycoord of point
        :param GraphRef: id-ref which must correspond to a (point) elsewhere
        :param RelX: don't know, default = 1.0
        :param RelY: don't know, default = 0.0
        :param ZOrder: presumably Z- in painter model; default 32768
        :param LineThickness: default 1.0
        :return: GPML point

        """
        E = ElementMaker(namespace=GPML_NS)
        point = E("Point", X=str(X), Y=str(Y), GraphRef=GraphRef, RelX=str(RelX), RelY=str(RelY))
        if ArrowHead is not None:
            point.set("ArrowHead", ArrowHead)
        return point

    def make_gpml_xref(self, Database=None, ID=None):
        """GPML Xref to database
        :param Database: example - CHEBI
        :param ID: example - "CHEBI:47775"
        """
        E = ElementMaker(namespace=GPML_NS)
        xref = E("Xref")
        if Database is not None:
            xref.set(self.DATABASE, Database)
        if Database is not None:
            xref.set(self.UPPER_ID, ID)
        return xref

    def make_text_data_node(self, CenterX, CenterY, TextLabel, GraphId, Type, Width, Height, ZOrder="32768", FontSize="12",
                            Valign="Middle", Color="0000ff", Database=None, ID=None):
        """Creates a data_node for GPML
        :param CenterX: x coord
        :param CenterY: y coord
        :param TextLabel: label for node
        :param GraphId: id referenced from , e.g. gpml_point
        :param Type: type (e.g. "Metabolite")
        :param Width: presumably width of box
        :param Height: presumably height of box
        :param ZOrder: ZOrder (painters model?)
        :param FontSize:
        :param Valign: e.g. "Middle"
        :param Color: I think the line color
        :param Database: Database (may be None)
        :param ID: database ID


        """
        E = ElementMaker(namespace=GPML_NS)
        data_node = E("DataNode", TextLabel=TextLabel, GraphId=GraphId, Type=Type)
        graphics = E("Graphics", CenterX=str(CenterX), CenterY=str(CenterY), Width=str(Width), Height=str(Height), ZOrder=ZOrder,
                     FontSize=str(FontSize), Valign=str(Valign), Color=Color)
        data_node.append(graphics)
        if Database is not None:
            xref = self.make_gpml_xref(Database=Database, ID=ID)
            data_node.append(xref)
        return data_node

    @classmethod
    def get_text_val(cls, text_container):
        """gets text value from an element containing <text> element
        :param text_container:
        :return: text vallue

        """
        text_val = ns_xpath(text_container, f"{{{SVG_NS}}}text")
        if type(text_val) is etree._Element:
            text_val = text_val.text
        elif type(text_val) is list:
            text_val = text_val[0].text  # several texts in box
        return text_val

    @classmethod
    def get_bbox(cls, bbox_elem):
        """get BoundingBox from any element with X, Y , width, height
        :param bbox_elem: element with these attributes
        :return: BBox element
        """
        return BBox.create_from_xy_w_h(
            [float(bbox_elem.get(BBox.X)), float(bbox_elem.get(BBox.Y))],
            float(bbox_elem.get(BBox.WIDTH)),
            float(bbox_elem.get(BBox.HEIGHT))
        )

    def clean_overlap_in_arrows(self):
        """
        Iterates over overlaps and applies heuristics to clean them
        :return:
        """
        pass

    def create_reactions(self):
        """iterarates over dict keys for arrow and extracts back/front"""
        for key in self.arrows_text_dict.keys():
            if key == self.ARROW:
                if ArrowBBox.FRONT in self.arrows_text_dict[key] and ArrowBBox.BACK in self.arrows_text_dict[key]:
                    print(self.arrows_text_dict[key][ArrowBBox.BACK], key, self.arrows_text_dict[key][ArrowBBox.FRONT])

    def get_total_bbox(self):
        """iterates over all bboxes to get total extent
        """
        total_bbox = BBox()

        print(f"arrows dict {self.arrows_dict}")
        for arrow_id in self.arrows_dict:
            bbox = self.arrows_dict[arrow_id][self.BBOX]
            # print (f"arrow_id: {arrow_id} {bbox}")
            total_bbox = bbox if not total_bbox.is_valid() else total_bbox.union(bbox)
            # print(f"new box {total_bbox}")
        for textbox_id in self.arrows_text_dict[self.TEXTBOXES]:
            bbox = self.textboxes_dict[textbox_id][self.BBOX]
            # print (f"textbox_id: {textbox_id} {bbox}")
            total_bbox = bbox if not total_bbox.is_valid() else total_bbox.union(bbox)

        return total_bbox
        print(f"total_box {total_bbox}")

class ArrowBBox:
    CORE = "core"

    # arrow frame
    BACK = "back"
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"

    # graphics frame
    PLUSX = "plus_x"
    MINUSX = "minus_x"
    PLUSY = "plus_y"
    MINUSY = "minus_y"
