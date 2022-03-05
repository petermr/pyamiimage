from lxml import etree
from lxml.etree import Element, QName
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from ..pyimage.bbox import BBox
    from ..pyimage.tesseract_hocr import TesseractOCR
except:
    from pyimage.bbox import BBox
    from pyimage.tesseract_hocr import TesseractOCR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HocrText:

    E_G = 'g'
    E_RECT = 'rect'
    E_SVG = 'svg'
    E_TEXT = "text"

    A_BBOX = "bbox"
    A_FILL = "fill"
    A_FONT_SIZE = "font-size"
    A_FONT_FAMILY = "font-family"
    A_HEIGHT = "height"
    A_STROKE = "stroke"
    A_STROKE_WIDTH = "stroke-width"
    A_TITLE = "title"
    A_WIDTH = "width"
    A_XLINK = 'xlink'
    A_X = "x"
    A_Y = "y"

    def parse_hocr_title(self, title):
        """
         title="bbox 336 76 1217 111; baseline -0.006 -9; x_size 28; x_descenders 6; x_ascenders 7"

        :param title:
        :return:
        """
        if title is None:
            return None
        parts = title.split("; ")
        title_dict = {}
        for part in parts:
            vals = part.split()
            kw = vals[0]
            if kw == self.A_BBOX:
                val = ((vals[1], vals[3]), (vals[2], vals[4]))
            else:
                val = vals[1:]
            title_dict[kw] = val
        return title_dict

    def create_svg_from_hocr(self, hocr_html, target_svg=None):
        """

        :param hocr_html:
        :type: string
        :param target_svg: not yet used
        :return:
        """
        html = etree.parse(hocr_html) 
        word_spans = html.findall("//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
        svg = Element(QName(XMLNamespaces.svg, self.E_SVG), nsmap={
            self.E_SVG: XMLNamespaces.svg,
            self.A_XLINK: XMLNamespaces.xlink,
        })
        for word_span in word_spans:
            title = word_span.attrib[self.A_TITLE]
            title_dict = self.parse_hocr_title(title)
            bbox = title_dict[self.A_BBOX]
            text = word_span.text
            g = self.create_svg_text_box_from_hocr(bbox, text)
            svg.append(g)
        bb = etree.tostring(svg, encoding='utf-8', method='xml')
        s = bb.decode("utf-8")
        path_svg = Path(Path(__file__).parent.parent, "temp", "textbox.svg")
        with open(path_svg, "w", encoding="UTF-8") as f:
            f.write(s)
            print(f"Wrote textboxes to {path_svg}")

    def create_svg_text_box_from_hocr(self, bbox, txt):

        g = Element(QName(XMLNamespaces.svg, self.E_G))
        height = int(bbox[1][1]) - int(bbox[1][0])
        print("height", height)

        rect = Element(QName(XMLNamespaces.svg, self.E_RECT))
        rect.attrib[self.A_X] = bbox[0][0]
        rect.attrib[self.A_WIDTH] = str(int(bbox[0][1]) - int(bbox[0][0]))
        rect.attrib[self.A_Y] = str(int(bbox[1][0]))  # kludge for offset of inverted text
        rect.attrib[self.A_HEIGHT] = str(height)
        rect.attrib[self.A_STROKE_WIDTH] = "1.0"
        rect.attrib[self.A_STROKE] = "red"
        rect.attrib[self.A_FILL] = "none"
        g.append(rect)

        text = Element(QName(XMLNamespaces.svg, self.E_TEXT))
        text.attrib[self.A_X] = bbox[0][0]
        text.attrib[self.A_Y] = str(int(bbox[1][0]) + height)
        text.attrib[self.A_FONT_SIZE] = str(0.9 * height)
        text.attrib[self.A_STROKE] = "blue"
        text.attrib[self.A_FONT_FAMILY] = "sans-serif"

        text.text = txt
        g.append(text)
        return g


class TextBox:
    def __init__(self):
        self.text = None
        self.bbox = None

    @classmethod
    def find_text_boxes(cls, elem):
        text_boxes = []
        phrases, bboxes = TesseractOCR.find_phrases(elem)
        # TODO mend this
        for phrase, bbox in zip(phrases, bboxes):
            assert type(phrase) is str, f"type of phrase {type(phrase)}"
            text_box = TextBox.create_text_box(phrase, bbox)
            text_boxes.append(text_box)
        return text_boxes

    @classmethod
    def create_text_box(cls, text, bbox):
        assert type(text) is str, f"{type(text)} of {text} should be str"
        assert type(bbox) is list
        assert len(bbox) == 4
        text_box = TextBox()
        text_box.text = text
        text_box.bbox = BBox([[bbox[0], bbox[2]], [bbox[1], bbox[3]]])
        return text_box

    def create_svg(self):
        logger.warning("SVG NYI")

    @classmethod
    def find_and_plot_text_boxes(cls, elem, image):
        text_boxes = TextBox.find_text_boxes(elem)
        text_boxes1 = []
        for text_box in text_boxes:
            if TextUtil.is_text_from_tesseract(text_box.text):
                assert type(text_box) is TextBox, f"cannot add {type(text_box)} as TextBox"
                text_boxes1.append(text_box)
        assert len(text_boxes1) > 0, "require non_zero count of text_boxes"
        logger.info(f"{__name__} plotting {len(text_boxes1)} text_boxes of type {type(text_boxes1[0])}")
        fig, ax = plt.subplots()
        TextBox.plot_text_box_boxes(image, ax, text_boxes1)
        fig.tight_layout()
        plt.show()

    @classmethod
    def plot_text_box_boxes(cls, img_array, ax, text_boxes1):
        for text_box in text_boxes1:
            assert type(text_box) is TextBox, f"should be TextBox found {type(text_box)}"
            assert type(text_box.bbox) is BBox, f"expected BBox found {type(text_box.bbox)}"
            TextBox.add_bbox_rect(ax, text_box.bbox)
        ax.imshow(img_array)

    @classmethod
    def add_bbox_rect(cls, axis, bbox, linewidth=1, edgecolor="red", facecolor="none"):
        """
        adds rectangle to axis subplot
        :param axis: axis from matplotlib subplots
        :param bbox: BBox from pyamiimage or its ranges
        :param linewidth: linewidth of plotted rect (1)
        :param edgecolor: stroke color of line ("red")
        :param facecolor: fill of rect ("none")
        :return:
        """
        assert type(bbox) is BBox, f"bbox should be BBox, found {type(bbox)}"
        xyr = bbox.xy_ranges
        rect = patches.Rectangle((xyr[0][0], xyr[1][0]), bbox.get_width(), bbox.get_height(),
                                 linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
        axis.add_patch(rect)


class TextUtil:
    @classmethod
    def is_text_from_tesseract(cls, text):
        """
        some empirical acceptance of Tesseract output
        allows isalnum characters, spaces, commas
        :param text: extracted text
        :return: False if probably a garble
        """
        # print("text util ", text)
        # logger.warning(f"txt: {text}")
        txt = text
        if len(txt) <= 1:
            return False
        # fail on repeated '-' or '.' as probably a garble from horizontal graphic line
        if '--' in txt or '..' in txt:
            return False
        # remove common punctuation
        for s in "-.,;?()[]+ ":
            txt = txt.replace(s, "")
        # fail on any non-alphanum string (includin
        if not txt.isalnum():
            logger.warning(f"{__name__} rejected {text}")
            return False

        return True


class XMLNamespaces:
    svg = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"
