from lxml import etree
from lxml.etree import Element, QName
from pathlib import Path


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


class XMLNamespaces:
    svg = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"

