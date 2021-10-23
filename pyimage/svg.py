"""Common SVG elements in lightweight code"""
from lxml.etree import ElementTree, Element


class AbsSVG(Element):
    def __init__(self):
        super().__init__()


class SVGG(AbsSVG):
    def __init__(self):
        super().__init__()


class SVGShape(AbsSVG):
    def __init__(self):
        super().__init__()

    def get_bbox(self):
        pass


class SVGRect(SVGShape):
    def __init__(self):
        super().__init__()


class SVGPath(SVGShape):
    def __init__(self):
        super().__init__()


class SVGLine(SVGShape):
    def __init__(self):
        super().__init__()


class SVGCircle(SVGShape):
    def __init__(self):
        super().__init__()


class SVGText(AbsSVG):
    def __init__(self):
        super().__init__()


class SVGTextBox(AbsSVG):
    """This will contain text and a shape (for drawing)"""

    def __init__(self, svg_text=None, svg_shape=None):
        super().__init__()
        self.svgg = SVGG()
        self.set_text(svg_text)
        self.set_shape(svg_shape)

    def set_text(self, svg_text):
        """
        Need to check for uniqueness
        :param svg_text:
        :return:
        """
        if svg_text is not None:
            assert type(svg_text) is SVGText
            self.svgg.append(svg_text)

    def set_shape(self, svg_shape):
        if svg_shape is not None:
            assert type(svg_shape) is SVGShape
            self.svgg.append(svg_shape)


class AmiArrow(AbsSVG):
    """"""
