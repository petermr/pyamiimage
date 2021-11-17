#!/usr/bin/env python
#-*- coding: utf8 -*-


class Color(object):
    """
    Simple RGB class
    """

    def __init__(self, red=0, green=0, blue=0, alpha=None):
        """
        Initialize color
        """
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def to_hex(self):
        return f'{self.red:02x}' + f'{self.green:02x}' + f'{self.blue:02x}'

    def is_equal(self, col):
        # print("l", len(col))
        return len(col) == 3 and col[0] == self.red and col[1] == self.green and col[2] == self.blue

    def as_tuple(self):
        return self.red, self.green, self.blue
