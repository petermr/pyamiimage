from pathlib import Path

import PIL.Image
from PIL import Image
from pyimage import octree

class TestOctree():

    def test_red_black(self):
        image_name = "red_black_cv.png"
        # image_name = "purple_ocimum_basilicum.png"

        path = Path(Path(__file__).parent.parent, "assets", image_name)
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)

        # ImageLib.image_show(img)
        # print(img)

        out_image, palette, palette_image = octree.quantize(img, size=4)
        print(f"image {type(out_image)} ... {out_image}")
        out_image.show()

    def test_81481_diagram(self):
        size = 6
        path = Path(Path(__file__).parent.parent, "test/alex_pico/emss-81481-f001.png")

        out_image = self.octree_and_show(path, size)
        Image.getcolors(maxcolors=256)

    def octree_and_show(self, path, size):
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)
        out_image, palette, palette_image = octree.quantize(img, size=size)
        print(f"image {type(out_image)} ... {out_image}")
        out_image.save(Path(path.parent, "test1.png"), "png")
        out_image.show()
        return out_image

    def test_81481_octree_new(self):

        """Image.quantize(colors=256, method=None, kmeans=0, palette=None, dither=1)[source]
Convert the image to ‘P’ mode with the specified number of colors.

Parameters
colors – The desired number of colors, <= 256
method –
MEDIANCUT (median cut), MAXCOVERAGE (maximum coverage), FASTOCTREE (fast octree), LIBIMAGEQUANT (libimagequant; check support using PIL.features.check_feature() with feature="libimagequant").
By default, MEDIANCUT will be used.
The exception to this is RGBA images. MEDIANCUT and MAXCOVERAGE do not support RGBA images, so FASTOCTREE is used by default instead.
kmeans – Integer
palette – Quantize to the palette of given PIL.Image.Image.
dither – Dithering method, used when converting from mode “RGB” to “P” or from “RGB” or “L” to “1”. Available methods are NONE or FLOYDSTEINBERG (default). Default: 1 (legacy setting)
Returns A new image
"""
        path = Path(Path(__file__).parent.parent, "test/alex_pico/emss-81481-f001.png")
        img = Image.open(path)
        img_out = img.quantize(colors=16, method=PIL.Image.FASTOCTREE)
        img_out.save(Path(path.parent, "test_foctree.png"), "png")
