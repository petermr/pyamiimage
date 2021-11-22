from pathlib import Path

from PIL import Image
from ..pyimage import octree
from ..pyimage.image_lib import Quantizer


class TestOctree:

    def test_red_black(self):
        """old octree"""
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
        """old octree"""
        size = 6
        path = Path(Path(__file__).parent.parent, "test/alex_pico/emss-81481-f001.png")
        out_image = self.octree_and_show(path, size)

    def octree_and_show(self, path, size):
        """old octree"""
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)
        out_image, palette, palette_image = octree.quantize(img, size=size)
        print(f"image {type(out_image)} ... {out_image}")
        out_image.save(Path(path.parent, "test1.png"), "png")
        out_image.show()
        out_image.getcolors(maxcolors=256)
        return out_image

    def test_81481_octree_new(self):

        """Image.quantize(colors=256, method=None, kmeans=0, palette=None, dither=1)[source]
Convert the image to ‘P’ mode with the specified number of colors.

Parameters
colors – The desired number of colors, <= 256
method –
MEDIANCUT (median cut), MAXCOVERAGE (maximum coverage), FASTOCTREE (fast octree),
LIBIMAGEQUANT (libimagequant; check support using PIL.features.check_feature() with feature="libimagequant").
By default, MEDIANCUT will be used.
The exception to this is RGBA images. MEDIANCUT and MAXCOVERAGE do not support RGBA images,
so FASTOCTREE is used by default instead.
kmeans – Integer
palette – Quantize to the palette of given PIL.Image.Image.
dither – Dithering method, used when converting from mode “RGB” to “P” or from “RGB” or “L” to “1”.
Available methods are NONE or FLOYDSTEINBERG (default). Default: 1 (legacy setting)
Returns A new image
"""
        quantizer = Quantizer(input_dir=Path(self.get_py4ami_dir(), "test/alex_pico/"), root="emss-81481-f001")
        quantizer.extract_color_streams()

    def test_example_several_color_streams(self):
        roots = [
            "13068_2019_1355_Fig4_HTML",
            "fmicb-09-02460-g001",
            "pone.0054762.g004",
            "emss-81481-f001",
            ]
        quantizer = Quantizer(input_dir=Path(self.get_py4ami_dir(), "test/alex_pico/"), num_colors=16)
        for root in roots:
            quantizer.root = root
            quantizer.extract_color_streams()

    def test_green_battery(self):
        Quantizer(input_dir=Path(self.get_py4ami_dir(), "test/resources"), method="octree", root="green").extract_color_streams()

    def test_example_anuv_pathways(self):
        roots = [
            "biosynth_path_1",
            "biosynth_path_2",
            "biosynth_path_3",
            "biosynth_path_4",
            "biosynth_path_5",
            "biosynth_path_6",
            "biosynth_path_7",
            "biosynth_path_8",
        ]
        for root in roots:
            print(f"\n=====root: {root}=====")
            Quantizer(input_dir=Path(self.get_py4ami_dir(), "test/resources"), method="octree", root=root).extract_color_streams()

    # -------- Utility --------
    @classmethod
    def get_py4ami_dir(cls):
        return Path(__file__).parent.parent


"""
img_bytes = bytes([R1, G1, B1, R2, G2, B2,..., Rn, Gn, Bn])

im = Image.frombytes("RGB", (width, height), img_bytes)"""
