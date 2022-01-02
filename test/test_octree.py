# library imports
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
# local imports
from ..pyimage import octree
from ..pyimage.old_code.image_lib import Quantizer

interactive = False

PYAMI_DIR = Path(__file__).parent.parent
TEST_DIR = Path(PYAMI_DIR, "test")
PICO_DIR = Path(TEST_DIR, "alex_pico/")
RESOURCES_DIR = Path(TEST_DIR, "resources")


class TestOctree:

    def setup_method(self, method=None):
        pass

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
        if interactive:
            out_image.show()

    def test_81481_diagram(self):
        """old octree"""
        size = 6
        path = Path(PICO_DIR, "emss-81481-f001.png")
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)
        pil_rgb_image, palette, palette_image = octree.quantize(img, size=size)
        print(f"\npalette {type(palette)}  {palette}")
        assert type(pil_rgb_image) is PIL.Image.Image
        nparray = np.asarray(pil_rgb_image)
        assert nparray.shape == (555, 572, 3)
        print(f"image {type(pil_rgb_image)} ... {pil_rgb_image}")
        print(f"palette image {type(pil_rgb_image)}")
        palette_array = np.asarray(pil_rgb_image)
        assert palette_array.shape == (555, 572, 3)
        pil_rgb_image.save(Path(path.parent, "test1.png"), "png")
        if interactive:
            pil_rgb_image.show()
        pil_rgb_image.getcolors(maxcolors=256)

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
        quantizer = Quantizer(input_dir=PICO_DIR, root="emss-81481-f001")
        quantizer.extract_color_streams()

    def test_example_several_color_streams(self):
        """not yet useful tests"""
        roots = [
            "13068_2019_1355_Fig4_HTML",
            # "fmicb-09-02460-g001",
            # "pone.0054762.g004",
            # "emss-81481-f001",
            ]
        quantizer = Quantizer(input_dir=PICO_DIR, num_colors=16)
        for root in roots:
            quantizer.root = root
            quantizer.extract_color_streams()

    def test_green_battery(self):
        Quantizer(input_dir=RESOURCES_DIR, method="octree",
                  root="green").extract_color_streams()

    def test_example_anuv_pathways(self):
        """not yet useful tests"""
        roots = [
            "biosynth_path_1",
            # "biosynth_path_2",
            # "biosynth_path_3",
            # "biosynth_path_4",
            # "biosynth_path_5",
            # "biosynth_path_6",
            # "biosynth_path_7",
            # "biosynth_path_8",
        ]
        for root in roots:
            print(f"\n=====root: {root}=====")
            Quantizer(input_dir=RESOURCES_DIR, method="octree",
                      root=root).extract_color_streams()

    # -------- Utility --------
    @classmethod
    def get_py4ami_dir(cls):
        return Path(__file__).parent.parent


"""
img_bytes = bytes([R1, G1, B1, R2, G2, B2,..., Rn, Gn, Bn])

im = Image.frombytes("RGB", (width, height), img_bytes)"""
