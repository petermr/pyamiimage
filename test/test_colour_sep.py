# library imports
import unittest
from pathlib import Path
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import PIL
from PIL import Image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

# local imports
import context
from pyamiimage import octree
from pyamiimage._old_image_lib import Quantizer
from pyamiimage.ami_image import AmiImage
from pyamiimage.ami_util import AmiUtil

from resources import Resources

interactive = False
#interactive = True

PYAMI_DIR = Path(__file__).parent.parent
TEST_DIR = Path(PYAMI_DIR, "test")
PICO_DIR = Path(TEST_DIR, "alex_pico/")
RESOURCES_DIR = Path(TEST_DIR, "resources")

LONG_TEST = False

"""Tests both Octree and kmeans color separation"""
class TestOctree:
    def setup_method(self, method=None):
        pass

    def test_reading_red_black(self):
        """old octree"""
        image_name = "red_black_cv.png"
        # image_name = "purple_ocimum_basilicum.png"

        path = Path(Path(__file__).parent.parent, "assets", image_name)
        assert path.exists()
        # img = imageio.imread(path)
        pil_img = Image.open(path)  # PIL
        # img = io.imread(path)

        # ImageLib.image_show(img)
        # print(img)

        out_image, palette, palette_image = octree.quantize(pil_img, size=4)
        # print(f"image {type(out_image)} ... {out_image}")
        assert type(out_image) == PIL.Image.Image
        assert out_image.size == (850, 641)
        assert type(palette) == list
        print(f"palette {len(palette)}, {palette[0]}")
        if interactive:
            out_image.show()

    def test_81481_octree_quantize(self):
        """old octree"""
        size = 6
        path = Path(PICO_DIR, "emss-81481-f001.png")
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)
        pil_rgb_image, palette, palette_image = octree.quantize(img, size=size)
        print(f"\npalette {type(palette)}  {palette}")
        assert type(palette) is list, f"type palette {type(palette)}"
        assert len(palette) == 36
        assert type(pil_rgb_image) is PIL.Image.Image
        nparray = np.asarray(pil_rgb_image)
        assert nparray.shape == (555, 572, 3)
        print(f"image {type(pil_rgb_image)} ... {pil_rgb_image}")
        print(f"palette image {type(pil_rgb_image)}")
        palette_array = np.asarray(pil_rgb_image)
        assert palette_array.shape == (555, 572, 3)
        path = Path(Resources.TEMP_DIR, "test1.png")
        print(f"path {path}")
        pil_rgb_image.save(path, "png")
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
        Returns A new image"""
        quantizer = Quantizer(input_dir=PICO_DIR, root="emss-81481-f001")
        stream  = quantizer.extract_color_streams()
        print(f"stream {stream}")

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
            stream = quantizer.extract_color_streams()
            print(f"stream {stream}")

    def test_green_battery(self):
        streams = Quantizer(
            input_dir=Resources.BATTERY_DIR, method="octree", root="green"
        ).extract_color_streams()
        print(f"streams {streams}")

    def test_skimage(self):
        # Authors: Robert Layton <robertlayton@gmail.com>
        #          Olivier Grisel <olivier.grisel@ensta.org>
        #          Mathieu Blondel <mathieu@mblondel.org>
        #
        # License: BSD 3 clause

        """
        see https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin
        from sklearn.datasets import load_sample_image
        from sklearn.utils import shuffle
        from time import time

        n_colors = 64
        n_colors = 8

        # Load the Summer Palace photo
        china = load_sample_image("china.jpg")

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        china = np.array(china, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(china.shape)
        assert d == 3
        image_array = np.reshape(china, (w * h, d))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print(f"done in {time() - t0:0.3f}s.")

        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        print(f"done in {time() - t0:0.3f}s.")

        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        print("Predicting color indices on the full image (random)")
        t0 = time()
        labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
        print(f"done in {time() - t0:0.3f}s.")

        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            return codebook[labels].reshape(w, h, -1)

        # Display all results, alongside original image
        plt.figure(1)
        plt.clf()
        plt.axis("off")
        plt.title("Original image (96,615 colors)")
        plt.imshow(china)

        plt.figure(2)
        plt.clf()
        plt.axis("off")
        plt.title(f"Quantized image ({n_colors} colors, K-Means)")
        image_q = recreate_image(kmeans.cluster_centers_, labels, w, h)
        plt.imshow(image_q)

        plt.figure(3)
        plt.clf()
        plt.axis("off")
        plt.title(f"Quantized image ({n_colors} colors, Random)")
        image_qq = recreate_image(codebook_random, labels_random, w, h)
        plt.imshow(image_qq)
        if interactive:
            plt.show()

    def test_skimage1(self):

        n_colors = 4
        path = Path(Resources.BATTERY_DIR, "green.png")
        img0 = io.imread(path)
        img = np.array(img0, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img.shape)
        image_array = np.reshape(img, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

        plt.figure(1)
        plt.clf()
        plt.axis("off")
        plt.title("Original image (96,615 colors)")
        plt.imshow(img)

        plt.figure(2)
        plt.clf()
        plt.axis("off")
        plt.title(f"Quantized image ({n_colors} colors, K-Means)")
        image_q = kmeans.cluster_centers_[labels].reshape(w, h, -1)
        plt.imshow(image_q)

        plt.figure(3)
        plt.clf()
        plt.axis("off")
        plt.title(f"Quantized image ({n_colors} colors, Random)")
        image_qq = codebook_random[labels_random].reshape(w, h, -1)
        plt.imshow(image_qq)
        if interactive:
            plt.show()

    def test_kmeans2(self):
        """https://stackoverflow.com/questions/48222977/python-converting-an-image-to-use-less-colors"""
        import numpy as np
        from skimage import io
        from sklearn.cluster import KMeans


        expected = [[222, 227, 219], [16, 16, 17], [95, 96, 97], [253, 149, 75], [1, 235, 29], [254, 254, 254],
                    [20, 15, 196], [162, 187, 176], [240, 17, 23], [0, 118, 19]]

        background = [255, 255, 255]
        name = "green"
        self._means_test_write(name, expected=expected)

    @unittest.skipUnless(LONG_TEST, "takes too long")
    def test_kmeans_long(self):

        for name in [
            "pmc8839570",
            "MED_34909142_3",
            "Signal_transduction_pathways_wp",
            "red_black_cv",
            "prisma"
        ]:
            print(f"======{name}======")
            self._means_test_write(name, background=[255,255, 200], ncolors=10)

    def _means_test_write(self, name, background=[255, 255, 255], expected=None, ncolors=10):
        """skimage kmeans"""
        path = Path(Resources.TEST_RESOURCE_DIR, name + ".png")
        if not path.exists():
            path = Path(Resources.TEST_RESOURCE_DIR, name + ".jpeg")
        raw_image = io.imread(path)
        color_delta = 20
        labels, color_centers, quantized_images = AmiImage.kmeans(raw_image, ncolors, background)
        print(color_centers)
        for i, color in enumerate(color_centers):
            if AmiUtil.is_white(color, color_delta):
                print(f"white: {color}")
            elif AmiUtil.is_black(color, color_delta):
                print(f"black: {color}")
            elif AmiUtil.is_gray(color, color_delta):
                print(f"gray: {color}")
                continue
            hexs = ''.join(AmiUtil.int2hex(c)[-2:-1] for c in color)
            if expected:
                assert color == expected[i], f"color_centers {color}"
            dir_path = Path(Resources.TEMP_DIR, name)
            if not dir_path.exists():
                dir_path.mkdir()
            path = Path(dir_path, f"kmeans_{i}_{hexs}.png")
            io.imsave(path, quantized_images[i])

    # -------- Utility --------
    @classmethod
    def get_py4ami_dir(cls):
        return Path(__file__).parent.parent


"""
img_bytes = bytes([R1, G1, B1, R2, G2, B2,..., Rn, Gn, Bn])

im = Image.frombytes("RGB", (width, height), img_bytes)"""
