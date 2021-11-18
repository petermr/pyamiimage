from pathlib import Path
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

        self.octree_and_show(path, size)

    def octree_and_show(self, path, size):
        assert path.exists()
        # img = imageio.imread(path)
        img = Image.open(path)
        out_image, palette, palette_image = octree.quantize(img, size=size)
        print(f"image {type(out_image)} ... {out_image}")
        out_image.show()
