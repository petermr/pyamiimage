
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, color, io, data, draw
from skimage.exposure import histogram
import skimage.segmentation as seg
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
from pathlib import Path
from PIL import Image

# https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html

from skimage.morphology import medial_axis, skeletonize, thin

"""Code copied from earlier PMR ImageLib
being gradually converted into ImageProcessor
"""


class ImageLib:
    def __init__(self):
        self.image = None
        self.path = "assets/purple_ocimum_basilicum.png"
        # self.old_init()

    def old_init(self):
        # should not be run on init
        self.image = data.binary_blobs()
        self.image = io.imread('../../images/green.png')
        self.image = rgb2gray(self.image)
        self.image = 255 - self.image
        thresh = threshold_otsu(self.image)
        thresh = 40
        binary = self.image > thresh
        self.image = binary
        print("read image")

    def image_import(self, path=None):
        if path is None:
            path = self.path
        self.image = io.imread(path)

    @classmethod
    def image_show(cls, image, nrows=1, ncols=1, cmap='gray'):
        print("image shape", image.shape)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        return fig, ax

    def image_show_top(self):
        print("start image_show")

        self.text = data.page()
        print("text>>", self.text)
        self.image_show(self.text)

        fig, ax = plt.subplots(1, 1)
        ax.hist(self.text.ravel(), bins=32, range=[0, 256])
        ax.set_xlim(0, 256)

        text_segmented = self.text > 50
        self.image_show(text_segmented)

        text_segmented = self.text > 70
        self.image_show(text_segmented)

        text_segmented = self.text > 120
        self.image_show(text_segmented)

        text_threshold = filters.threshold_otsu(self.text)
        self.image_show(self.text > text_threshold)

        text_threshold = filters.threshold_li(self.text)
        self.image_show(self.text > text_threshold)

        text_threshold = filters.threshold_local(self.text, block_size=51, offset=10)
        self.image_show(self.text > text_threshold)
        print("end image_show")


class ImageExamples:

    def circle_points(self, resolution, center, radius):
        """
        Generate points defining a circle on an image.
        """
        radians = np.linspace(0, 2 * np.pi, resolution)

        c = center[1] + radius * np.cos(radians)
        r = center[0] + radius * np.sin(radians)

        return np.array([c, r]).T

    def blobs(self):
        print("start blobs")
        self.image = data.binary_blobs()
        plt.imshow(self.image, cmap='gray')
        io.imsave('../../outputs/misc1/blobs.png', self.image)

        #        self.image = data.astronaut()
        #        plt.imshow(image)

        self.image = io.imread('../../images/green.png')
        plt.imshow(self.image)
        io.imsave('../../outputs/misc1/green.png', self.image)

        #        images = io.ImageCollection('../images/*.png:../images/*.jpg')
        #        print('Type:', type(images))
        #        images.files
        #        Out[]: Type: <class ‘skimage.io.collection.ImageCollection’>

        io.imsave('logo.png', self.image)
        print("end blobs")

    @classmethod
    def supervised(cla):
        print("start supervised")

#        self.image = io.imread('girl.jpg')
        image = data.astronaut()

        plt.imshow(image)

        image_gray = color.rgb2gray(image)
        ImageLib.image_show(image_gray)

        # Exclude last point because a closed path should not have duplicate points
        points = ImageExamples.circle_points(200, [80, 250], 80)[:-1]

        fig, ax = ImageLib.image_show(image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

        snake = seg.active_contour(image_gray, points)
        fig, ax = ImageLib.image_show(image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)

        snake = seg.active_contour(image_gray, points, alpha=0.06, beta=0.3)
        fig, ax = ImageLib.image_show(image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)

        image_labels = np.zeros(image_gray.shape, dtype=np.uint8)
        indices = draw.circle_perimeter(80, 250, 20)  # from here
        image_labels[indices] = 1
        image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
        ImageLib.image_show(image_labels)

        image_segmented = seg.random_walker(image_gray, image_labels)
        # Check our results
        fig, ax = ImageLib.image_show(image_gray)
        ax.imshow(image_segmented == 1, alpha=0.3)

        image_segmented = seg.random_walker(image_gray, image_labels, beta=3000)
        # Check our results
        fig, ax = ImageLib.image_show(image_gray)
        ax.imshow(image_segmented == 1, alpha=0.3)
        print("end supervised")

    @classmethod
    def snake1(cls):

        img = data.astronaut()
        img = rgb2gray(img)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3),
                               init, alpha=0.015, beta=10, gamma=0.001)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap="Greys")
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()

    @classmethod
    def snake2(cls):
        img = data.text()

        r = np.linspace(136, 50, 100)
        c = np.linspace(5, 424, 100)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 1), init, boundary_condition='fixed',
                               alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img, cmap="Grays")
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()

# unsupervised

    @classmethod
    def unsupervised(cls, image):
        """
        requires self.image
        :return:
        """
        print("start unsupervised")
        image_slic = seg.slic(image, n_segments=155)
        ImageLib.image_show(color.label2rgb(image_slic, image, kind='avg'))

        image_felzenszwalb = seg.felzenszwalb(image)
        ImageLib.image_show(image_felzenszwalb)

        np.unique(image_felzenszwalb).size

        image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, image, kind='avg')
        ImageLib.image_show(image_felzenszwalb_colored)
        print("end unsupervised")

# thin
    @classmethod
    def thin(cls):
        print("start thin")

        from skimage.morphology import skeletonize
        from skimage.util import invert

        # Invert the horse image
        image = invert(data.horse())

        # perform skeletonization
        skeleton = skeletonize(image)

        # display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                 sharex=True, sharey=True)

        ax = axes.ravel()

        ax[0].imshow(image, cmap="Grays")
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(skeleton, cmap="Grays")
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()
        print("end thin")

    @classmethod
    def skel1(cls):
        print("start skel1")
        blobs = data.binary_blobs(200, blob_size_fraction=.2,
                                  volume_fraction=.35, seed=1)

        skeleton = skeletonize(blobs)
        skeleton_lee = skeletonize(blobs, method='lee')

        fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(blobs, cmap="Grays")
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(skeleton, cmap="Grays")
        ax[1].set_title('skeletonize')
        ax[1].axis('off')

        ax[2].imshow(skeleton_lee, cmap="Grays")
        ax[2].set_title('skeletonize (Lee 94)')
        ax[2].axis('off')

        fig.tight_layout()
        plt.show()
        print("end skel1")

    @classmethod
    def skel2(cls, image):
        print("start skel2")
        skeleton = skeletonize(image)
        thinned = thin(image)
        thinned_partial = thin(image, max_iter=25)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap="Grays")
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(skeleton, cmap="Grays")
        ax[1].set_title('skeleton')
        ax[1].axis('off')

        ax[2].imshow(thinned, cmap="Grays")
        ax[2].set_title('thinned')
        ax[2].axis('off')

        ax[3].imshow(thinned_partial, cmap="Grays")
        ax[3].set_title('partially thinned')
        ax[3].axis('off')

        fig.tight_layout()
        plt.show()
        print("end skel2")

    @classmethod
    def segment(cls):
        print("start segment")
        coins = data.coins()
        hist, hist_centers = histogram(coins)
        print("end segment")

    @classmethod
    def medial(cls):

        # Generate the data
        img = data.binary_blobs(200, blob_size_fraction=.2,
                                volume_fraction=.35, seed=1)
        """
        img = io.imread('../../images/PMC5453356.png')
        img = img > 20
        """

        img = io.imread('../../images/green.png')
        img = rgb2gray(img)
        img = 1 - img
        print(sum(img), img)
#        img = img < 128

        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = medial_axis(img, return_distance=True)

        # Compare with other skeletonization algorithms
        skeleton = skeletonize(img)
        skeleton_lee = skeletonize(img, method='lee')

        # Distance to the background for pixels of the skeleton
        dist_on_skel = distance * skel
        print("dist on skel", dist_on_skel.shape, dist_on_skel)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img, cmap="Grays")
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(dist_on_skel, cmap='magma')
        ax[1].contour(img, [0.5], colors='w')
        ax[1].set_title('medial_axis')
        ax[1].axis('off')

        dist_on_skel = dist_on_skel > 5.0

        ax[2].imshow(skeleton, cmap="Grays")
        ax[2].set_title('skeletonize')
        ax[2].axis('off')

        """
        ax[3].imshow(skeleton_lee, cmap="Grays")
        ax[3].set_title("skeletonize (Lee 94)")
        ax[3].axis('off')
        """
        ax[3].imshow(dist_on_skel, cmap="Grays")
        ax[3].set_title("thick lines")
        ax[3].axis('off')

#        fig.tight_layout()
        plt.show()


class Quantizer:
    """colour quantizer for pixel images

    a tortous journey to flattem images to a small set of colours
    finally arriving at FASTOCTREE and convert()
    """
    OCTREE = "octree"

    def __init__(self, input_dir, root=None, num_colors=8, method=None):
        """   """
        assert input_dir is not None and input_dir.exists(), "input dir must be an existing directory"
        self.input_dir = input_dir
        self.root = root
        self.num_colors = num_colors
        self.method = method
        self.palette_dict = None

    def create_and_write_color_streams(self, pil_img, num_colors=8, out_dir=None, out_form="png", out_root=None,
                                       method=OCTREE, kmeans=8, dither=None):
        """
        Separates colours and flattens them.
        The default method does this by histogram (I think). It is terrible for JPEGs
        The octree separates by colour distance. recommended

        :param pil_img:
        :param num_colors:
        :param out_dir:
        :param out_form:
        :param out_root:
        :param method: if none uses octree
                        else if method == "octree" uses PIL.Image.FASTOCTREE
                        else uses given method

        :param kmeans: default 8
        :param dither: used in quantize, def = None, option  PIL.Image.FLOYDSTEINBERG

        :return:
        """
        if method is not None:
            self.method = method
        if self.method:
            if self.method == self.OCTREE:
                self.method = Image.FASTOCTREE
            img_out = pil_img.quantize(colors=self.num_colors, method=self.method, kmeans=kmeans, dither=dither)
        else:
            img_out = pil_img.convert('P', palette=Image.ADAPTIVE, colors=self.num_colors)
        print(f"\nform {img_out.format}, size {img_out.size}, mode {img_out.mode}")
        img_out.save(Path(out_dir, "palette"+"." + out_form), out_form)
        img_rgb = img_out.convert("RGB")
        img_rgb.save(Path(out_dir, "rgb"+"." + out_form), out_form)
        # [146 209  80]
        rgb_array = np.array(img_rgb)
        single_chan = self.replace_single_color(rgb_array,
                                                old_col=[146, 209, 80],
                                                new_col=[255, 0, 0],
                                                back_col=[220, 255, 255])
        plt.imsave(Path(out_dir, "single" + "." + out_form), single_chan)
        self.palette_dict = self.create_palette(img_out)
        print("palette", self.palette_dict)
        self.create_monochrome_images_of_color_streams(np.array(img_out), out_dir, out_form)
        image_by_hx = self.create_monochrome_images_from_rgb(np.array(img_out))
        print("image by hex", image_by_hx)

    def create_palette(self, img):
        """
        extract palette for "P" image
        index on hexstring
        :param img:
        :return: dict of counts with 6-char hex index
        """
        palette_dict = {}
        palette = img.getpalette()
        rgb_palette = np.reshape(palette, (-1, 3))
        count_rgb_list = img.getcolors(self.num_colors)
        print(f"colours {len(count_rgb_list)}")  # ca 48 non-zer0
        for count_rgb in count_rgb_list:
            rgb = rgb_palette[count_rgb[1]]
            hx = rgb2hex(rgb)
            # print(f"{count_rgb[0]} {hx} {rgb}")
            count = count_rgb[0]
            if count != 0:
                palette_dict[hx] = count
        return palette_dict

    def replace_single_color(self, rgb_array, old_col, new_col, back_col=None):
        if back_col is None:
            back_col = [0., 0., 0.]
        single_chan = np.where(rgb_array == old_col, new_col, back_col)
        single_chan = np.multiply(single_chan, 1.0 / 255.)
        return single_chan

    def create_monochrome_images_of_color_streams(self, img_array, out_dir, out_form="png"):
        for palette_index in range(self.num_colors):
            if out_dir:
                out_path = Path(out_dir, "p" + str(palette_index) + "." + out_form)
                # img1 = np.where(img_array == color, True, False)
                img1 = np.where(img_array == palette_index, palette_index, 254)
                plt.imsave(out_path, img1)

    def create_monochrome_images_from_rgb(self, rgb_array, back_col=None):
        if back_col is None:
            back_col = [0., 110., 220., ]
        new_array_dict = {}
        print("RGB ", rgb_array.shape)
        for hex_col in self.palette_dict:
            rgb = hex2rgb(hex_col)
            rgbx = [float(rgb[0]), float(rgb[1]), float(rgb[2])]
            new_array = None
            # new_array = np.where(int(rgb_array) == rgb, rgbx, back_col)
            # print("NP COUNT", np.count_nonzero(new_array))
            # print ("rgb shape...", new_array.shape)
            new_array_dict[rgb2hex(rgb)] = new_array
        return new_array_dict

    def extract_color_streams(self):
        in_path = None
        suffixes = ["png", "jpeg", "jpg"]
        for suffix in suffixes:
            in_path = Path(self.input_dir, self.root + "." + suffix)
            if in_path.exists():
                break
        if in_path is None:
            print(f"cannot find images with root {self.root}")
            return
        out_dir = self.make_out_dir(self.input_dir, self.root)
        img = Image.open(in_path)
        self.create_and_write_color_streams(img, num_colors=8, out_dir=out_dir)

    def make_out_dir(self, in_dir, root):
        out_root = Path(in_dir, root)
        if not out_root.exists():
            out_root.mkdir()
        return out_root


def rgb2hex(rgb):
    """convert rgb 3-array to 8 char hex string
    :param rgb: 3-array of ints
    :return: "hhhhhh" string does NOT prepend "0x"
    """
    assert len(rgb) == 3
    # assert type(rgb[0]) is int, f"found {type(rgb[0])} {rgb[0]}, in rgb"
    assert 0 <= rgb[0] <= 255, f"found {rgb[0]}, in rgb"
    s = ""
    for r in rgb:
        h = hex(r)[2:] if r >= 16 else "0" + hex(r)[2:]
        s += h
    return s


def hex2rgb(hx):
    """
    transform 6-digit hex number into [r,g,b] integers

    :param hx:
    :return:
    """
    assert len(hx) == 6
    rgb = []
    for r in range(3):
        ss = "0x" + hx[2 * r: 2 * r + 2]
        rr = int(ss, 16)
        rgb.append(rr)
    return rgb


def main():
    print("started image_lib")
    image_lib = ImageLib()
    segment = False
    blobs = False
    show_top = False
    supervised = False
    snake1 = False
    snake2 = False
    unsupervised = False
    thin = False
    skel1 = False
    skel2 = False
    medial = True

    if segment:
        ImageExamples.segment()
    if blobs:
        ImageExamples.blobs()
    if show_top:
        ImageExamples.image_show_top()
    if supervised:
        ImageExamples.supervised()
    if snake1:
        ImageExamples.snake1()
    if snake2:
        ImageExamples.snake2()
    if unsupervised:
        ImageExamples.unsupervised()
    if thin:
        ImageExamples.thin()
    if skel1:
        ImageExamples.skel1()
    if skel2:
        ImageExamples.skel2()
    if medial:
        ImageExamples.medial()

    """
    while True:
        print(".")
        time.sleep(1.0)
        return
    """

    print("END")

    print("finished image_lib")


if __name__ == "__main__":
    main()
