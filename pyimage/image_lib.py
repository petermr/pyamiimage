
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
class ImageLib():
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
        plt.imshow(self.image);
        io.imsave('../../outputs/misc1/green.png', self.image)

#        images = io.ImageCollection('../images/*.png:../images/*.jpg')
#        print('Type:', type(images))
#        images.files
#        Out[]: Type: <class ‘skimage.io.collection.ImageCollection’>

        io.imsave('logo.png', self.image)
        print("end blobs")

    def image_show_top(self):
        print("start image_show")

        self.text = data.page()
        print("text>>", self.text)
        self.image_show(self.text)

        fig, ax = plt.subplots(1, 1)
        ax.hist(self.text.ravel(), bins=32, range=[0, 256])
        ax.set_xlim(0, 256);

        text_segmented = self.text > 50
        self.image_show(text_segmented);

        text_segmented = self.text>70
        self.image_show(text_segmented);

        text_segmented = self.text>120
        self.image_show(text_segmented);

        text_threshold = filters.threshold_otsu(self.text)  # Hit tab with the cursor after the underscore, try several methods
        self.image_show(self.text > text_threshold);

        text_threshold = filters.threshold_li(self.text)  # Hit tab with the cursor after the underscore, try several methods
        self.image_show(self.text > text_threshold);

        text_threshold = filters.threshold_local(self.text, block_size=51, offset=10)
        self.image_show(self.text > text_threshold);
        print("end image_show")

    def supervised(self):
        print("start supervised")

#        self.image = io.imread('girl.jpg')
        self.image = data.astronaut()

        plt.imshow(self.image);

        self.image_gray = color.rgb2gray(self.image)
        self.image_show(self.image_gray);


        # Exclude last point because a closed path should not have duplicate points
        points = self.circle_points(200, [80, 250], 80)[:-1]

        fig, ax = self.image_show(self.image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)

        snake = seg.active_contour(self.image_gray, points)
        fig, ax = self.image_show(self.image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

        snake = seg.active_contour(self.image_gray, points, alpha=0.06, beta=0.3)
        fig, ax = self.image_show(self.image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

        image_labels = np.zeros(self.image_gray.shape, dtype=np.uint8)
        indices = draw.circle_perimeter(80, 250, 20)  # from here
        image_labels[indices] = 1
        image_labels[points[:, 1].astype(np.int), points[:, 0].astype(np.int)] = 2
        self.image_show(image_labels);

        image_segmented = seg.random_walker(self.image_gray, image_labels)
        # Check our results
        fig, ax = self.image_show(self.image_gray)
        ax.imshow(image_segmented == 1, alpha=0.3);

        image_segmented = seg.random_walker(self.image_gray, image_labels, beta=3000)
        # Check our results
        fig, ax = self.image_show(self.image_gray)
        ax.imshow(image_segmented == 1, alpha=0.3);
        print("end supervised")

    def snake1(self):

        img = data.astronaut()
        img = rgb2gray(img)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3),
                               init, alpha=0.015, beta=10, gamma=0.001)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()

    def snake2(self):
        img = data.text()

        r = np.linspace(136, 50, 100)
        c = np.linspace(5, 424, 100)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 1), init, boundary_condition='fixed',
                               alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()
# unsupervised
    def unsupervised(self):
        print("start unsupervised")
        image_slic = seg.slic(self.image,n_segments=155)
        self.image_show(color.label2rgb(image_slic, self.image, kind='avg'));

        image_felzenszwalb = seg.felzenszwalb(self.image)
        self.image_show(image_felzenszwalb);

        np.unique(image_felzenszwalb).size

        image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, self.image, kind='avg')
        self.image_show(image_felzenszwalb_colored);
        print("end unsupervised")

#thin
    def thin(self):
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

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()
        print ("end thin")


    def skel1(self):
        print ("start skel1")
        blobs = data.binary_blobs(200, blob_size_fraction=.2,
                                  volume_fraction=.35, seed=1)

        skeleton = skeletonize(blobs)
        skeleton_lee = skeletonize(blobs, method='lee')

        fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(blobs, cmap=plt.cm.gray)
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].set_title('skeletonize')
        ax[1].axis('off')

        ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
        ax[2].set_title('skeletonize (Lee 94)')
        ax[2].axis('off')

        fig.tight_layout()
        plt.show()
        print ("end skel1")

    def skel2(self):
        print ("start skel2")
        skeleton = skeletonize(self.image)
        thinned = thin(self.image)
        thinned_partial = thin(self.image, max_iter=25)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(self.image, cmap=plt.cm.gray)
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(skeleton, cmap=plt.cm.gray)
        ax[1].set_title('skeleton')
        ax[1].axis('off')

        ax[2].imshow(thinned, cmap=plt.cm.gray)
        ax[2].set_title('thinned')
        ax[2].axis('off')

        ax[3].imshow(thinned_partial, cmap=plt.cm.gray)
        ax[3].set_title('partially thinned')
        ax[3].axis('off')

        fig.tight_layout()
        plt.show()
        print ("end skel2")

    def segment(self):
        print("start segment")
        coins = data.coins()
        hist, hist_centers = histogram(coins)
        print("end segment")

    def medial(self):
        from skimage.morphology import medial_axis, skeletonize

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

        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('original')
        ax[0].axis('off')

        ax[1].imshow(dist_on_skel, cmap='magma')
        ax[1].contour(img, [0.5], colors='w')
        ax[1].set_title('medial_axis')
        ax[1].axis('off')

        dist_on_skel = dist_on_skel > 5.0


        ax[2].imshow(skeleton, cmap=plt.cm.gray)
        ax[2].set_title('skeletonize')
        ax[2].axis('off')

        """
        ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
        ax[3].set_title("skeletonize (Lee 94)")
        ax[3].axis('off')
        """
        ax[3].imshow(dist_on_skel, cmap=plt.cm.gray)
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
        img_out.save(Path(out_dir, "palette_rgb"+"." + out_form), out_form)
        palette = img_out.getpalette()
        palette_image = np.array(img_out)
        rgb_palette = np.reshape(palette, (-1, 3))
        count_rgb_list = img_out.getcolors(self.num_colors)
        print(f"colours {len(count_rgb_list)}")  # ca 48 non-zer0
        for count_rgb in count_rgb_list:
            rgb = rgb_palette[count_rgb[1]]
            hx = rgb2hex(rgb)
            print(f"{count_rgb[0]} {hx} {rgb_palette[count_rgb[1]]}")

        self.create_monochrome_images_of_color_streams( palette_image, out_dir, out_form)

    def create_monochrome_images_of_color_streams(self, img_array, out_dir, out_form="png"):
        for color in range(self.num_colors):
            if out_dir:
                out_path = Path(out_dir, "p" + str(color) + "." + out_form)
                img1 = np.where(img_array == color, True, False)
                img1 = np.where(img_array == color, color, 254)

                plt.imsave(out_path, img1)
                # arry = img1.tolist()
                # print(arry)
                # img_pil = Image.fromarray(arry)
                # img_pil = Image.fromarray(img_array)
                # img_pil.save(out_path)
                # img_rgb = img_pil.convert('RGB')
                # img2 = np.where(img1 != 254, np, 0)
                # print(f"non zero {np.count_nonzero(img2)}")
                # print(f"{color} {img1}")
                # print(f"out_path {out_path}")
                # img_rgb.save(out_path)


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
    assert type(rgb[0]) is np.int64, f"found {type(rgb[0])} {rgb[0]}, in rgb"
    assert rgb[0] >= 0 and rgb[0] <= 255, f"found {rgb[0]}, in rgb"
    s = ""
    for r in rgb:
        h = hex(r)[2:] if r >= 16 else "0" + hex(r)[2:]
        s += h
    return s

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
    medial =  True

    if segment:
        image_lib.segment()
    if blobs:
        image_lib.blobs()
    if show_top:
        image_lib.image_show_top()
    if supervised:
        image_lib.supervised()
    if snake1:
        image_lib.snake1()
    if snake2:
        image_lib.snake2()
    if unsupervised:
        image_lib.unsupervised()
    if thin:
        image_lib.thin()
    if skel1:
        image_lib.skel1()
    if skel2:
        image_lib.skel2()
    if medial:
        image_lib.medial()

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
