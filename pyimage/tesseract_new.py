import numpy as np
from pathlib import Path
from matplotlib.image import imsave
from pyamiimage.test.resources import Resources
from pyamiimage.pyimage.ami_ocr import AmiOCR
from pyamiimage.pyimage.ami_image import AmiImage
from matplotlib import pyplot as plt
from skimage import io
# tess = Resources.TESSERACT1
# tess = Resources.TESSERACT_BENG
tess = Resources.BIOSYNTH2
# tess = Resources.TESSERACT_GER2
# tess = Resources.TESSERACT_ITA
tess_ocr = AmiOCR(tess)
tess_img = io.imread(tess)



# words_with_bounding_boxes = tess_ocr.plot_bboxes_on_image(tess_img, words)
# io.imshow(words_with_bounding_boxes)
# io.show()

# phrases = tess_ocr.get_phrases()
# phrases_with_bounding_boxes = tess_ocr.plot_bboxes_on_image(tess_img, phrases)

# for phrase in phrases:
#     print(phrase)

# io.imshow(phrases_with_bounding_boxes)
# io.show()


tess_img_gray = AmiImage.create_grayscale_from_image(tess_img)
tess_img_bin = AmiImage.create_white_binary_from_image(tess_img_gray)
tess_ocr = AmiOCR(image=tess_img_bin)
# io.imshow(tess_img_bin)
# io.show()

words = tess_ocr.get_words()
words_with_baseline  = tess_ocr.find_baseline(tess_ocr.hocr)
for word in words:
    print(word)

# box_bin = AmiOCR.plot_bboxes_on_image(tess_img_bin, words)
# io.imshow(box_bin)
# io.show()

patches = AmiOCR.bounding_box_patches(tess_img_bin, words)
TESSERACT_TEMP_PATH = Path(Path(__file__).parent.parent, "temp/tesseract/")
AmiImage.write_image_group(TESSERACT_TEMP_PATH, patches, filename="black_on_white", mkdir=False)

def patches_pixel_stats(patches, signal_val, axis=1):
    """sum of signal values along a axis in an array"""
    signal = []
    for patch in patches:
        signal.append(np.count_nonzero(patch==signal_val, axis=axis))
    signal = [np.array(item) for item in signal]
    return signal

signal = patches_pixel_stats(patches, signal_val=255, axis=1)

row = [np.arange(0, item.shape[0]) for item in signal]
import matplotlib.pyplot as plt

for idx in range(36, 48):
    im = plt.imread(Path(TESSERACT_TEMP_PATH, f"black_on_white{idx}.png"))
    # implot = plt.imshow(im)
    import scipy.ndimage as ndimage

    angle = 90 # in degrees

    new_data = ndimage.rotate(im, angle, reshape=True)

    plt.imshow(new_data)

    plt.plot(row[idx], signal[idx], color='red')

    plt.show()
# for x, y in zip(row, signal):
#     plt.plot(x, y)
#     plt.show()

# AmiOCR.pretty_print_hocr(tess_ocr.hocr)
# print(words_with_baseline[0].baseline)
