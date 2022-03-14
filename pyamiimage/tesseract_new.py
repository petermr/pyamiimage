import numpy as np
from pathlib import Path
from matplotlib.image import imsave
from pyamiimage.pyimage.bbox import BBox
from pyamiimage.test.resources import Resources
from pyamiimage.pyimage.ami_ocr import AmiOCR
from pyamiimage.pyimage.ami_image import AmiImage
from matplotlib import pyplot as plt
from skimage import io
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# tess = Resources.TESSERACT1
# tess = Resources.TESSERACT_BENG
tess = Resources.BIOSYNTH2
tess = Resources.YW5003_5

# tess = Resources.TESSERACT_GER2
# tess = Resources.TESSERACT_ITA
tess_ocr = AmiOCR(tess)
tess_img = io.imread(tess)

med_xrd = Resources.MED_XRD_FIG_A
med_xrd_img = io.imread(med_xrd)





# phrases = tess_ocr.get_phrases()
# phrases_with_bounding_boxes = tess_ocr.plot_bboxes_on_image(tess_img, phrases)

# for phrase in phrases:
#     print(phrase)

# io.imshow(phrases_with_bounding_boxes)
# io.show()


tess_img_gray = AmiImage.create_grayscale_from_image(tess_img)
tess_img_bin = AmiImage.create_white_binary_from_image(tess_img_gray)
tess_ocr = AmiOCR(image=tess_img_gray)
# io.imshow(tess_img_bin)
# io.show()

# words = tess_ocr.get_words()
# words_with_baseline  = tess_ocr.find_baseline(tess_ocr.hocr)
# for word in words:
#     re_read_word = AmiOCR.read_textbox(tess_img_gray, word)
#     print(f"word: {word}")

# for word in words:
#     re_read_word = AmiOCR.read_textbox(tess_img_gray, word)
#     print(f"word: {word}; re_read word: {re_read_word}")

# words_with_bounding_boxes = AmiOCR.plot_bboxes_on_image(tess_img_gray, words)
# io.imshow(words_with_bounding_boxes)
# io.show()
# box_bin = AmiOCR.plot_bboxes_on_image(tess_img_bin, words)
# io.imshow(box_bin)
# io.show()

# # # patches = AmiOCR.bounding_box_patches(tess_img_bin, words)
# # # TESSERACT_TEMP_PATH = Path(Path(__file__).parent.parent, "temp/tesseract/")
# # # AmiImage.write_image_group(TESSERACT_TEMP_PATH, patches, filename="black_on_white", mkdir=False)

# def patches_pixel_stats(patches, signal_val, axis=1):
#     """sum of signal values along a axis in an array"""
#     signal = []
#     for patch in patches:
#         signal.append(np.count_nonzero(patch==signal_val, axis=axis))
#     signal = [np.array(item) for item in signal]
#     return signal


y_range = [26, 389]
x_range = [80, 386]
box = BBox([y_range, x_range]) #from peter
# med_xrd_plot_rm = AmiOCR.set_bbox_to_bg(med_xrd_img, box)
med_xrd_bin = AmiImage.create_white_binary_from_image(med_xrd_img)
img_height = med_xrd_img.shape[0]
img_width = med_xrd_img.shape[0]
# Very rash code, uses a lot of assumptions, works for a very specific case
x_label = med_xrd_img[y_range[1]:img_height, x_range[0]: img_width]
y_label = med_xrd_img[0:y_range[1], 0: x_range[0]]
angle = 270 # in degrees
y_label_rot90 = ndimage.rotate(y_label, angle, reshape=True)

print("this is it")
AmiOCR.plot_image_pixel_stats(med_xrd_img, 255, axis=1)

# io.imshow(x_label)
# io.show()
io.imshow(y_label_rot90)
io.show()

x_label_ocr = AmiOCR(image=x_label)
words = x_label_ocr.get_words()
for word in words:
    print(word)

y_label_ocr = AmiOCR(image=y_label_rot90)
words = y_label_ocr.get_words()
for word in words:
    print(word)

y_plot_bbox = AmiOCR.plot_bboxes_on_image(y_label_rot90, words)
io.imshow(y_plot_bbox)
io.show()

def image_pixel_stats(image, signal_val, axis=1):
    """sum of signal values along a axis in an array"""
    signal = np.count_nonzero(image==signal_val, axis=axis)
    signal = np.array(signal)
    return signal

vertical_signal = image_pixel_stats(med_xrd_bin, 255, axis=1)


angle = 90 # in degrees

new_data = ndimage.rotate(med_xrd_img, angle, reshape=True)
row = np.arange(0, med_xrd_img.shape[0])
plt.imshow(new_data)

plt.plot(row, vertical_signal, color='red')

plt.show()


horizontal_signal = image_pixel_stats(med_xrd_bin, 255, axis=0)

row = np.arange(0, med_xrd_img.shape[1])
plt.imshow(med_xrd_img)

plt.plot(row, horizontal_signal, color='red')

plt.show()






# signal = patches_pixel_stats(patches, signal_val=255, axis=1)

# # # row = [np.arange(0, item.shape[0]) for item in signal]
# # # import matplotlib.pyplot as plt

# # # for idx in range(10, 11):
# # #     im = plt.imread(Path(TESSERACT_TEMP_PATH, f"black_on_white{idx}.png"))
# # #     # implot = plt.imshow(im)
# # #     import scipy.ndimage as ndimage

# # #     angle = 90 # in degrees

# # #     new_data = ndimage.rotate(im, angle, reshape=True)

# # #     plt.imshow(new_data)

# # #     plt.plot(row[idx], signal[idx], color='red')

# # #     plt.show()

    
# for x, y in zip(row, signal):
#     plt.plot(x, y)
#     plt.show()

# AmiOCR.pretty_print_hocr(tess_ocr.hocr)
# print(words_with_baseline[0].baseline)
