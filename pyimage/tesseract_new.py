from pyamiimage.test.resources import Resources
from pyamiimage.pyimage.ami_ocr import AmiOCR
from pyamiimage.pyimage.ami_image import AmiImage

from skimage import io
tess = Resources.TESSERACT1

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
for word in words:
    print(word)

box_bin = AmiOCR.plot_bboxes_on_image(tess_img_bin, words)
io.imshow(box_bin)
io.show()
