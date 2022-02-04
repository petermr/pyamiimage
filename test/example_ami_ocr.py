from pyimage.ami_ocr import TextBox, AmiOCR
from test.resources import Resources
from skimage import io

biosynth2 = Resources.BIOSYNTH2
biosynth2_img = io.imread(biosynth2)
biosynth2_ocr = AmiOCR(biosynth2)

med_xrd = Resources.MED_XRD_FIG_A_LABELS
med_xrd_img = io.imread(med_xrd)
med_xrd_ocr = AmiOCR(med_xrd)

# words = biosynth2_ocr.get_words()
# # biosynth2_ocr.find_baseline()
# for word in words:
#     print(word)


# phrases = biosynth2_ocr.get_phrases()
# for phrase in phrases:
#     print(phrase)

# # biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(biosynth2_img, words)
# # io.imshow(biosynth2_img_bboxes)
# # io.show()

# biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(biosynth2_img, phrases)
# io.imshow(biosynth2_img_bboxes)
# io.show()

# # groups = biosynth2_ocr.get_groups()
# # for group in groups:
# #     print(group)


words = med_xrd_ocr.get_words()
for word in words:
    print(word)


phrases = med_xrd_ocr.get_phrases()
for phrase in phrases:
    print(phrase)

# biosynth2_img_bboxes = AmiOCR.plot_bboxes_on_image(biosynth2_img, words)
# io.imshow(biosynth2_img_bboxes)
# io.show()

med_xrd_img_bboxes = AmiOCR.plot_bboxes_on_image(med_xrd_img, words)
io.imshow(med_xrd_img_bboxes)
io.show()

# groups = biosynth2_ocr.get_groups()
# for group in groups:
#     print(group)