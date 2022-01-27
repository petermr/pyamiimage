from pyimage.ami_ocr import TextBox, AmiOCR
from test.resources import Resources


biosynth2 = Resources.BIOSYNTH2
biosynth2_ocr = AmiOCR(biosynth2)

# words = biosynth2_ocr.get_words()
biosynth2_ocr.find_baseline()
# for word in words:
#     print(word)

