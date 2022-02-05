from pyamiimage.test.resources import Resources

from pyimage.ami_ocr import AmiOCR
from pyimage.bbox import BBox
from pyimage.ami_image import AmiImage

from skimage import io

img = Resources.MED_XRD_FIG_A
img_gray = AmiImage.create_grayscale_from_file(img)
plot_bbox = BBox([[82, 389], [28, 386]])

img_ticks = Resources.MED_XRD_FIG_A_YTICKS
img_tick_gray = AmiImage.create_grayscale_from_file(img_ticks)

io.imshow(img_tick_gray)
io.show()

img_tk_ocr = AmiOCR(image=img_tick_gray)
hocr_str = img_tk_ocr.hocr_string_from_path("C:/Users/chakr/projects/pyamiimage/temp/tesseract/default.png", psm='3')
print(hocr_str.decode('utf-8'))