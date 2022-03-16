import context
from configparser import ConfigParser
from pyamiimage.ami_ocr import AmiOCR, TextBox
from pyamiimage.ami_image import AmiImage
import easyocr

file = 'config.ini'
image_num = 3
file_format = 'png'
# image_file = f'test/resources/biosynth_path_{image_num}/raw.{file_format}'
image_file = "test/resources/Signal_transduction_pathways_wp.png"
config = ConfigParser()
config.read(file)
print(config.sections())
print(config['ocr']['backend'])
reader = easyocr.Reader(['en'])

def create_xy_range_from_bbox(bbox):
    """bbox is in format [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]"""
    xy_range = [[bbox[0][0], bbox[2][0]], [bbox[0][1], bbox[2][1]]]
    return xy_range

def get_words():
    data = reader.readtext(image_file)
    words = []
    for text in data:
        xy_range = create_xy_range_from_bbox(text[0])
        textbox = TextBox(text[1], xy_range)
        words.append(textbox)
    return words

if __name__ == '__main__':
    words = get_words()
    image = AmiImage.create_grayscale_from_file(image_file)
    for item in words:
        print(item)
    bboxed_image = AmiOCR.plot_bboxes_on_image(image, words)
    AmiImage.show(bboxed_image)
    patched_image = AmiOCR.remove_textboxes_from_image(image, words)
    AmiImage.show(patched_image)