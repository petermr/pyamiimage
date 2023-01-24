
import os

# class TestImageNew:

def test_read_image():
    # file = "/Users/pm286/workspace/pyamiimage/test/resources/biosynth1_cropped/raw.png"
    # print (file)
    # img = io.imread(file)
    import cv2
    import skimage
    file = "/Users/pm286/Documents/pmrSignature.png"
    file = "/Users/pm286/workspace/pyamiimage/test/resources/biosynth1_cropped/arrows_removed.png"
    file = "/Users/pm286/workspace/pyamiimage/test/resources/biosynth1_cropped/text_removed.png"
    file = "/Users/pm286/workspace/pyamiimage/test/resources/biosynth_path_1/raw.png"
    img = cv2.imread(file)
    print(f"img {img} {img.shape}")
    print("=================")
    img = skimage.io.imread(file)
    print(f"img {img} {img.shape}")
