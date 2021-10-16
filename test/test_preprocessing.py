from pyimage.preprocessing import ImageProcessor

def test_load_image():
    image_processor = ImageProcessor()
    image_processor.load_image()
    assert image_processor.image is not None

def test_to_gray():
    image_processor = ImageProcessor()
    image_processor.load_image()
    image_processor.to_gray()
    # converts a 3 dimensional array(length, width & channel) to 2 dimensional array (length, width)
    assert len(image_processor.image.shape) is 2

def test_show_image():
    image_processor = ImageProcessor()
    assert image_processor.show_image()