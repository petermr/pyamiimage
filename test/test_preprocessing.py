from pyimage.preprocessing import Image

def test_image_import():
    image = Image()
    image.image_import()
    assert image is not None