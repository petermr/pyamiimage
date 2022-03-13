import os
from pathlib import Path

from skimage import io

from pyamiimagex.ami_image import AmiImage

RESOURCE_DIR = Path(Path(__file__).parent, "resources")
COMPARE_DIR = Path(Path(__file__).parent, "comparison_images")

RGB_SNIPPET = Path(RESOURCE_DIR, "snippet_rgb.png")


class AmiImageFileGenerator:
    """This class generates image files for comparison in test_ami_image"""

    def __init__(self, image_path) -> None:
        self.image = io.imread(image_path)
        if not COMPARE_DIR.exists():
            print("Comparison directory does not exist, making one now...")
            os.mkdir(COMPARE_DIR)

    def save_original(self):
        filename = "original.png"
        filepath = Path(COMPARE_DIR, filename)
        self.write_image_to_path(self.image, filepath)
        print(f"Saved {filename} to comparison dir")

    def generate_rbg2gray_image(self):
        filename = "gray.png"
        gray = AmiImage.create_grayscale_from_image(self.image)
        filepath = Path(COMPARE_DIR, filename)
        self.write_image_to_path(gray, filepath)
        print(f"Saved {filename} to comparion dir")

    def generate_inverted_image(self):
        filename = "inverted.png"
        inverted = AmiImage.create_inverted_image(self.image)
        filepath = Path(COMPARE_DIR, filename)
        self.write_image_to_path(inverted, filepath)
        print(f"Saved {filename} to comparion dir")

    def generate_white_binary_image(self):
        filename = "white_binary.png"
        white_binary = AmiImage.create_white_binary_from_image(self.image)
        filepath = Path(COMPARE_DIR, filename)
        self.write_image_to_path(white_binary, filepath)
        print(f"Saved {filename} to comparion dir")

    def generate_white_skeleton_image(self):
        filename = "white_skeleton.png"
        inverted = AmiImage.create_inverted_image(self.image)
        white_skeleton = AmiImage.create_white_skeleton_from_image(inverted)
        filepath = Path(COMPARE_DIR, filename)
        self.write_image_to_path(white_skeleton, filepath)
        print(f"Saved {filename} to comparion dir")

    def generate_all(self):
        self.save_original()
        self.generate_rbg2gray_image()
        self.generate_inverted_image()
        self.generate_white_binary_image()
        self.generate_white_skeleton_image()

    @staticmethod
    def write_image_to_path(image, path):
        io.imsave(path, image)


def main():
    image_path = RGB_SNIPPET
    file_generator = AmiImageFileGenerator(image_path)
    file_generator.generate_all()


if __name__ == "__main__":
    main()
