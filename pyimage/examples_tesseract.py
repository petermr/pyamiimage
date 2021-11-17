from pathlib import Path
from tesseract_hocr import TesseractOCR
from pyimage.preprocessing import ImageProcessor

def example_extract_bbox_for_image_without_arrows():
    ocr = TesseractOCR()
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped_arrows_removed.png")
    bbox_coordinates = ocr.extract_bbox_from_image(IMAGE_PATH)
    print("bbox coordinates: ", bbox_coordinates)

def example_extract_bbox_from_hocr_file():
    ocr = TesseractOCR()
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    HOCR_PATH = Path(RESOURCES_DIR, "hocr1.html")
    root = ocr.read_hocr_file(HOCR_PATH)
    bbox_coordinates = ocr.extract_bbox_from_hocr(root)
    print("bbox coordinates: ", bbox_coordinates)

def example_fill_bbox_in_image():
    ocr = TesseractOCR()
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped_arrows_removed.png")
    # IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    image = image_processor.load_image(IMAGE_PATH)
    bbox_coordinates = ocr.extract_bbox_from_image(IMAGE_PATH)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)
    

def example_2_fill_bbox_in_image():
    ocr = TesseractOCR()
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_3.png")
    # IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    image = image_processor.load_image(IMAGE_PATH)
    bbox_coordinates, words = ocr.extract_bbox_from_image(IMAGE_PATH)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)

def example_2_fill_bbox_for_phrases():
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_3.png")
    # IMAGE_PATH = Path(RESOURCES_DIR, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    ocr = TesseractOCR()

    image = image_processor.load_image(IMAGE_PATH)
    print(image.shape)
    # bbox_coordinates, words = extract_bbox_from_image(IMAGE_PATH)
    hocr = ocr.hocr_from_image_path(IMAGE_PATH)
    root = ocr.parse_hocr_string(hocr)
    phrases, bbox_coordinates  = ocr.find_phrases(root)

    words, bbox_coordinates_words = ocr.extract_bbox_from_hocr(root)

    # Print out the phrase and its corresponding coordinates
    for word, bbox in zip(words, bbox_coordinates_words):
        print("Phrase: ", word, " Coordinates: ", bbox)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)

def example_find_phrases():
    ocr = TesseractOCR()
    RESOURCES_DIR = Path(Path(__file__).parent.parent, "test/resources")
    image_processor = ImageProcessor()
    for num in range(4, 9):
        print(f"Now processing biosynth_path_{num}.jpeg")
        IMAGE_PATH = Path(RESOURCES_DIR, f"biosynth_path_{num}.jpeg")
        image = image_processor.load_image(IMAGE_PATH)
    
        hocr = ocr.hocr_from_image_path(IMAGE_PATH)
        root = ocr.parse_hocr_string(hocr)
        phrases, bbox_coordinates  = ocr.find_phrases(root)

        words, bbox_coordinates_words = ocr.extract_bbox_from_hocr(root)

        # Print out the phrase and its corresponding coordinates
        for word, bbox in zip(words, bbox_coordinates_words):
            print("Phrase: ", word, " Coordinates: ", bbox)
    
        ocr.output_phrases_to_file(phrases, f"biosynth_path_{num}.txt")
        bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
        image_processor.show_image(bbox_around_words_image)


def main():
    # example_extract_bbox_for_image_without_arrows()
    # example_extract_bbox_from_hocr_file()
    # example_2_fill_bbox_in_image()
    example_find_phrases()

if __name__ == '__main__':
    main()