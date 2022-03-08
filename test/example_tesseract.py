from pyamiimage.tesseract_hocr import TesseractOCR
from test.resources import Resources
from pyamiimage.cleaner import WordCleaner

image_list = [Resources.BIOSYNTH1, Resources.BIOSYNTH2, Resources.BIOSYNTH3, Resources.BIOSYNTH4,
                Resources.BIOSYNTH5, Resources.BIOSYNTH6, Resources.BIOSYNTH7, Resources.BIOSYNTH8]gi

for path in image_list:
    image_hocr = TesseractOCR.hocr_from_image_path(path)
    image_elem = TesseractOCR.parse_hocr_string(image_hocr)

    bboxes, words = TesseractOCR.extract_bbox_from_hocr(image_elem)
    
    print('\n')

    for word, bbox in zip(words, bboxes):
        print(word, bbox)

    print("cleaning words... please wait")
    cleaned, cleaned_bboxes = WordCleaner.remove_trailing_special_characters(words, bboxes)
    cleaned, cleaned_bboxes = WordCleaner.remove_all_single_characters(cleaned, cleaned_bboxes)
    cleaned, cleaned_bboxes = WordCleaner.remove_all_sequences_of_special_characters(cleaned, bboxes)
    cleaned, cleaned_bboxes = WordCleaner.remove_leading_special_characters(cleaned, bboxes)
    cleaned, cleaned_bboxes = WordCleaner.remove_numbers_only(cleaned, bboxes)
    cleaned, cleaned_bboxes = WordCleaner.remove_misread_letters(cleaned, bboxes)

    for word, bbox in zip(cleaned, cleaned_bboxes):
        print(word, bbox)

    print(len(cleaned))
    print(len(cleaned_bboxes))
    phrases, pbboxes = TesseractOCR.find_phrases(cleaned, cleaned_bboxes)
    with open('extracted_phrases.txt', 'a') as f:
        f.writelines(s + '\n' for s in phrases)
    # with open('extracted_words.txt', 'a') as f:
    #     f.writelines(s + '\n' for s in cleaned)
    # print(cleaned)
    # assert phrases is not None
    # assert len(phrases) == 29
    # assert len(bboxes) == 29
    # assert bboxes[0] == [201, 45, 830, 68]
    # assert phrases[0] == "Straight chain ester biosynthesis from fatty acids"    
