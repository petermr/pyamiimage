from pathlib import Path
from tesseract_hocr import TesseractOCR
from pyimage.old_code.preprocessing import ImageProcessor


def example_extract_bbox_for_image_without_arrows():
    ocr = TesseractOCR()
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    image_path = Path(resources_dir, "biosynth_path_1_cropped_arrows_removed.png")
    bbox_coordinates = ocr.extract_bbox_from_image(image_path)
    print("bbox coordinates: ", bbox_coordinates)


def example_extract_bbox_from_hocr_file():
    ocr = TesseractOCR()
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    hocr_path = Path(resources_dir, "hocr1.html")
    root = ocr.read_hocr_file(hocr_path)
    bbox_coordinates = ocr.extract_bbox_from_hocr(root)
    print("bbox coordinates: ", bbox_coordinates)


def example_fill_bbox_in_image():
    ocr = TesseractOCR()
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    image_path = Path(resources_dir, "biosynth_path_1_cropped_arrows_removed.png")
    # image_path = Path(resources_dir, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    image = image_processor.load_image(image_path)
    bbox_coordinates = ocr.extract_bbox_from_image(image_path)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)
    

def example_2_fill_bbox_in_image():
    ocr = TesseractOCR()
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    image_path = Path(resources_dir, "biosynth_path_3.png")
    # image_path = Path(resources_dir, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    image = image_processor.load_image(image_path)
    bbox_coordinates, words = ocr.extract_bbox_from_image(image_path)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)


def example_2_fill_bbox_for_phrases():
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    image_path = Path(resources_dir, "biosynth_path_3.png")
    # image_path = Path(resources_dir, "biosynth_path_1_cropped.png")
    image_processor = ImageProcessor()
    
    ocr = TesseractOCR()

    image = image_processor.load_image(image_path)
    print(image.shape)
    # bbox_coordinates, words = extract_bbox_from_image(image_path)
    hocr = ocr.hocr_from_image_path(image_path)
    root = ocr.parse_hocr_string(hocr)
    phrases, bbox_coordinates = ocr.find_phrases(root)

    words, bbox_coordinates_words = ocr.extract_bbox_from_hocr(root)

    # Print out the phrase and its corresponding coordinates
    for word, bbox in zip(words, bbox_coordinates_words):
        print("Phrase: ", word, " Coordinates: ", bbox)
    
    bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
    image_processor.show_image(bbox_around_words_image)


def example_find_phrases():
    ocr = TesseractOCR()
    resources_dir = Path(Path(__file__).parent.parent, "test/resources")
    image_processor = ImageProcessor()
    for num in range(4, 9):
        print(f"Now processing biosynth_path_{num}.jpeg")
        image_path = Path(resources_dir, f"biosynth_path_{num}.jpeg")
        image = image_processor.load_image(image_path)
    
        hocr = ocr.hocr_from_image_path(image_path)
        root = ocr.parse_hocr_string(hocr)
        phrases, bbox_coordinates = ocr.find_phrases(root)

        words, bbox_coordinates_words = ocr.extract_bbox_from_hocr(root)

        # Print out the phrase and its corresponding coordinates
        for word, bbox in zip(words, bbox_coordinates_words):
            print("Phrase: ", word, " Coordinates: ", bbox)

        ocr.output_phrases_to_file(phrases, f"biosynth_path_{num}.txt")
        bbox_around_words_image = ocr.draw_bbox_around_words(image, bbox_coordinates)
        image_processor.show_image(bbox_around_words_image)


def example_alex_pico_images():
    pass


# not yet working
"""
Hi all,

In preparation for our meeting next week, I went ahead and prepared a subset of our “25 years of pathway figures” database using some simple searches. Please know that I know nothing about terpene biochemistry, so I made two simple-minded subsets:
	1. Based on “terpene” being found in paper title, figure title, caption, or chemical name
	2. All of #1, plus figures with “ADH” found among the extracted gene symbols

Here are shiny apps to browse these subsets and some summary stats:
	1. Terpenes: https://gladstone-bioinformatics.shinyapps.io/shiny-terpene/
		* Figures: 282
		* Papers: 273
		* Total genes: 1,276
		* Unique genes: 491
		* Total chemicals: 2,465
		* Unique chemicals: 551
	2. Terpenes + ADH: https://gladstone-bioinformatics.shinyapps.io/shiny-terpeneadh/
		* Figures: 719
		* Papers: 700
		* Total genes: 10,667
		* Unique genes: 2,699
		* Total chemicals: 6,004
		* Unique chemicals: 931

These subsets are also available as RDS and TSV files if you want to browse them directly.

Maybe these will help make some of our discussion points more tangible.

Note the following caveats that we’d certainly want to improve upon for a “real” effort:
	* Covers pathway figures from 1995 - 2019. We have a 2020 collection, but it’s not online yet.
	* Based on a general literature search for all types of pathway biology; not targeted or trained for terpene biochemistry
	* Based on HGNC human gene lexicon and pubtator chemicals; not targeted for plants or terpenes

Cheers,
 - Alex

 PMC6348685	Comparative transcriptome analysis of a long-time ...	Luodong Huang, ...	2019	Fig. 4	Putative genes and their expression...
PMC6360084	Identification, expression, and phylogenetic analy...	Jipeng Mao, Zid...	2019	Figure 8	Identification, expression, and phy...
PMC6374618	Agarwood Induction: Current Developments and Futur...	Cheng Seng CS T...	2019	FIGURE 4	Schematic relationships between the...
PMC6394206	A chromosomal-scale genome assembly of Tectona gra...	Dongyan Zhao, J...	2019	Figure 5:	Proposed diterpene pathway based on...
PMC6412517	New Insights on Arabidopsis thaliana Root Adaption...	Inmaculada Cole...	2019	Figure 2	Induction of secondary metabolism i...
PMC6423058	Why Algae Release Volatile Organic Compounds—The E...	Zhaojiang Zuo	2019	FIGURE 2	Pathway of terpene synthesis
PMC6437448	Production of ginsenoside aglycone (protopanaxatri...	Yu Shin YS Gwak...	2019	Fig. 1	Biosynthetic pathway engineering fo...
PMC6453882	Proteomic analysis reveals novel insights into tan...	Angela Contrera...	2019	Figure 4	Overview of the proposed biosynthet...
PMC6475556	Protective Effect of the Total Triterpenes of Eusc...	Wei Huang, Hui ...	2019	Figure 7	The mechanisms of total triterpenes...
PMC6486373	Emerging Antitumor Activities of the Bitter Melon ...	Evandro Fei EF ...	2019	Fig. (1)	Possible mechanisms of the anti-tum...
"""


def main():
    # example_extract_bbox_for_image_without_arrows()
    # example_extract_bbox_from_hocr_file()
    # example_2_fill_bbox_in_image()
    example_find_phrases()


if __name__ == '__main__':
    main()
