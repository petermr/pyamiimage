"""Resources such as data used by other modules
This may develop into a dataclass"""

from pathlib import Path


class Resources:
    TEST_RESOURCE_DIR = Path(Path(__file__).parent, "resources")
    assert TEST_RESOURCE_DIR.exists(), f"dir exists {TEST_RESOURCE_DIR}"
    BIOSYNTH1 = Path(TEST_RESOURCE_DIR, "biosynth_path_1.png")
    assert BIOSYNTH1.exists(), f"file exists {BIOSYNTH1}"
    BIOSYNTH1_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_1.hocr")
    assert BIOSYNTH1_HOCR.exists(), f"file exists {BIOSYNTH1_HOCR}"
    BIOSYNTH1_CROPPED = Path(TEST_RESOURCE_DIR, "biosynth_path_1_cropped.png")
    assert BIOSYNTH1_CROPPED.exists(), f"file exists {BIOSYNTH1_CROPPED}"
    BIOSYNTH1_TEXT = Path(TEST_RESOURCE_DIR, "biosynth_path_1_cropped_arrows_removed.png")
    assert BIOSYNTH1_TEXT.exists(), f"file exists {BIOSYNTH1_TEXT}"
    BIOSYNTH1_ARROWS = Path(TEST_RESOURCE_DIR, "biosynth_path_1_cropped_text_removed.png")
    assert BIOSYNTH1_ARROWS.exists(), f"file exists {BIOSYNTH1_ARROWS}"
    BIOSYNTH2 = Path(TEST_RESOURCE_DIR, "biosynth_path_2.jpg")
    assert BIOSYNTH2.exists(), f"file exists {BIOSYNTH2}"
    BIOSYNTH3 = Path(TEST_RESOURCE_DIR, "biosynth_path_3.png")
    assert BIOSYNTH3.exists(), f"file exists {BIOSYNTH3}"
    BIOSYNTH4 = Path(TEST_RESOURCE_DIR, "biosynth_path_4.jpeg")
    assert BIOSYNTH4.exists(), f"file exists {BIOSYNTH4}"
    BIOSYNTH5 = Path(TEST_RESOURCE_DIR, "biosynth_path_5.jpeg")
    assert BIOSYNTH5.exists(), f"file exists {BIOSYNTH5}"
    BIOSYNTH6 = Path(TEST_RESOURCE_DIR, "biosynth_path_6.jpeg")
    assert BIOSYNTH6.exists(), f"file exists {BIOSYNTH6}"
    BIOSYNTH7 = Path(TEST_RESOURCE_DIR, "biosynth_path_7.jpeg")
    assert BIOSYNTH7.exists(), f"file exists {BIOSYNTH7}"
    BIOSYNTH8 = Path(TEST_RESOURCE_DIR, "biosynth_path_8.jpeg")
    assert BIOSYNTH8.exists(), f"file exists {BIOSYNTH8}"

    BIOSYNTH3_HOCR = Path(TEST_RESOURCE_DIR, "tesseract_biosynth_path_3.hocr.html")
    assert BIOSYNTH3_HOCR.exists(), f"file exists {BIOSYNTH3_HOCR}"
    BIOSYNTH4_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_4.hocr")
    assert BIOSYNTH4_HOCR.exists(), f"file exists {BIOSYNTH4_HOCR}"
    BIOSYNTH5_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_5.hocr")
    assert BIOSYNTH5_HOCR.exists(), f"file exists {BIOSYNTH5_HOCR}"
    BIOSYNTH6_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_6.hocr")
    assert BIOSYNTH6_HOCR.exists(), f"file exists {BIOSYNTH6_HOCR}"
    BIOSYNTH7_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_7.hocr")
    assert BIOSYNTH7_HOCR.exists(), f"file exists {BIOSYNTH7_HOCR}"
    BIOSYNTH8_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_8.hocr")
    assert BIOSYNTH8_HOCR.exists(), f"file exists {BIOSYNTH8_HOCR}"

    ISLANDS_5_SKEL = Path(TEST_RESOURCE_DIR, "islands_5.png")
    assert ISLANDS_5_SKEL.exists(), f"file exists {ISLANDS_5_SKEL}"

    PRISMA = Path(TEST_RESOURCE_DIR, "prisma.png")
    assert PRISMA.exists(), f"file exists {PRISMA}"

    BATTERY1 = Path(TEST_RESOURCE_DIR, "green.png")
    assert BATTERY1.exists(), f"file exists {BATTERY1}"
    BATTERY1BSQUARE = Path(TEST_RESOURCE_DIR, "battery1bsquare.png")
    assert BATTERY1BSQUARE.exists(), f"file exists {BATTERY1BSQUARE}"
