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

    BIOSYNTH3_HOCR = Path(TEST_RESOURCE_DIR, "tesseract_biosynth_path_3.hocr.html")
    assert BIOSYNTH3_HOCR.exists(), f"file exists {BIOSYNTH3_HOCR}"

