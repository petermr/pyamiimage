"""Resources such as data used by other modules
This may develop into a dataclass"""

from pathlib import Path
from skimage import io
import logging
import numpy as np
# local
from ..pyimage.ami_graph_all import AmiGraph
from ..pyimage.tesseract_hocr import TesseractOCR
from ..pyimage.ami_image import AmiImageDTO

logger = logging.getLogger(__name__)

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
    BIOSYNTH6COMPOUND = Path(TEST_RESOURCE_DIR, "biosynth_path_6_compounds_only.jpeg")
    assert BIOSYNTH6COMPOUND.exists(), f"file exists {BIOSYNTH6COMPOUND}"
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
    PRIMITIVES = Path(TEST_RESOURCE_DIR, "primitives.png")
    assert PRIMITIVES.exists(), f"file exists {PRIMITIVES}"

    def __init__(self):
        self.cached = False

        self.arrows1_image = None
        self.nx_graph_arrows1 = None


    def create_ami_graph_objects(self):
        """creates image derivatives
        """
        logger.debug(f"{__name__} create_ami_graph_objects {self.cached}")
        if not self.cached:
            logger.warning(f"{__name__} setting up Resources" )
            self.cached = True
            self.arrows1_image = io.imread(Resources.BIOSYNTH1_ARROWS)
            assert self.arrows1_image.shape == (315, 1512)
            self.arrows1_image = np.where(self.arrows1_image < 127, 0, 255)
            self.nx_graph_arrows1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)
            self.arrows1_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_ARROWS)

# biosynth1 has solid arrowheads and (unfortunately) some primitives overlap
            self.biosynth1 = io.imread(Resources.BIOSYNTH1)
            assert self.biosynth1.shape == (1167, 1515)
            self.biosynth1_binary = np.where(self.biosynth1 < 127, 0, 255)
            self.nx_graph_biosynth1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1)
            self.biosynth1_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH1)
            self.biosynth1_elem = TesseractOCR.parse_hocr_string(self.biosynth1_hocr)
            self.biosynth1_ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1)

            self.biosynth3_dto = self.get_processed_image_objects(raw_image_file = Resources.BIOSYNTH3, raw_image_shape = (972, 1020), threshold = 127)
            self.biosynth6_compounds_dto = self.get_processed_image_objects(raw_image_file = \
                                                    Resources.BIOSYNTH6COMPOUND, raw_image_shape = (967, 367, 3), threshold = 127)


            prisma = io.imread(Resources.PRISMA)
            assert prisma.shape == (667, 977, 4)
            self.nx_graph_prisma = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.PRISMA)

            self.battery1_image = io.imread(Resources.BATTERY1)
            assert self.battery1_image.shape == (546, 1354, 3)
            self.battery1_binary = np.where(self.battery1_image < 127, 0, 255)
            self.nx_graph_battery1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BATTERY1)

            self.battery1bsquare = io.imread(Resources.BATTERY1BSQUARE)
            assert self.battery1_image.shape == (546, 1354, 3)
            # self.battery1_binary = np.where(self.battery1 < 127, 0, 255)
            self.nx_graph_battery1bsquare = AmiGraph.create_nx_graph_from_arbitrary_image_file(
                Resources.BATTERY1BSQUARE)

            self.primitives = io.imread(Resources.PRIMITIVES)
            assert self.primitives.shape == (405, 720, 3)
            self.nx_graph_primitives = AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.PRIMITIVES)

            # clear plot
            #     plt.figure().close("all")
            #     plt.clf() # creates unwanted blank screens

            return self

    def get_processed_image_objects(self, raw_image_file, raw_image_shape=None, threshold=127, ):
        """
        return Data Transfer Object containin downstream image artefacts
        :return: DTO with artefacts
        """

        image_dto = AmiImageDTO()
        image_dto.np_image = io.imread(raw_image_file)
        if raw_image_shape is not None:
            assert image_dto.np_image.shape == raw_image_shape, f"expected {image_dto.np_image.shape}"
        image_dto.image_binary = np.where(image_dto.np_image < threshold, 0, 255)
        image_dto.nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(raw_image_file)
        image_dto.ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(raw_image_file)
        image_dto.hocr = TesseractOCR.hocr_from_image_path(raw_image_file)
        image_dto.hocr_html_element = TesseractOCR.parse_hocr_string(image_dto.hocr)

        return image_dto





