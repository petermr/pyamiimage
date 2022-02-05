"""Resources such as data used by other modules
This may develop into a dataclass"""

from pathlib import Path
from skimage import io
import logging
import numpy as np
# local
try:
    from ..pyimage.ami_graph_all import AmiGraph
    from ..pyimage.tesseract_hocr import TesseractOCR
    from ..pyimage.ami_image import AmiImageDTO
except:
    from pyimage.ami_graph_all import AmiGraph
    from pyimage.tesseract_hocr import TesseractOCR
    from pyimage.ami_image import AmiImageDTO

logger = logging.getLogger(__name__)

class Resources:
    TEST_RESOURCE_DIR = Path(Path(__file__).parent, "resources")
    assert TEST_RESOURCE_DIR.exists(), f"dir exists {TEST_RESOURCE_DIR}"
    # biosynth1 is the most analysed image. In places it is cropped to provide a subset
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
    BIOSYNTH1_ARROWS_TEXT_SVG = Path(TEST_RESOURCE_DIR, "biosynth1_arrows_text.svg")
    assert BIOSYNTH1_ARROWS_TEXT_SVG.exists(), f"file exists {BIOSYNTH1_ARROWS_TEXT_SVG}"
    BIOSYNTH1_RAW_ARROWS_SVG = Path(TEST_RESOURCE_DIR, "biosynth1_raw_arrows.svg")
    assert BIOSYNTH1_RAW_ARROWS_SVG.exists(), f"file exists {BIOSYNTH1_RAW_ARROWS_SVG}"

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
    TESSERACT1 = Path(TEST_RESOURCE_DIR, "tesseract_test.png")
    assert TESSERACT1.exists(), f"file exists {TESSERACT1}"
    TESSERACT_BENG = Path(TEST_RESOURCE_DIR, "tesseract_beng.jpg")
    assert TESSERACT_BENG.exists(), f"file doesn't exist {TESSERACT_BENG}"
    TESSERACT_GER = Path(TEST_RESOURCE_DIR, "tesseract_ger.jpg")
    assert TESSERACT_GER.exists(), f"file doesn't exist {TESSERACT_GER}"
    TESSERACT_GER2 = Path(TEST_RESOURCE_DIR, "tesseract_ger2.png")
    assert TESSERACT_GER2.exists(), f"file doesn't exist {TESSERACT_GER2}"
    TESSERACT_ITA = Path(TEST_RESOURCE_DIR, "tesseract_ita.png")
    assert TESSERACT_ITA.exists(), f"file doesn't exist {TESSERACT_ITA}"
    MED_XRD = Path(TEST_RESOURCE_DIR, "MED_34909142_3.jpeg")
    assert MED_XRD.exists(), f"file doesn't exist {MED_XRD}"
    MED_XRD_FIG_A = Path(TEST_RESOURCE_DIR, "MED_34909142_3_figA.jpeg")
    assert MED_XRD_FIG_A.exists(), f"file doesn't exist {MED_XRD_FIG_A}"
    MED_XRD_FIG_A_LABELS = Path(TEST_RESOURCE_DIR, "MED_34909142_3_figA_labels.jpg")
    assert MED_XRD_FIG_A_LABELS.exists(), f"file doesn't exist {MED_XRD_FIG_A_LABELS}"
    MED_XRD_FIG_A_YTICKS = Path(TEST_RESOURCE_DIR, "MED_34909142_3_figA_vert_label_num.png")
    assert MED_XRD_FIG_A_YTICKS.exists(), f"file doesn't exist {MED_XRD_FIG_A_YTICKS}"
    SHAPES_1 = Path(TEST_RESOURCE_DIR, "test_img_shapes.png")
    assert SHAPES_1.exists(), f"file doesn't exist {SHAPES_1}"
    

    YW5003_5 = Path(TEST_RESOURCE_DIR, "iucr_yw5003_fig5.png")
    assert YW5003_5.exists(), f"file exists {YW5003_5}"

# =====================

    TEMP_DIR = Path(TEST_RESOURCE_DIR.parent.parent, "temp")
    assert TEMP_DIR.is_dir(), f"file exists {TEMP_DIR}"

    def __init__(self):
        self.cached = False

        self.arrows1_image = None
        self.nx_graph_arrows1 = None

        # DTO approach
        self.biosynth1_dto = None
        self.biosynth2_dto = None
        self.biosynth3_dto = None

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

# DTO approach
            if not self.biosynth1_dto:
                self.biosynth1_dto = self.get_image_dto(raw_image_file = Resources.BIOSYNTH1, raw_image_shape = (1167, 1515), threshold = 127)
            # if not self.biosynth2_dto:
            #     self.biosynth2_dto = self.get_image_dto(raw_image_file = Resources.BIOSYNTH2, raw_image_shape = (1391, 1420, 3), threshold = 127)
            if not self.biosynth3_dto:
                self.biosynth3_dto = self.get_image_dto(raw_image_file = Resources.BIOSYNTH3, raw_image_shape = (972, 1020), threshold = 127)
            self.biosynth6_compounds_dto = self.get_image_dto(raw_image_file = \
                                                    Resources.BIOSYNTH6COMPOUND, raw_image_shape = (967, 367, 3), threshold = 127)

            return self

    def get_image_dto(self, raw_image_file, raw_image_shape=None, threshold=127, ):
        """
        return Data Transfer Object containin downstream image artefacts
        create one of these for each image being processed
        :return: DTO with artefacts
        """

        image_dto = AmiImageDTO()

        image_dto.image_file = raw_image_file
        image_dto.image = io.imread(raw_image_file)
        if raw_image_shape is not None:
            assert image_dto.image.shape == raw_image_shape, f"expected {image_dto.image.shape}"
        image_dto.image_binary = np.where(image_dto.image < threshold, 0, 255)
        image_dto.nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(raw_image_file)
        image_dto.ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(raw_image_file)
        image_dto.hocr = TesseractOCR.hocr_from_image_path(raw_image_file)
        image_dto.hocr_html_element = TesseractOCR.parse_hocr_string(image_dto.hocr)

        return image_dto





