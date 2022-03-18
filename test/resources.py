"""Resources such as data used by other modules
This may develop into a dataclass"""

import logging
from pathlib import Path

import numpy as np
from skimage import io

# local
import context
from pyamiimage.ami_graph_all import AmiGraph
from pyamiimage.ami_image import AmiImageDTO
from pyamiimage.tesseract_hocr import TesseractOCR

logger = logging.getLogger(__name__)


class Resources:
    TEST_RESOURCE_DIR = Path(Path(__file__).parent, "resources")
    TEST_RESOURCE_DIR.exists(), f"dir exists {TEST_RESOURCE_DIR}"
    # biosynth1 is the most analysed image. In places it is cropped to provide a subset

# biosynth1
    BIOSYNTH1_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_1")
    BIOSYNTH1_RAW = Path(BIOSYNTH1_DIR, "raw.png")
    # assert BIOSYNTH1_RAW.exists(), f"file exists {BIOSYNTH1_RAW}"
    BIOSYNTH1_HOCR = Path(BIOSYNTH1_DIR, "hocr.html")
    # assert BIOSYNTH1_HOCR.exists(), f"file exists {BIOSYNTH1_HOCR}"

# cropped
    BIOSYNTH1_CROPPED_DIR = Path(TEST_RESOURCE_DIR, "biosynth1_cropped")
    BIOSYNTH1_CROPPED_PNG = Path(BIOSYNTH1_CROPPED_DIR , "raw.png")
    # assert BIOSYNTH1_CROPPED_PNG.exists(), f"file exists {BIOSYNTH1_CROPPED_PNG}"
    BIOSYNTH1_TEXT = Path(BIOSYNTH1_CROPPED_DIR, "arrows_removed.png")
    # assert BIOSYNTH1_TEXT.exists(), f"file exists {BIOSYNTH1_TEXT}"

    BIOSYNTH1_CROPPED_ARROWS_RAW = Path(BIOSYNTH1_CROPPED_DIR, "text_removed.png")
    # assert BIOSYNTH1_CROPPED_ARROWS_RAW.exists(), f"file exists {BIOSYNTH1_CROPPED_ARROWS_RAW}"

    BIOSYNTH1_ARROWS_DIR = Path(TEST_RESOURCE_DIR, "biosynth1_arrows")
    BIOSYNTH1_ARROWS_TEXT_SVG = Path(BIOSYNTH1_ARROWS_DIR, "text.svg")
    # assert (BIOSYNTH1_ARROWS_TEXT_SVG.exists()), f"file exists {BIOSYNTH1_ARROWS_TEXT_SVG}"
    BIOSYNTH1_RAW_ARROWS_SVG = Path(BIOSYNTH1_ARROWS_DIR, "raw_arrows.svg")
    # assert BIOSYNTH1_RAW_ARROWS_SVG.exists(), f"file exists {BIOSYNTH1_RAW_ARROWS_SVG}"

    BIOSYNTH2_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_2")
    BIOSYNTH2_RAW = Path(BIOSYNTH2_DIR, "raw.jpg")
    # assert BIOSYNTH2_RAW.exists(), f"file exists {BIOSYNTH2_RAW}"
    BIOSYNTH3_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_3")
    BIOSYNTH3_RAW = Path(BIOSYNTH3_DIR, "raw.png")
    # assert BIOSYNTH3_RAW.exists(), f"file exists {BIOSYNTH3_RAW}"
    BIOSYNTH4_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_4")
    BIOSYNTH4_RAW = Path(BIOSYNTH4_DIR, "raw.jpeg")
    # assert BIOSYNTH4_RAW.exists(), f"file exists {BIOSYNTH4_RAW}"
    BIOSYNTH5_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_5")
    BIOSYNTH5_RAW = Path(BIOSYNTH5_DIR, "raw.jpeg")
    # assert BIOSYNTH5_RAW.exists(), f"file exists {BIOSYNTH5_RAW}"
    BIOSYNTH6_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_6")
    BIOSYNTH6_RAW = Path(BIOSYNTH6_DIR, "raw.jpeg")
    # assert BIOSYNTH6_RAW.exists(), f"file exists {BIOSYNTH6_RAW}"
    BIOSYNTH6COMPOUND_RAW = Path(BIOSYNTH6_DIR, "compounds_only.jpeg")
    # assert BIOSYNTH6COMPOUND_RAW.exists(), f"file exists {BIOSYNTH6COMPOUND_RAW}"
    BIOSYNTH7_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_7")
    BIOSYNTH7_RAW = Path(BIOSYNTH7_DIR, "raw.jpeg")
    # assert BIOSYNTH7_RAW.exists(), f"file exists {BIOSYNTH7_RAW}"
    BIOSYNTH8_DIR = Path(TEST_RESOURCE_DIR, "biosynth_path_8")
    BIOSYNTH8_RAW = Path(BIOSYNTH8_DIR, "raw.jpeg")
    # assert BIOSYNTH8_RAW.exists(), f"file exists {BIOSYNTH8_RAW}"

    BIOSYNTH3_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_3", "hocr.html")
    # assert BIOSYNTH3_HOCR.exists(), f"file exists {BIOSYNTH3_HOCR}"
    BIOSYNTH4_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_4", "hocr.html")
    # assert BIOSYNTH4_HOCR.exists(), f"file exists {BIOSYNTH4_HOCR}"
    BIOSYNTH5_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_5", "hocr.html")
    # assert BIOSYNTH5_HOCR.exists(), f"file exists {BIOSYNTH5_HOCR}"
    BIOSYNTH6_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_6", "hocr.html")
    # assert BIOSYNTH6_HOCR.exists(), f"file exists {BIOSYNTH6_HOCR}"
    BIOSYNTH7_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_7", "hocr.html")
    # assert BIOSYNTH7_HOCR.exists(), f"file exists {BIOSYNTH7_HOCR}"
    BIOSYNTH8_HOCR = Path(TEST_RESOURCE_DIR, "biosynth_path_8", "hocr.html")
    # assert BIOSYNTH8_HOCR.exists(), f"file exists {BIOSYNTH8_HOCR}"

    ISLANDS_5_SKEL_RAW = Path(TEST_RESOURCE_DIR, "islands_5.png")
    # assert ISLANDS_5_SKEL_RAW.exists(), f"file exists {ISLANDS_5_SKEL_RAW}"

    PRISMA_RAW = Path(TEST_RESOURCE_DIR, "prisma.png")
    # assert PRISMA_RAW.exists(), f"file exists {PRISMA_RAW}"

    # https://europepmc.org/article/MED/34909142#figures-and-tables
    MED_34909142_3_RAW = Path(TEST_RESOURCE_DIR, "MED_34909142_3.jpeg")
    # assert MED_34909142_3_RAW.exists(), f"file exists {MED_34909142_3_RAW}"
    MED_XRD_FIG_A_RAW = Path(TEST_RESOURCE_DIR, "MED_34909142_3_figA.jpeg")
    # assert MED_XRD_FIG_A_RAW.exists(), f"file {MED_XRD_FIG_A_RAW} doesn't exist"

    BATTERY_DIR = Path(TEST_RESOURCE_DIR, "battery")
    BATTERY1_RAW = Path(BATTERY_DIR, "capacity_r_g_b.png")
    # assert BATTERY1_RAW.exists(), f"file exists {BATTERY1_RAW}"
    BATTERY1BSQUARE_RAW = Path(BATTERY_DIR, "battery1bsquare.png")
    # assert BATTERY1BSQUARE_RAW.exists(), f"file exists {BATTERY1BSQUARE_RAW}"
    BATTERY2_RAW = Path(BATTERY_DIR, "battery2.png")
    # assert BATTERY2_RAW.exists(), f"file exists {BATTERY2_RAW}"

    PRIMITIVES_RAW = Path(TEST_RESOURCE_DIR, "primitives.png")
    # assert PRIMITIVES_RAW.exists(), f"file exists {PRIMITIVES_RAW}"

    YW5003_5_RAW = Path(TEST_RESOURCE_DIR, "iucr_yw5003_fig5.png")
    # assert YW5003_5_RAW.exists(), f"file exists {YW5003_5_RAW}"

    SHAPES_1_RAW = Path(TEST_RESOURCE_DIR, "test_img_shapes.png")
    # assert SHAPES_1_RAW.exists(), f"file {SHAPES_1_RAW} doesn't exist"

    # =====================

    TEMP_DIR = Path(TEST_RESOURCE_DIR.parent.parent, "temp")
    # assert TEMP_DIR.is_dir(), f"file exists {TEMP_DIR}"

    def __init__(self):
        self.cached = False

        self.arrows1_image = None
        self.nx_graph_arrows1 = None

        # DTO approach
        self.biosynth1_dto = None
        self.biosynth2_dto = None
        self.biosynth3_dto = None

    def create_ami_graph_objects(self):
        """creates image derivatives"""
        logger.debug(f"{__name__} create_ami_graph_objects {self.cached}")
        if not self.cached:
            logger.warning(f"{__name__} setting up Resources")
            self.cached = True
            self.arrows1_image = io.imread(Resources.BIOSYNTH1_CROPPED_ARROWS_RAW)
            # assert self.arrows1_image.shape == (315, 1512)
            self.arrows1_image = np.where(self.arrows1_image < 127, 0, 255)
            self.nx_graph_arrows1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(
                Resources.BIOSYNTH1_CROPPED_ARROWS_RAW
            )
            self.arrows1_ami_graph = (
                AmiGraph.create_ami_graph_from_arbitrary_image_file(
                    Resources.BIOSYNTH1_CROPPED_ARROWS_RAW
                )
            )

            # biosynth1 has solid arrowheads and (unfortunately) some primitives overlap
            self.biosynth1 = io.imread(Resources.BIOSYNTH1_RAW)
            # assert self.biosynth1.shape == (1167, 1515)
            self.biosynth1_binary = np.where(self.biosynth1 < 127, 0, 255)
            self.nx_graph_biosynth1 = (
                AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_RAW)
            )
            self.biosynth1_hocr = TesseractOCR.hocr_from_image_path(Resources.BIOSYNTH1_RAW)
            self.biosynth1_elem = TesseractOCR.parse_hocr_string(self.biosynth1_hocr)
            self.biosynth1_ami_graph = (
                AmiGraph.create_ami_graph_from_arbitrary_image_file(Resources.BIOSYNTH1_RAW)
            )

            prisma = io.imread(Resources.PRISMA_RAW)
            # assert prisma.shape == (667, 977, 4)
            self.nx_graph_prisma = AmiGraph.create_nx_graph_from_arbitrary_image_file(
                Resources.PRISMA_RAW
            )

            self.battery1_image = io.imread(Resources.BATTERY1_RAW)
            # assert self.battery1_image.shape == (546, 1354, 3)
            self.battery1_binary = np.where(self.battery1_image < 127, 0, 255)
            self.nx_graph_battery1 = AmiGraph.create_nx_graph_from_arbitrary_image_file(
                Resources.BATTERY1_RAW
            )

            self.battery1bsquare = io.imread(Resources.BATTERY1BSQUARE_RAW)
            # assert self.battery1_image.shape == (546, 1354, 3)
            # self.battery1_binary = np.where(self.battery1 < 127, 0, 255)
            self.nx_graph_battery1bsquare = (
                AmiGraph.create_nx_graph_from_arbitrary_image_file(
                    Resources.BATTERY1BSQUARE_RAW
                )
            )

            self.primitives = io.imread(Resources.PRIMITIVES_RAW)
            # assert self.primitives.shape == (405, 720, 3)
            self.nx_graph_primitives = (
                AmiGraph.create_nx_graph_from_arbitrary_image_file(Resources.PRIMITIVES_RAW)
            )

            # DTO approach
            if not self.biosynth1_dto:
                self.biosynth1_dto = self.get_image_dto(
                    raw_image_file=Resources.BIOSYNTH1_RAW,
                    raw_image_shape=(1167, 1515),
                    threshold=127,
                )
            # if not self.biosynth2_dto:
            #     self.biosynth2_dto = self.get_image_dto(raw_image_file = Resources.BIOSYNTH2, raw_image_shape = (1391, 1420, 3), threshold = 127)
            if not self.biosynth3_dto:
                self.biosynth3_dto = self.get_image_dto(
                    raw_image_file=Resources.BIOSYNTH3_RAW,
                    raw_image_shape=(972, 1020),
                    threshold=127,
                )
            self.biosynth6_compounds_dto = self.get_image_dto(
                raw_image_file=Resources.BIOSYNTH6COMPOUND_RAW,
                raw_image_shape=(967, 367, 3),
                threshold=127,
            )

            return self

    def get_image_dto(
        self,
        raw_image_file,
        raw_image_shape=None,
        threshold=127,
    ):
        """
        return Data Transfer Object containin downstream image artefacts
        create one of these for each image being processed
        :return: DTO with artefacts
        """

        image_dto = AmiImageDTO()

        image_dto.image_file = raw_image_file
        image_dto.image = io.imread(raw_image_file)
        if raw_image_shape is not None:
            assert (
                image_dto.image.shape == raw_image_shape
            ), f"expected {image_dto.image.shape}"
        image_dto.image_binary = np.where(image_dto.image < threshold, 0, 255)
        image_dto.nx_graph = AmiGraph.create_nx_graph_from_arbitrary_image_file(
            raw_image_file
        )
        image_dto.ami_graph = AmiGraph.create_ami_graph_from_arbitrary_image_file(
            raw_image_file
        )
        image_dto.hocr = TesseractOCR.hocr_from_image_path(raw_image_file)
        image_dto.hocr_html_element = TesseractOCR.parse_hocr_string(image_dto.hocr)

        return image_dto





