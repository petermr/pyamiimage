import argparse
import sys
import numpy as np
import imageio.v3 as iio

from pyamiimage.ami_ocr import AmiOCR
from pyamiimage.ami_image import AmiImage


class Pyamiimage:
    """[summary]"""

    def __init__(self) -> None:
        self.version = "0.0.7"



    def execute(self, args):
        if args.text:
            ocr = AmiOCR(args.infile.name, backend=args.ocr_wrapper)
            textboxes = ocr.get_textboxes()
            AmiOCR.write_text_to_file(textboxes, args.outfile.name)
        elif args.skeletonize:
            # Read input image
            image = iio.imread(args.infile.name)
            # Create skeleton using specified method
            skeleton = AmiImage.create_white_skeleton_from_image(image, method=args.skeletonize_method)
            # Save output skeleton
            iio.imwrite(args.outfile.name, skeleton)
            print(f"Skeleton created using {args.skeletonize_method} method and saved to {args.outfile.name}")

    def handlecli(self):
        """Handles the command line interface using argpase"""
        parser = argparse.ArgumentParser(description="Welcome to pyamiimage, view --help")
        parser.add_argument(
            '--infile', 
            nargs='?', 
            type=argparse.FileType('r'),
            default=sys.stdin,
            required=True
            )
        parser.add_argument(
            '--outfile',
            nargs='?',
            type=argparse.FileType('w'),
            default=sys.stdout,
            required=True
        )
        parser.add_argument(
            "-t",
            "--text",
            action="store_true",
            help="Run AmiOCR on a given Image"
        )
        parser.add_argument(
            "--ocr_wrapper",
            type=str,
            default="easyocr",
            choices=["easyocr", "tesseract"],
            help="Choose between easyocr and tesseract for ocr"
        )
        parser.add_argument(
            "-s",
            "--skeletonize",
            action="store_true",
            help="Run skeletonization on a given image"
        )
        parser.add_argument(
            "--skeletonize-method",
            type=str,
            default="medial_axis",
            choices=["medial_axis", "skeletonize", "thin"],
            help="Choose skeletonization method: medial_axis, skeletonize, or thin"
        )
        args = parser.parse_args()
        self.execute(args)

def main():
    """Runs the CLI"""
    pyamiimage = Pyamiimage()
    pyamiimage.handlecli()