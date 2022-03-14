import argparse
import sys

from pyamiimagex.ami_ocr import AmiOCR

class CommandLine:
    """[summary]"""

    def __init__(self) -> None:
        self.version = "0.0.7"

    def execute(self, args):
        if args.text:
            ocr = AmiOCR(args.infile.name)
            words = ocr.get_words()
            words = ocr.clean_all(words)
            ocr.write_list_to_file(words, args.outfile.name)

    def handlecli(self):
        """Handles the command line interface using argpase"""
        parser = argparse.ArgumentParser(description="Welcome to pyamiimagex, view --help")
        parser.add_argument(
            'infile', 
            nargs='?', 
            type=argparse.FileType('r'),
            default=sys.stdin
            )
        parser.add_argument(
            'outfile',
            nargs='?',
            type=argparse.FileType('w'),
            default=sys.stdout
        )
        parser.add_argument(
            "-t",
            "--text",
            action="store_true",
            help="Run AmiOCR on a given Image"
        )

        # TODO add octree OR colour_separation
        # TODO manage temp repository
        # TODO use CProject strategy - iterate over CProject or list of CTrees
        # TODO write images from matplotlib to disk or screen
        # TODO store text and image analysis out put in CTrees/CProject
        # TODO remove either graphics or text from images (Anuv)
        # TODO use pygetpapers to download images and captions / liaise with docanalysis


        args = parser.parse_args()
        print(args)
        self.execute(args)

def main():
    """Runs the CLI"""
    pyamiimage = CommandLine()
    pyamiimage.handlecli()

if __name__ == "__main__":
    main()