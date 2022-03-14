import argparse
import sys

from ami_ocr import AmiOCR


class Pyamiimage:
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
        parser = argparse.ArgumentParser(description="Welcome to pyamiimage, view --help")
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
        args = parser.parse_args()
        print(args)
        self.execute(args)

def main():
    """Runs the CLI"""
    pyamiimage = Pyamiimage()
    pyamiimage.handlecli()

if __name__ == "__main__":
    main()