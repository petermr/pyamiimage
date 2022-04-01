import argparse
import sys
import pprint

from pyamiimage.ami_ocr import AmiOCR

from argparse import ArgumentParser

class Pyamiimage:
    """[summary]"""

    def __init__(self) -> None:
        self.version = "0.0.7"
        self.parser = None
        self.ocr_parser = None
        self.color_parser = None

    def execute(self, args):
        if args.text:
            ocr = AmiOCR(args.infile.name)
            textboxes = ocr.get_textboxes()
            AmiOCR.write_text_to_file(textboxes, args.outfile.name)

    def create_parsers(self):
        """Handles the command line interface using argpase"""
        self.parser = argparse.ArgumentParser(description="Welcome to pyamiimage, view --help")
        self.parser.add_argument(
            "-i",
            '--infile',
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

        # args = parser.parse_args()
        # print(f"a1 {args}")
        # v = vars(args)
        # print(f"v1 {v}")

        subparsers1 = self.parser.add_subparsers(dest='subparser1')
        self.ocr_parser = subparsers1.add_parser('ocr')
        self.ocr_parser.add_argument('--method', default='tesseract', choices=['tesseract', 'easyocr'])
        self.ocr_parser.add_argument('--temp', nargs=1, type=int, default=3)

        self.color_parser = subparsers1.add_parser("col")
        self.color_parser.add_argument('--ncol', type=int, default=8)
        self.color_parser.add_argument('--method', default='kmeans', choices=['kmeans', 'octree'])

        # subparsers2 = self.parser.add_subparsers(dest='subparsers2')
        # self.subparser2_1 = subparsers2.add_parser('sub2_1')
        # self.subparser2_1.add_argument('--sub2x', default='sss', choices=['sss', 'ttt'])
        # self.subparser2_1.add_argument('--sub2y', nargs=1, type=int, default=3)
        #
        # self.subparser2_2 = subparsers1.add_parser("sub2_2")
        # self.subparser2_2.add_argument('--sub2_2x', type=int, default=22)
        # self.subparser2_2.add_argument('--sub2_2y', default='c22', choices=['c22', 'd22'])

        self.print_parsers()

    def print_parsers(self):
        print("no parse")
        return
        # args = self.parser.parse_args("--infile inin --outfile outout")
        # args = self.parser.parse_args(["ocr", "-s", "colour", "-c"])
        args = self.parser.parse_args(["--text"])
        print(f"\ntest {vars(args)}")
        args = parser.parse_args()
        self.execute(args)
        args = self.parser.parse_args(["col", "--ncol", "8"])
        print(f"\ncol {vars(args)}")
        args = self.parser.parse_args(["col", "--method", "octree"])
        # print(args)
        # v = vars(args)
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(f" vars: {v}")
        self.execute(args)
        print("\nparser............")
        self.parser.print_help()
        print("\nocr_parser.........")
        self.ocr_parser.print_help()
        print("\ncolor_parser.........")
        self.color_parser.print_help()
        # parser.parse_args(['2', 'frobble'])
        # Namespace(subparser_name='2', y='frobble')


def main():
    """Runs the CLI"""
    # pyamiimage = Pyamiimage()
    # pyamiimage.create_parsers()
    test1()
    #test2()

def test1():
    pyamiimage = Pyamiimage()
    pyamiimage.create_parsers()
    args = ["col", "--ncol", "6"]
    pyamiimage.execute(args)
    args = ["col", "--method", "octree", "--ncol", "6"]
    pyamiimage.execute(args)
    args = ["ocr", "--method", "easyocr"]
    pyamiimage.execute(args)
    args = ["ocr", "--method", "easyocr", "--temp", "5"]
    pyamiimage.execute(args)

if __name__ == '__main__':
    main()

# python CLI
"""
https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f
"""

# cli = ArgumentParser()
# subparsers = cli.add_subparsers(dest="subcommand")
#
#
# def argument(*name_or_flags, **kwargs):
#     """Convenience function to properly format arguments to pass to the
#     subcommand decorator.
#     """
#     return (list(name_or_flags), kwargs)
#
#
# def subcommand(args=[], parent=subparsers):
#     """Decorator to define a new subcommand in a sanity-preserving way.
#     The function will be stored in the ``func`` variable when the parser
#     parses arguments so that it can be called directly like so::
#         args = cli.parse_args()
#         args.func(args)
#     Usage example::
#         @subcommand([argument("-d", help="Enable debug mode", action="store_true")])
#         def subcommand(args):
#             print(args)
#     Then on the command line::
#         $ python cli.py subcommand -d
#     """
#     def decorator(func):
#         parser = parent.add_parser(func.__name__, description=func.__doc__)
#         for arg in args:
#             parser.add_argument(*arg[0], **arg[1])
#         parser.set_defaults(func=func)
#     return decorator
#
#
# @subcommand()
# def nothing(args):
#     print("Nothing special!")
#
#
# @subcommand([argument("-d", help="Debug mode", action="store_true")])
# def test(args):
#     print(args)
#
#
# @subcommand([argument("-f", "--filename", help="A thing with a filename")])
# def filename(args):
#     print(args.filename)
#
#
# @subcommand([argument("name", help="Name")])
# def name(args):
#     print(args.name)
#
#
# def test2(args):
#     args = cli.parse_args()
#     if args.subcommand is None:
#         cli.print_help()
#     else:
#         args.func(args)