import configargparse
import sys


class ImageAnalysis:
    """This is to test PMR args, will probably disappear when Anuv creates some"""
    version = "0.0.1"

    def __init__(self):
        pass

    def handlecli(self):
        """Handles the command line interface using argparse"""
        version = self.version

        parser = configargparse.ArgParser(
            description=f"Welcome to ImageAnalysis version {version}. -h or --help for help",
            add_config_file_help=False,
        )
        parser.add_argument(
            "--image",
            is_config_file=True,
            help="image ... ",
        )
        parser.add_argument(
            "-i",
            "--input",
            help="inout file",
        )

        parser.add_argument(
            "--octree",
            help="octree parameters as name=value pairs; ncol=<cols> ",
            nargs="+"
        )

        parser.add_argument(
            "-v",
            "--version",
            default=False,
            action="store_true",
            help="output the version number",
        )
        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            return

        args = parser.parse_args()
        print(f" <input> {args.input} ; <octree> {args.octree}")
        if args.input:
            self.input(args.input)
        if args.octree:
            self.octree(args.octree)

    def input(self, args):
        print(f"INPUT: {args}")

    def octree(self, args):
        print(f"OCTREE: {len(args)}")
        dikt = dict()
        for arg in args:
            bits = arg.split("=")
            dikt[bits[0]]=bits[1]
        print(f" {dikt}")

def demo():
    """

    :return:
    """
    image_analysis = ImageAnalysis()
    image_analysis.read_file()


def main():
    """Runs the CLI"""
    image_analysis = ImageAnalysis()
    image_analysis.handlecli()


if __name__ == "__main__":
    main()
