from pyamiimage.commandline import Pyamiimage

def main():
    """Runs the CLI"""
    pyamiimage = Pyamiimage()
    pyamiimage.create_parsers()

if __name__ == "__main__":
    main()