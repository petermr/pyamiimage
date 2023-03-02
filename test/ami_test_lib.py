import unittest

class AmiAnyTest(unittest.TestCase):
    # for marking and skipping unittests
    # skipUnless
    ADMIN = True  # check that low-level files, tools, etc. work
    CMD = True   # test runs the commandline
    DEBUG = True   # test runs the commandline
    LONG = True   # test runs for a long time
    NET = True    # test requires Internet
    OLD = True    # test probably out of data
    VERYLONG = False   # test runs for a long time
    # skipIf
    NYI = True    # test not yet implemented
    USER = True   # user-facing test
    BUG = True    # skip BUGs

    def setUp(self) -> None:
        # if len(sys.argv) == 0:
        #     sys.argv = ["ami"]
        # self.argv_copy = list(sys.argv)
        pass

    def tearDown(self) -> None:
        # print(f"argv_copy {self.argv_copy}")
        # print(f"argv {sys.argv}")
        # self.argv = list(self.argv_copy)
        pass


