import os
import sys

top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# list of packges to be included under the top level
include_packages = []

# add top level root directory to path
sys.path.insert(0, top_level)

# add each package to be included in sys.path for execution during test
for package in include_packages:
    sys.path.append(os.path.join(top_level, package))

# test if import works, if not will throw error
import pyamiimage