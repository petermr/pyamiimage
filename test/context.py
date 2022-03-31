import os
import sys

top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# add top level root directory to path
sys.path.insert(0, top_level)

# test if import works, if not will throw error
import pyamiimage