import os

top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# list of packges to be included under the top level
include_packages = []

# test if import works, if not will throw error
import pyamiimage