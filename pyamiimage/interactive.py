#!/usr/bin/env python3
"""
Interactive - Interactive Skeletonization Dashboard

Main entry point for the interactive skeletonization and graph analysis dashboard.
This application reuses existing pyamiimage skeletonization code and provides
an interactive Tkinter interface for parameter adjustment and visualization.

Usage:
    python -m pyamiimage.interactive
    python pyamiimage/interactive.py
"""

import sys
import tkinter as tk
from pathlib import Path

try:
    from pyamiimage.interactive.gui.main_window import SkeletonizationDashboard
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install opencv-python scikit-image skan networkx pillow numpy matplotlib")
    sys.exit(1)


def main():
    """Main entry point for the Interactive dashboard."""
    try:
        # Create the main application
        app = SkeletonizationDashboard()
        
        # Run the dashboard
        print("Interactive Dashboard started successfully!")
        print("Use File -> Open Image to load an image for skeletonization.")
        app.run()
        
    except Exception as e:
        print(f"Error starting Interactive dashboard: {e}")
        print("Please check that all dependencies are installed correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
