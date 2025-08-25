#!/usr/bin/env python3
"""
Main entry point for running ATPOE as a module.
"""

from pyamiimage.atpoe.gui.main_window import SkeletonizationDashboard


def main():
    """Main entry point for the ATPOE dashboard."""
    try:
        # Create the main application
        app = SkeletonizationDashboard()
        
        # Run the dashboard
        print("ATPOE Dashboard started successfully!")
        print("Use File -> Open Image to load an image for skeletonization.")
        app.run()
        
    except Exception as e:
        print(f"Error starting ATPOE dashboard: {e}")
        print("Please check that all dependencies are installed correctly.")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()




