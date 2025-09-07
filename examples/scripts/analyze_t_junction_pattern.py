#!/usr/bin/env python3
"""
Analyze specific T-junction patterns to understand the correct algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v3 as iio
from skimage import measure
import cv2

def count_neighbors(skeleton, y, x):
    """Count 8-connected neighbors of a pixel."""
    count = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if (0 <= ny < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and 
                skeleton[ny, nx] > 0):
                count += 1
    return count

def analyze_t_junction_patterns():
    """Analyze specific T-junction patterns."""
    
    print("=== T-Junction Pattern Analysis ===")
    print("Date: September 6, 2025 (system date of generation)")
    print()
    
    # Load skeleton
    skeleton_path = Path("examples/pixel_edit_notebooks/outputs/blackwhite/skeletonization/skeletonize_default.png")
    if not skeleton_path.exists():
        print(f"Error: Skeleton file not found at {skeleton_path}")
        return
    
    print(f"Loading skeleton: {skeleton_path}")
    skeleton = iio.imread(skeleton_path)
    print(f"Loaded skeleton: {skeleton.shape}, dtype: {skeleton.dtype}")
    print()
    
    # Find connected components
    print("--- Finding connected components ---")
    labeled = measure.label(skeleton, connectivity=2)
    regions = measure.regionprops(labeled)
    print(f"Found {len(regions)} connected components")
    
    # Sort regions by pixel count (largest first)
    regions = sorted(regions, key=lambda r: len(r.coords), reverse=True)
    print(f"Sorted regions by size: largest has {len(regions[0].coords)} pixels")
    print()
    
    # Get the largest component
    if len(regions) == 0:
        print("Error: No components found")
        return
    
    largest_component = regions[0]
    coords = largest_component.coords
    print(f"Largest component: {len(coords)} pixels")
    print()
    
    # Find 4-connected pixels
    print("--- Finding 4-connected pixels ---")
    four_connected = []
    for y, x in coords:
        if (0 <= y < skeleton.shape[0] and 
            0 <= x < skeleton.shape[1] and 
            skeleton[y, x] > 0):
            neighbor_count = count_neighbors(skeleton, y, x)
            if neighbor_count == 4:
                four_connected.append((y, x))
    
    print(f"Found {len(four_connected)} 4-connected pixels")
    print()
    
    # Analyze first 10 4-connected pixels in detail
    print("--- Detailed analysis of first 10 4-connected pixels ---")
    for i, (y, x) in enumerate(four_connected[:10]):
        print(f"\n4-connected pixel {i+1}: ({y}, {x})")
        
        # Get all neighbors with their coordinates and degrees
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and 
                    0 <= nx < skeleton.shape[1] and 
                    skeleton[ny, nx] > 0):
                    neighbor_count = count_neighbors(skeleton, ny, nx)
                    neighbors.append((ny, nx, neighbor_count))
                    print(f"  Neighbor ({ny}, {nx}): {neighbor_count} connections")
        
        # Group neighbors by degree
        by_degree = {}
        for ny, nx, count in neighbors:
            if count not in by_degree:
                by_degree[count] = []
            by_degree[count].append((ny, nx))
        
        print(f"  Summary: {len(neighbors)} neighbors")
        for degree in sorted(by_degree.keys()):
            print(f"    {degree}-connected: {len(by_degree[degree])} neighbors")
        
        # Check if this could be a T-junction
        # Look for pattern: mostly 3-connected neighbors, maybe one 2-connected
        three_connected = by_degree.get(3, [])
        two_connected = by_degree.get(2, [])
        four_connected_neighbors = by_degree.get(4, [])
        
        print(f"  T-junction analysis:")
        print(f"    3-connected neighbors: {len(three_connected)}")
        print(f"    2-connected neighbors: {len(two_connected)}")
        print(f"    4-connected neighbors: {len(four_connected_neighbors)}")
        
        # For each 3-connected neighbor, check if removing it would make center 3-connected
        for ny, nx in three_connected:
            # Simulate removal
            center_neighbors_after = 0
            for check_dy in [-1, 0, 1]:
                for check_dx in [-1, 0, 1]:
                    if check_dy == 0 and check_dx == 0:
                        continue
                    check_y, check_x = y + check_dy, x + check_dx
                    if (0 <= check_y < skeleton.shape[0] and 
                        0 <= check_x < skeleton.shape[1] and 
                        skeleton[check_y, check_x] > 0 and 
                        not (check_y == ny and check_x == nx)):  # Exclude the removable pixel
                        center_neighbors_after += 1
            
            if center_neighbors_after == 3:
                print(f"    REMOVABLE: 3-connected neighbor ({ny}, {nx}) would reduce center to 3-connected")
            else:
                print(f"    NOT removable: 3-connected neighbor ({ny}, {nx}) would leave center with {center_neighbors_after} connections")

if __name__ == "__main__":
    analyze_t_junction_patterns()
