#!/usr/bin/env python3
"""
Visualize T-junction candidates without removing them.

This script identifies 4-connected nodes and their neighbor 3-connected nodes
that would be removed in T-junction simplification, but doesn't actually remove them.
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

def has_diagonal_neighbors(skeleton, y, x):
    """Check if a pixel has any diagonal neighbors."""
    diagonal_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonal directions
    
    for dy, dx in diagonal_directions:
        ny, nx = y + dy, x + dx
        if (0 <= ny < skeleton.shape[0] and 
            0 <= nx < skeleton.shape[1] and 
            skeleton[ny, nx] > 0):
            return True
    return False

def has_four_connected_neighbors(skeleton, y, x):
    """Check if a 4-connected pixel has any 4-connected neighbors."""
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if (0 <= ny < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and 
                skeleton[ny, nx] > 0):
                neighbor_count = count_neighbors(skeleton, ny, nx)
                if neighbor_count == 4:
                    return True
    return False

def find_removable_neighbors(skeleton, y, x):
    """Find all orthogonally-connected 3-connected neighbors of a 4-connected pixel that can be removed.
    
    Additional rules:
    - Candidate must not have diagonal neighbors
    - 4-connected pixel must not have 4-connected neighbors
    
    Args:
        skeleton: Binary skeleton image
        y, x: Coordinates of the 4-connected pixel
    
    Returns:
        removable_neighbors: List of (ny, nx) coordinates of removable 3-connected neighbors
    """
    removable_neighbors = []
    
    # Rule: 4-connected pixel must not have 4-connected neighbors
    if has_four_connected_neighbors(skeleton, y, x):
        return removable_neighbors
    
    # Find only orthogonally-connected 3-connected neighbors (not diagonal)
    orthogonal_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    for dy, dx in orthogonal_directions:
        ny, nx = y + dy, x + dx
        if (0 <= ny < skeleton.shape[0] and 
            0 <= nx < skeleton.shape[1] and 
            skeleton[ny, nx] > 0):
            neighbor_count = count_neighbors(skeleton, ny, nx)
            if neighbor_count == 3:
                # Rule: Candidate must not have diagonal neighbors
                if has_diagonal_neighbors(skeleton, ny, nx):
                    continue
                
                # Verify that removing this pixel would leave the center with 3 neighbors
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
                    removable_neighbors.append((ny, nx))
    
    return removable_neighbors

def visualize_t_junction_candidates():
    """Visualize T-junction candidates without removing them."""
    
    print("=== T-Junction Candidates Visualization ===")
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
    
    # Find T-junction candidates
    print("--- Finding T-junction candidates ---")
    four_connected_centers = []  # 4-connected nodes that stay (red)
    three_connected_removable = []  # 3-connected neighbors to remove (yellow)
    
    for y, x in four_connected:
        removable_neighbors = find_removable_neighbors(skeleton, y, x)
        if removable_neighbors:
            four_connected_centers.append((y, x))  # 4-connected node stays
            three_connected_removable.extend(removable_neighbors)  # All removable 3-connected neighbors
    
    print(f"Found {len(four_connected_centers)} 4-connected pixels with removable 3-connected neighbors")
    print(f"Found {len(three_connected_removable)} total 3-connected removable neighbors")
    print()
    
    # Create visualization
    print("--- Creating visualization ---")
    
    # Get bounding box
    min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
    min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
    
    # Add padding
    padding = 50
    min_y = max(0, min_y - padding)
    max_y = min(skeleton.shape[0], max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(skeleton.shape[1], max_x + padding)
    
    # Create base image (skeleton in white)
    img = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
    
    # Draw skeleton pixels in white
    for y, x in coords:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            img[rel_y, rel_x] = [255, 255, 255]  # White
    
    # Mark 4-connected T-junction centers in red (these stay)
    for y, x in four_connected_centers:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            img[rel_y, rel_x] = [0, 0, 255]  # Red
    
    # Mark 3-connected removable neighbors in yellow (these get removed)
    for y, x in three_connected_removable:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            img[rel_y, rel_x] = [0, 255, 255]  # Yellow
    
    # Save visualization
    output_dir = Path("examples/pixel_edit_notebooks/outputs/blackwhite/bridges")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "component_1_t_junction_candidates.png"
    iio.imwrite(output_path, img)
    print(f"Saved T-junction candidates to: {output_path}")
    
    print()
    print("=== Summary ===")
    print(f"Largest component: {len(coords)} pixels")
    print(f"4-connected T-junction centers: {len(four_connected_centers)} (red - stay)")
    print(f"3-connected removable neighbors: {len(three_connected_removable)} (yellow - remove)")
    print(f"Total pixels that would be affected: {len(four_connected_centers) + len(three_connected_removable)}")

if __name__ == "__main__":
    visualize_t_junction_candidates()
