#!/usr/bin/env python3
"""
Remove T-junctions by deleting 3-connected pixels that meet the criteria.
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

def remove_t_junctions():
    """Remove T-junctions by deleting qualifying 3-connected pixels."""
    
    print("=== T-Junction Removal ===")
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
    
    # Create a copy of the skeleton for modification
    modified_skeleton = skeleton.copy()
    
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
    
    # Find and remove T-junction candidates
    print("--- Finding and removing T-junction candidates ---")
    removed_pixels = []
    four_connected_centers = []
    
    for y, x in four_connected:
        removable_neighbors = find_removable_neighbors(skeleton, y, x)
        if removable_neighbors:
            four_connected_centers.append((y, x))
            for ny, nx in removable_neighbors:
                # Remove the 3-connected pixel
                modified_skeleton[ny, nx] = 0
                removed_pixels.append((ny, nx))
    
    print(f"Found {len(four_connected_centers)} 4-connected pixels with removable neighbors")
    print(f"Removed {len(removed_pixels)} 3-connected pixels")
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
    
    # Create before image (original skeleton in white, 4-connected centers in red, removable in yellow)
    before_img = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
    
    # Draw skeleton pixels in white
    for y, x in coords:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            before_img[rel_y, rel_x] = [255, 255, 255]  # White
    
    # Mark 4-connected centers in red
    for y, x in four_connected_centers:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            before_img[rel_y, rel_x] = [0, 0, 255]  # Red
    
    # Mark removed pixels in yellow
    for y, x in removed_pixels:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            before_img[rel_y, rel_x] = [0, 255, 255]  # Yellow
    
    # Create after image (modified skeleton in white, resulting 3-connected in blue)
    after_img = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
    
    # Draw modified skeleton pixels in white
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if (0 <= y < modified_skeleton.shape[0] and 
                0 <= x < modified_skeleton.shape[1] and 
                modified_skeleton[y, x] > 0):
                rel_x, rel_y = x - min_x, y - min_y
                after_img[rel_y, rel_x] = [255, 255, 255]  # White
    
    # Mark resulting 3-connected pixels in blue
    for y, x in four_connected_centers:
        rel_x, rel_y = x - min_x, y - min_y
        if 0 <= rel_x < max_x - min_x and 0 <= rel_y < max_y - min_y:
            # Check if this pixel is now 3-connected
            neighbor_count = count_neighbors(modified_skeleton, y, x)
            if neighbor_count == 3:
                after_img[rel_y, rel_x] = [255, 0, 0]  # Blue
    
    # Save visualizations
    output_dir = Path("examples/pixel_edit_notebooks/outputs/blackwhite/bridges")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    before_path = output_dir / "component_1_t_junctions_before_removal.png"
    after_path = output_dir / "component_1_t_junctions_after_removal.png"
    modified_skeleton_path = output_dir / "component_1_skeleton_after_t_removal.png"
    
    iio.imwrite(before_path, before_img)
    iio.imwrite(after_path, after_img)
    iio.imwrite(modified_skeleton_path, modified_skeleton)
    
    print(f"Saved before removal to: {before_path}")
    print(f"Saved after removal to: {after_path}")
    print(f"Saved modified skeleton to: {modified_skeleton_path}")
    
    print()
    print("=== Summary ===")
    print(f"Original skeleton: {len(coords)} pixels")
    print(f"4-connected centers: {len(four_connected_centers)} (red in before, blue in after if now 3-connected)")
    print(f"Removed 3-connected pixels: {len(removed_pixels)} (yellow in before)")
    print(f"Modified skeleton: {np.sum(modified_skeleton > 0)} pixels")
    print(f"Pixels removed: {len(coords) - np.sum(modified_skeleton > 0)}")

if __name__ == "__main__":
    remove_t_junctions()
