"""
Debug version of sknw algorithm with proper logging and variable names.
This will help us understand why edge tracing is failing.
"""
import numpy as np
import networkx as nx
from numba import jit

def neighbors(shape):
    """Compute neighbor offsets for a given image shape"""
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

def mark_pixels(img, neighbor_offsets):
    """Mark pixels as: 0=background, 1=edge, 2=node"""
    img = img.ravel()
    print(f"DEBUG: Marking pixels in image of size {len(img)}")
    
    for pixel_idx in range(len(img)):
        if img[pixel_idx] == 0:
            continue
            
        # Count neighbors
        neighbor_count = 0
        for neighbor_offset in neighbor_offsets:
            neighbor_idx = pixel_idx + neighbor_offset
            if neighbor_idx >= 0 and neighbor_idx < len(img) and img[neighbor_idx] != 0:
                neighbor_count += 1
                
        # Mark based on neighbor count
        if neighbor_count == 2:
            img[pixel_idx] = 1  # Edge pixel
        else:
            img[pixel_idx] = 2  # Node pixel
            
    # Count marked pixels
    edge_pixels = np.sum(img == 1)
    node_pixels = np.sum(img == 2)
    print(f"DEBUG: Marked {edge_pixels} edge pixels, {node_pixels} node pixels")

def idx2rc(indices, acc):
    """Convert flattened indices to row/column coordinates"""
    result = np.zeros((len(indices), len(acc)), dtype=np.int16)
    for i in range(len(indices)):
        for j in range(len(acc)):
            result[i,j] = indices[i] // acc[j]
            indices[i] -= result[i,j] * acc[j]
    result -= 1
    return result

def fill_node_region(img, start_pixel, node_number, neighbor_offsets, acc, buffer):
    """Fill connected node region and return if isolated"""
    img[start_pixel] = node_number
    buffer[0] = start_pixel
    current = 0
    size = 1
    is_isolated = True
    
    print(f"DEBUG: Filling node region starting at pixel {start_pixel}")
    
    while True:
        current_pixel = buffer[current]
        for neighbor_offset in neighbor_offsets:
            neighbor_pixel = current_pixel + neighbor_offset
            if neighbor_pixel >= 0 and neighbor_pixel < len(img):
                if img[neighbor_pixel] == 2:  # Another node pixel
                    img[neighbor_pixel] = node_number
                    buffer[size] = neighbor_pixel
                    size += 1
                elif img[neighbor_pixel] == 1:  # Edge pixel
                    is_isolated = False
                    
        current += 1
        if current == size:
            break
            
    print(f"DEBUG: Filled node region with {size} pixels, isolated={is_isolated}")
    return is_isolated, idx2rc(buffer[:size], acc)

def trace_edge_path(img, start_pixel, neighbor_offsets, acc, buffer):
    """Trace edge path between two nodes"""
    print(f"DEBUG: Tracing edge path starting at pixel {start_pixel}")
    
    node1_id = 0
    node2_id = 0
    new_pixel = 0
    current = 1
    
    buffer[current] = start_pixel
    img[start_pixel] = 0  # Mark as visited
    
    while True:
        current_pixel = buffer[current]
        current += 1
        
        # Look for neighbors
        for neighbor_offset in neighbor_offsets:
            neighbor_pixel = current_pixel + neighbor_offset
            if neighbor_pixel >= 0 and neighbor_pixel < len(img):
                if img[neighbor_pixel] >= 10:  # Found a node
                    if node1_id == 0:
                        node1_id = img[neighbor_pixel]
                        buffer[0] = neighbor_pixel
                        print(f"DEBUG: Found first node {node1_id}")
                    else:
                        node2_id = img[neighbor_pixel]
                        buffer[current] = neighbor_pixel
                        print(f"DEBUG: Found second node {node2_id}")
                elif img[neighbor_pixel] == 1:  # Edge pixel
                    new_pixel = neighbor_pixel
                    
        if node2_id != 0:
            break
            
        if new_pixel == 0:
            print(f"DEBUG: No more edge pixels to trace")
            break
            
        p = new_pixel
        
    print(f"DEBUG: Traced edge from node {node1_id-10} to node {node2_id-10}")
    return (node1_id-10, node2_id-10, idx2rc(buffer[:current+1], acc))

def parse_structure(img, neighbor_offsets, acc, find_isolated, find_rings):
    """Parse image to extract nodes and edges"""
    img = img.ravel()
    buffer = np.zeros(131072, dtype=np.int64)
    node_number = 10
    nodes = []
    edges = []
    
    print(f"DEBUG: Starting structure parsing")
    print(f"DEBUG: Image size: {len(img)}")
    print(f"DEBUG: Parameters: isolated={find_isolated}, rings={find_rings}")
    
    # Step 1: Find and fill node regions
    print(f"\n--- STEP 1: Finding Nodes ---")
    for pixel_idx in range(len(img)):
        if img[pixel_idx] == 2:  # Node pixel
            is_isolated, node_coords = fill_node_region(img, pixel_idx, node_number, neighbor_offsets, acc, buffer)
            if is_isolated and not find_isolated:
                print(f"DEBUG: Skipping isolated node {node_number}")
                continue
            print(f"DEBUG: Added node {node_number} with {len(node_coords)} coordinates")
            nodes.append(node_coords)
            node_number += 1
            
    print(f"DEBUG: Found {len(nodes)} nodes")
    
    # Step 2: Trace edges between nodes
    print(f"\n--- STEP 2: Tracing Edges ---")
    edge_count = 0
    for pixel_idx in range(len(img)):
        if img[pixel_idx] < 10:  # Not a node
            continue
            
        # Look for edge pixels adjacent to this node
        for neighbor_offset in neighbor_offsets:
            neighbor_pixel = pixel_idx + neighbor_offset
            if neighbor_pixel >= 0 and neighbor_pixel < len(img) and img[neighbor_pixel] == 1:
                print(f"DEBUG: Found edge pixel {neighbor_pixel} adjacent to node {img[pixel_idx]}")
                edge = trace_edge_path(img, neighbor_pixel, neighbor_offsets, acc, buffer)
                edges.append(edge)
                edge_count += 1
                
    print(f"DEBUG: Traced {edge_count} edges")
    
    # Step 3: Handle rings (if requested)
    if find_rings:
        print(f"\n--- STEP 3: Finding Rings ---")
        ring_count = 0
        for pixel_idx in range(len(img)):
            if img[pixel_idx] != 1:  # Not an edge pixel
                continue
                
            img[pixel_idx] = node_number
            nodes.append(idx2rc([pixel_idx], acc))
            print(f"DEBUG: Added ring node {node_number}")
            node_number += 1
            
            # Trace edges from this ring node
            for neighbor_offset in neighbor_offsets:
                neighbor_pixel = pixel_idx + neighbor_offset
                if neighbor_pixel >= 0 and neighbor_pixel < len(img) and img[neighbor_pixel] == 1:
                    edge = trace_edge_path(img, neighbor_pixel, neighbor_offsets, acc, buffer)
                    edges.append(edge)
                    ring_count += 1
                    
            print(f"DEBUG: Found {ring_count} ring edges")
    
    print(f"\n--- FINAL RESULT ---")
    print(f"DEBUG: Total nodes: {len(nodes)}")
    print(f"DEBUG: Total edges: {len(edges)}")
    
    return nodes, edges

def build_graph(nodes, edges, multi=False, full=True):
    """Build NetworkX graph from nodes and edges"""
    print(f"DEBUG: Building graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Calculate node centroids
    node_centroids = np.array([node_coords.mean(axis=0) for node_coords in nodes])
    if full:
        node_centroids = node_centroids.round().astype(np.uint16)
        
    # Create graph
    graph = nx.MultiGraph() if multi else nx.Graph()
    
    # Add nodes
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=node_centroids[i])
        
    # Add edges
    for start_node, end_node, path_points in edges:
        if full:
            path_points[[0,-1]] = node_centroids[[start_node, end_node]]
        path_length = np.linalg.norm(path_points[1:]-path_points[:-1], axis=1).sum()
        graph.add_edge(start_node, end_node, pts=path_points, weight=path_length)
        
    print(f"DEBUG: Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def build_sknw_debug(skeleton_image, multi=False, iso=True, ring=True, full=True):
    """Debug version of build_sknw with detailed logging"""
    print(f"=== SKNW DEBUG EXECUTION ===")
    print(f"Input skeleton shape: {skeleton_image.shape}")
    print(f"Parameters: multi={multi}, iso={iso}, ring={ring}, full={full}")
    
    # Pad image
    padded_image = np.pad(skeleton_image, (1,1), mode='constant').astype(np.uint16)
    print(f"Padded image shape: {padded_image.shape}")
    
    # Compute neighbor offsets and accumulation
    neighbor_offsets = neighbors(padded_image.shape)
    acc = np.cumprod((1,)+padded_image.shape[::-1][:-1])[::-1]
    print(f"Neighbor offsets shape: {neighbor_offsets.shape}")
    print(f"Accumulation array: {acc}")
    
    # Mark pixels
    print(f"\n--- MARKING PIXELS ---")
    mark_pixels(padded_image, neighbor_offsets)
    
    # Parse structure
    print(f"\n--- PARSING STRUCTURE ---")
    nodes, edges = parse_structure(padded_image, neighbor_offsets, acc, iso, ring)
    
    # Build graph
    print(f"\n--- BUILDING GRAPH ---")
    graph = build_graph(nodes, edges, multi, full)
    
    return graph

if __name__ == '__main__':
    # Test with the problematic Y-shape
    test_image = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    
    print("Testing Y-shape with debug sknw...")
    graph = build_sknw_debug(test_image, multi=True, iso=True, ring=True, full=True)
    
    print(f"\nFinal result:")
    print(f"Graph type: {type(graph)}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    if len(graph.nodes) > 0:
        print(f"Node data: {dict(graph.nodes)}")
    if len(graph.edges) > 0:
        print(f"Edge data: {dict(graph.edges)}")









