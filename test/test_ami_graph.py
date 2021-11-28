"""
tests AmiGraph, AmiNode, AmiEdge, AmiIsland
"""

from pathlib import Path, PosixPath
import numpy as np
import networkx as nx

from test.resources import Resources
from pyimage.graph_lib import AmiGraph, AmiSkeleton
from .test_ami_skeleton import check_type

class TestAmiGraph:

    def test_sknw_5_islands(self):
        """
        This checks all the fields that sknw returns
        :return:
        """
        # island5_skel = AmiGraph.create_ami_graph(Resources.ISLANDS_5_SKEL)
        # print (f"island5_skel = {island5_skel}")
        # assert type(island5_skel) is str, f"type {type(island5_skel)} {island5_skel} should be {str}"
        skel_path = Resources.BIOSYNTH1_ARROWS
        check_type(skel_path, PosixPath)

        ami_skel = AmiSkeleton()
        skeleton_array = ami_skel.create_white_skeleton_image_from_file_IMAGE(skel_path)
        check_type(skeleton_array, np.ndarray)

        nx_graph = AmiSkeleton.create_nx_graph_from_skeleton_wraps_sknw_NX_GRAPH(skeleton_array)
        check_type(nx_graph, nx.classes.graph.Graph)

        check_type(nx_graph.nodes, nx.classes.reportviews.NodeView)
        assert list(nx_graph.nodes) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        check_type(nx_graph.edges, nx.classes.reportviews.EdgeView)
        assert list(nx_graph.edges) == [(0, 2), (1, 4), (2, 4), (2, 3), (2, 7), (4, 5), (4, 6), (8, 19), (9, 19), (10, 12), (11, 13), (12, 13), (12, 18), (13, 14), (13, 15), (16, 18), (17, 18), (18, 20), (19, 26), (21, 24), (22, 24), (23, 24), (24, 25)]


