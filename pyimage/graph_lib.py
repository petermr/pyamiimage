# various experiments by PMR

# =======================================
# This draws a graph from a map of 0/1 pixels . Uses networkx
# looks useful
# from https://github.com/Image-Py/sknw/blob/master/sknw/sknw.py

import numpy as np
from numba import jit
import networkx as nx
from skimage.morphology import skeletonize
from skimage import data, util
import skimage.io
from pathlib import Path
from skimage.filters import threshold_otsu
import copy

class Sknw:
    def __init__(self):
        self.graph = None
        self.img = None

    def calc_neighbors(self, shape):
        """not sure what this does yet
        I think it might take a region and create margins
        """
        dim = len(shape)
        block = np.ones([3] * dim)
        block[tuple([1] * dim)] = 0
        idx = np.where(block > 0)
        idx = np.array(idx, dtype=np.uint8).T
        idx = np.array(idx - [1] * dim)
        acc = np.cumprod((1,) + shape[::-1][:-1])
        neighbors = np.dot(idx, acc[::-1])
        return neighbors


    # my mark
    def mark(self, buf, nbs):  # mark the array use (0, 1, 2)
        buf = buf.ravel()
        for p in range(len(buf)):
            if buf[p] == 0: continue
            s = 0
            for dp in nbs:
                if buf[p + dp] != 0: s += 1
            if s == 2:
                buf[p] = 1
            else:
                buf[p] = 2


    # trans index to r, c...
    def idx2rc(self, idx, acc):
        rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
        for i in range(len(idx)):
            for j in range(len(acc)):
                rst[i, j] = idx[i] // acc[j]
                idx[i] -= rst[i, j] * acc[j]
        rst -= 1
        return rst


    # fill a node (may be two or more points)
    def fill(self, imgx, p, num, nbs, acc, buf):
        imgx[p] = num
        buf[0] = p
        cur = 0;
        s = 1;
        iso = True;

        while True:
            p = buf[cur]
            for dp in nbs:
                cp = p + dp
                if imgx[cp] == 2:
                    imgx[cp] = num
                    buf[s] = cp
                    s += 1
                if imgx[cp] == 1:
                    iso = False
            cur += 1
            if cur == s:
                break
        # transform ? 1-dim index to row + column indexes
        rc = self.idx2rc(buf[:s], acc)
        return iso, rc


    # trace the edge and use a buffer, then buf.copy, if use [] numba not works
    def trace_edge_of_something(self, img, p, nbs, acc, buf):
        c1 = 0;
        c2 = 0;
        newp = 0
        cur = 1
        while True:
            buf[cur] = p
            img[p] = 0
            cur += 1
            for dp in nbs:
                cp = p + dp
                if img[cp] >= 10:
                    if c1 == 0:
                        c1 = img[cp]
                        buf[0] = cp
                    else:
                        c2 = img[cp]
                        buf[cur] = cp
                if img[cp] == 1:
                    newp = cp
            p = newp
            if c2 != 0:
                break
        # create row and column indexes
        rc = self.idx2rc(buf[:cur + 1], acc)

        rc1 = (c1 - 10, c2 - 10, rc)
        # print("rc1.shape", rc1.shape)
        return rc1


    # parse the image then get the nodes and edges
    def parse_struc(self, img, nbs, acc, iso, ring):
        imgx = img.ravel() # flattened matrix
        # NO IDEA where this number comes from
        buf = np.zeros(131072, dtype=np.int64)
        num = 10 # no idea what this is
        nodes, num = self.create_nodes(acc, buf, imgx, iso, nbs, num)
        edges = self.create_edges(acc, buf, imgx, nbs)
        if ring:
            self.create_cyclic_edges(acc, buf, edges, imgx, nbs, nodes, num)
        return nodes, edges

    def create_cyclic_edges(self, acc, buf, edges, imgx, nbs, nodes, num):
        for p in range(len(imgx)):
            if imgx[p] != 1: continue
            imgx[p] = num;
            num += 1
            nodes.append(self.idx2rc([p], acc))
            for dp in nbs:
                if imgx[p + dp] == 1:
                    edge = self.trace_edge_of_something(imgx, p + dp, nbs, acc, buf)
                    edges.append(edge)

    def create_nodes(self, acc, buf, imgx, iso, nbs, num):
        nodes = []
        for p in range(len(imgx)):
            if imgx[p] == 2:
                isiso, nds = self.fill(imgx, p, num, nbs, acc, buf)
                if isiso and not iso: continue
                num += 1
                nodes.append(nds)
        return nodes, num

    def create_edges(self, acc, buf, imgx, nbs):
        edges = []
        for p in range(len(imgx)):
            if imgx[p] < 10: continue
            for dp in nbs:
                if imgx[p + dp] == 1:
                    edge = self.trace_edge_of_something(imgx, p + dp, nbs, acc, buf)
                    edges.append(edge)
        return edges

    # use nodes and edges build a networkx graph
    def build_graph(self, nodes, edges, multi=False, full=True):
        os = np.array([i.mean(axis=0) for i in nodes])
        if full:
            os = os.round().astype(np.uint16)
        graph = nx.MultiGraph() if multi else nx.Graph()
        for i in range(len(nodes)):
            graph.add_node(i, pts=nodes[i], o=os[i])
        for s, e, pts in edges:
            if full:
                pts[[0, -1]] = os[[s, e]]
            l = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
            graph.add_edge(s, e, pts=pts, weight=l)
        return graph


    def mark_node(self, img):
        node_buf = np.pad(img, (1, 1), mode='constant')
        nbs = self.calc_neighbors(node_buf.shape)
        acc = np.cumprod((1,) + node_buf.shape[::-1][:-1])[::-1]
        self.mark(node_buf, nbs)
        return node_buf

    def build_sknw(self, img, multi=False, iso=True, ring=True, full=True):
        buf = np.pad(img, (1, 1), mode='constant')
        nbs = self.calc_neighbors(buf.shape)
        acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
        self.mark(buf, nbs)
        nodes, edges = self.parse_struc(buf, nbs, acc, iso, ring)
        return self.build_graph(nodes, edges, multi, full)

    def read_thinned_image_calculate_graph_and_plot(self, img):
        node_img = self.mark_node(img)
        self.graph = self.build_sknw(img, False, iso=True, ring=True)

        plt.imshow(node_img[1:-1, 1:-1], cmap='gray')
        self.plot_edges()
        self.plot_nodes()
        plt.title('Build Graph')
        plt.show()


    def plot_nodes(self):
        nodes = self.graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')


    def plot_edges(self):
        # plt.imshow(node_img[1:-1, 1:-1], cmap='gray')
        # draw edges by pts
        for (s, e) in self.graph.edges():
            ps = self.graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    def example1(self):
        self.read_thinned_image_calculate_graph_and_plot(img)

    def example2(self):
        # open and skeletonize
        img = data.horse()
        # ske = skeletonize(img).astype(np.uint16)
        # self.read_thinned_image_calculate_graph_and_plot(ske)
        ske = skeletonize(~img).astype(np.uint16)

        # build graph from skeleton
        graph = sknw.build_sknw(ske)

        # draw image
        plt.imshow(img, cmap='gray')

        # draw edges by pts
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

        # title and show
        plt.title('Build Graph')
        plt.show()

    def example3(self):
        img = Path(Path(__file__).parent.parent, "assets/red_black_cv.png")
        self.skeleton_and_plot(img)

    def example4(self):
        img = Path(Path(__file__).parent.parent, "test/resources/biosynth_path_3.png")
        self.skeleton_and_plot(img)

    def skeleton_and_plot(self, img):
        assert img is not None, "cannot read image"
        img = skimage.io.imread(img, as_gray=True)
        thresh = threshold_otsu(img)
        binary = img < thresh # swap <> to invert image
        print(binary)
        plt.imshow(binary)
        # plt.imshow(inv_ske, cmap='gray')
        plt.title('Build Graph1')
        plt.show()
        # ske = skeletonize(img).astype(np.uint16)
        ske = skeletonize(binary)
        print(ske)
        print("========ske=========")
        # inv_ske = util.invert(ske)
        plt.imshow(ske)
        # plt.imshow(inv_ske, cmap='gray')
        plt.title('Build Graph2')
        plt.show()
        print("========inv=========")
        inv_ske = ske
        self.read_thinned_image_calculate_graph_and_plot(inv_ske)

        # red_black_img = Path(Path(__file__).parent.parent, "assets/red_black_cv.png")
        # assert red_black_img is not None, "cannot read red_black"
        # img = skimage.io.imread(red_black_img, as_gray=True)
        # ske = skeletonize(img).astype(np.uint16)
        # print("Ske values: ", ske)
        # print("Type: ", type(ske))
        # print("Array value datatype: ", type(ske[0][0]))
        # print("Shape: ", ske.shape)
        # self.read_thinned_image_calculate_graph_and_plot(ske)

class AmiGraph():
    """holds AmiNodes and AmiEdges
    may also hold subgraphs
    """
    def __init__(self, generate_nodes=True):
        """create fro nodes and edges"""
        self.ami_node_dict = {}
        self.ami_edge_dict = {}
        self.generate_nodes = generate_nodes

    def read_nodes(self, nodes):
        """create a list of AmiNodes """
        if nodes is not None:
            for node in nodes:
                self.add_raw_node(node)

    def add_raw_node(self, raw_node, fail_on_duplicate=False):
        """add a raw node either a string or string-indexed dict
        if already a dict, deepcopy it
        if a primitive make a node_dict and start it with raw_node as id
        :raw_node: node to add, must have key
        :fail_on_duplicate: if true fail if key already exists
        """
        if raw_node is not None:
            ami_node = AmiNode();
            key = raw_node.key if type(raw_node) is dict else str(raw_node)
            key = "n"+str(key)
            if key in self.ami_node_dict and fail_on_duplicate:
                raise AmiGraphError(f"cannot add same node twice {key}")
            if type(raw_node) is dict:
                self.ami_node_dict[key] = copy.deepcopy(raw_node)
            else:
                self.ami_node_dict[key] = "node" # store just the key at present
        else:
            self.logger.warn("node cannot be None")

    def read_edges(self, edges):
        self.edges = edges
        if len(self.ami_node_dict.keys()) == 0 and self.generate_nodes:
            self.generate_nodes_from_edges()
            print("after node generation", str(self))
        for i, edge in enumerate(self.edges):
            id = "e"+str(i)
            self.add_edge(edge, id)

    def add_edge(self, raw_edge, id, fail_on_duplicate=True):
        if raw_edge is None:
            raise AmiGraphError("cannot add edge=None")
        # node0 =
        edge1 = ("n"+str(raw_edge[0]), "n"+str(raw_edge[1]))
        self.ami_edge_dict[id] = edge1



    def generate_nodes_from_edges(self):
        if self.edges is not None:
            for edge in self.edges:
                self.add_raw_node(edge[0])
                self.add_raw_node(edge[1])

    def __str__(self):
        s = "nodes: " + str(self.ami_node_dict) + \
            "\n edges: " + str(self.ami_edge_dict)
        return s


class AmiNode():
    def __init__(self):
        self.node_dict = {}

class AmiEdge():
    def __init__(self):
        pass

class AmiGraphError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0]])

    sknw = Sknw()
    # sknw.example1()
    sknw.example2() # works
    # sknw.example3() # needs flipping White to black
    # sknw.example4() # needs flipping White to black

