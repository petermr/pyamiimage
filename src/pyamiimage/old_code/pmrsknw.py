from skimage.morphology import skeletonize
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage import data
import skimage.io


# various experiments by PMR

# =======================================
# This draws a graph from a map of 0/1 pixels . Uses networkx
# looks useful
# from https://github.com/Image-Py/sknw/blob/master/sknw/sknw.py

class PmrSknw:
    def __init__(self):
        self.graph = None
        self.img = None

    @classmethod
    def calc_mask_values_of_neighbors(cls, shape):
        """not sure what this does yet
        I think it might take a region and create margins
        """
        dim = len(shape)  # dimension of shape (?always 2)
        # create 3*3 block around a point
        block = np.ones([3] * dim)
        # centre is now 0, so [1,1,1  1,0,1  1,1,1]
        block[tuple([1] * dim)] = 0  # sets (1,1) = 0 (i.e. the centre)
        idx = np.where(block > 0)  # (array([0,0,0 1,1 2,2,2]), array([0,1,2 0,2 0,1,2]))
        idx1 = np.array(idx, dtype=np.uint8).T  # [[0,0], [0,1], [0,2],   [1.0], [1,2],   [2,0], [2,1], [2,2]]
        idx2 = np.array(idx1 - [1] * dim)  # [[-1 -1] [-1 0] [-1 1]   [0, -1] [0 1]   [1 -1] [1 0] [1 1]]
        shape_ = (1,) + shape[::-1][:-1]  # (1, 402)
        acc = np.cumprod(shape_)  # [1 402]
        neighbors = np.dot(idx2, acc[::-1])
        print("neighbors", neighbors.shape, neighbors)  # (8,) [-403 -402 -401 -1 1 401 402 403
        return neighbors

    # my mark
    @classmethod
    def mark(cls, buf, nbs):  # mark the array use (0, 1, 2)
        buf = buf.ravel()
        for p in range(len(buf)):
            if buf[p] == 0:
                continue
            s = 0
            for dp in nbs:
                if buf[p + dp] != 0:
                    s += 1
            if s == 2:
                buf[p] = 1
            else:
                buf[p] = 2

    # trans index to r, c...
    @classmethod
    def idx2rc(cls, idx, acc):
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
        cur = 0
        s = 1
        iso = True

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
        c1 = 0
        c2 = 0
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
        imgx = img.ravel()  # flattened matrix
        # NO IDEA where this number comes from
        buf = np.zeros(131072, dtype=np.int64)
        num = 10  # no idea what this is
        nodes, num = self.create_nodes(acc, buf, imgx, iso, nbs, num)
        edges = self.create_edges(acc, buf, imgx, nbs)
        if ring:
            self.create_cyclic_edges(acc, buf, edges, imgx, nbs, nodes, num)
        return nodes, edges

    def create_cyclic_edges(self, acc, buf, edges, imgx, nbs, nodes, num):
        for p in range(len(imgx)):
            if imgx[p] != 1:
                continue
            imgx[p] = num
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
                if isiso and not iso:
                    continue
                num += 1
                nodes.append(nds)
        return nodes, num

    def create_edges(self, acc, buf, imgx, nbs):
        edges = []
        for p in range(len(imgx)):
            if imgx[p] < 10:
                continue
            for dp in nbs:
                if imgx[p + dp] == 1:
                    edge = self.trace_edge_of_something(imgx, p + dp, nbs, acc, buf)
                    edges.append(edge)
        return edges

    # use nodes and edges build a networkx graph
    @classmethod
    def build_nx_graph(cls, nodes, edges, multi=False, full=True):
        os = np.array([i.mean(axis=0) for i in nodes])
        if full:
            os = os.round().astype(np.uint16)
        nx_graph = nx.MultiGraph() if multi else nx.Graph()
        for i in range(len(nodes)):
            nx_graph.add_node(i, pts=nodes[i], o=os[i])
        for s, e, pts in edges:
            if full:
                pts[[0, -1]] = os[[s, e]]
            ll = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
            nx_graph.add_edge(s, e, pts=pts, weight=ll)
        return nx_graph

    def mark_node(self, img):
        node_buf = np.pad(img, (1, 1), mode='constant')
        nbs = self.calc_mask_values_of_neighbors(node_buf.shape)
        acc = np.cumprod((1,) + node_buf.shape[::-1][:-1])[::-1]
        self.mark(node_buf, nbs)
        return node_buf

    def build_sknw(self, img, multi=False, iso=True, ring=True, full=True):
        """

        :param img: image (probably thinned , maybe binary will have to research)
        :param multi: ??
        :param iso:  ??
        :param ring: I think True allows cycles
        :param full: ??
        :return: nx_graph, nodes, edges
        """
        buf = np.pad(img, (1, 1), mode='constant')  # (330, 402)
        nbs = self.calc_mask_values_of_neighbors(buf.shape)
        acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
        self.mark(buf, nbs)
        nodes, edges = self.parse_struc(buf, nbs, acc, iso, ring)
        nx_graph = self.build_nx_graph(nodes, edges, multi, full)
        return nx_graph, nodes, edges

    def read_thinned_image_calculate_graph_and_plot(self, img):
        node_img = self.mark_node(img)
        self.graph = self.build_sknw(img, False, iso=True, ring=True)
        from networkx.algorithms import tree
        mst = tree.maximum_spanning_edges(self.graph, algorithm="kruskal", data=False)
        edgelist = list(mst)
        print("edges", edgelist)

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
        for (s, e) in self.graph.nx_edges():
            ps = self.graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # def example1(self):
    #     self.read_thinned_image_calculate_graph_and_plot(img)

    @classmethod
    def example2horse(cls):
        # open and skeletonize
        img = data.horse()
        # ske = skeletonize(img).astype(np.uint16)
        # self.read_thinned_image_calculate_graph_and_plot(ske)
        ske = skeletonize(~img).astype(np.uint16)

        # build graph from skeleton
        graph, nodes, edges = PmrSknw().build_sknw(ske)

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
        return  # so we can display in debug mode!

    def example4(self):
        img = Path(Path(__file__).parent.parent, "test/resources/biosynth_path_3.png")
        self.skeleton_and_plot(img)
        return  # so we can display in debug mode!

    def skeleton_and_plot(self, img):
        assert img is not None, "cannot read image"
        img = skimage.io.imread(img, as_gray=True)
        thresh = threshold_otsu(img)
        binary = img < thresh  # swap <> to invert image
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


def main():
    PmrSknw().example3()
    PmrSknw().example4()


if __name__ == '__main__':
    main()
