# from myslic import *
# from otherslic import *
import time
import unittest
# from myslic import *
import cv2
import math
import numpy as np
from skimage import io, color
from numpy.testing import assert_almost_equal
import copy
from slic_fcts import *


class TestSLIC(unittest.TestCase):
    K = 40
    M = 50
    data = io.imread('parrot0.jpg')
    data = color.rgb2lab(data)
    image_height = data.shape[0]
    image_width = data.shape[1]
    N = image_height * image_width
    S = int(math.sqrt(N / K))
    coord = get_grid_coordinates(image_height, image_width, S, 10)
    border = 2 * S
    clusters = []
    for c in coord:
        h_ = int(c[0])
        w_ = int(c[1])
        # Replace make clusters
        c = Cluster(h_, w_, data[h_][w_][0], data[h_][w_][1], data[h_][w_][2])
        clusters.append(c)
    data = cv2.copyMakeBorder(data, border, border, border, border, cv2.BORDER_CONSTANT)
    image_height = data.shape[0]
    image_width = data.shape[1]
    for c in clusters:
        c.update(c.h + border, c.w + border, c.l, c.a, c.b)

    cluster_pos0 = np.empty((len(clusters), 2))
    for i, cluster in enumerate(clusters):
        cluster_pos0[i] = [cluster.h, cluster.w]

    label = {}
    dis = np.full((image_height, image_width), -1.0)
    dis[border:image_height - border, border:image_width - border] = np.inf
    D = np.full((image_height, image_width), -1.0)
    D[border:image_height - border, border:image_width - border] = np.inf
    LABEL = np.full((image_height, image_width), -1.0)
    clusters_temp = np.empty((clusters[0].cluster_index, 3))

    test_slices = np.empty((len(clusters), 4 * S, 4 * S, 5), dtype='float32')  # [x,y,l,a,b] per pixel
    test_slices_d = np.empty((len(clusters), 4 * S, 4 * S), dtype='float32')
    test_slices_mask = np.empty((len(clusters), 4 * S, 4 * S), dtype='bool') * False
    test_final_labels = np.ones(D.shape) * -1

    for i, cluster in enumerate(clusters):
        # print("this is cluster ", i)
        for h in range(cluster.h - 2 * S, cluster.h + 2 * S):
            for w in range(cluster.w - 2 * S, cluster.w + 2 * S):
                L, A, B = data[h][w]
                h_ = h - (cluster.h - 2 * S)
                w_ = w - (cluster.w - 2 * S)
                test_slices[i, h_, w_] = h, w, L, A, B
                Dc = math.sqrt(
                    math.pow(L - cluster.l, 2) +
                    math.pow(A - cluster.a, 2) +
                    math.pow(B - cluster.b, 2))
                Ds = math.sqrt(
                    math.pow(h - cluster.h, 2) +
                    math.pow(w - cluster.w, 2))
                dist = math.sqrt(math.pow(Dc / M, 2) + math.pow(Ds / S, 2))
                D[h, w] = dist
                test_slices_d[i, h_, w_] = dist
                if D[h, w] < dis[h][w]:
                    if (h, w) not in label:
                        label[(h, w)] = cluster
                    else:
                        label[(h, w)].pixels.remove((h, w))
                        label[(h, w)] = cluster
                    cluster.pixels.append((h, w))
                    # here
                    test_slices_mask[i, h_, w_] = True
                    test_final_labels[h, w] = int(cluster.no)
                    dis[h][w] = D[h, w]

    new_clusters = copy.deepcopy(clusters)
    clusters_pos = np.empty((len(new_clusters), 2), dtype=float)
    for i, cluster in enumerate(new_clusters):
        sum_h = sum_w = number = 0
        for p in cluster.pixels:
            sum_h += p[0]
            sum_w += p[1]
            number += 1
        _h = int(sum_h / number)
        _w = int(sum_w / number)
        cluster.update(_h, _w, data[_h][_w][0], data[_h][_w][1], data[_h][_w][2])
        clusters_pos[i] = np.array([int(_h), int(_w)])

    def test_slices_fct(self):
        mat = get_slices(self.cluster_pos0, self.S, self.data)
        np.testing.assert_allclose(self.test_slices, mat)

    def test_slices_d_fct(self):
        slices = get_slices(self.cluster_pos0, self.S, self.data)
        mat = get_slices_d(self.cluster_pos0, slices, self.S, self.M, self.data)
        np.testing.assert_allclose(self.test_slices_d, mat, rtol=1e-6, atol=1e-6)

    def test_slices_mask_fct(self):
        slices = get_slices(self.cluster_pos0, self.S, self.data)
        slices_d =  get_slices_d(self.cluster_pos0, slices, self.S, self.M, self.data)
        mat2, _ = get_slices_mask(slices_d, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        self.assertTrue((self.test_slices_mask == mat2).all())

    def test_dis_fct(self):
        mat = self.dis
        slices = get_slices(self.cluster_pos0, self.S, self.data)
        slices_d =  get_slices_d(self.cluster_pos0, slices, self.S, self.M, self.data)
        _, mat2 = get_slices_mask(slices_d, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        np.testing.assert_allclose(mat, mat2, rtol=1e-6, atol=1e-6)

    def test_final_labels_fct(self):
        mat = self.test_final_labels
        slices = get_slices(self.cluster_pos0, self.S, self.data)
        slices_d =  get_slices_d(self.cluster_pos0, slices, self.S, self.M, self.data)
        slices_mask, final_dis = get_slices_mask(slices_d, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        mat2 = get_final_labels(slices_mask, final_dis, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border, self.D)
        np.testing.assert_allclose(mat, mat2, rtol=1e-6, atol=1e-6)

    def test_new_clusters_fct(self):
        pos = self.clusters_pos
        slices = get_slices(self.cluster_pos0, self.S, self.data)
        slices_d = get_slices_d(self.cluster_pos0, slices, self.S, self.M, self.data)
        slices_mask, final_dis = get_slices_mask(slices_d, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        labels = get_final_labels(slices_mask, final_dis, self.cluster_pos0, self.S, self.M, self.data, self.image_height, self.image_width, self.border, self.D)
        pos2 = get_new_clusters(labels, self.cluster_pos0)
        np.testing.assert_allclose(pos, pos2, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
