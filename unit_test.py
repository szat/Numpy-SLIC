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
# from slic_fcts import *

def get_triangle_coordinates(rows, cols, L, frame):
    col_nb = (int)((cols - 2 * frame - L / 2) / L)
    col_coord = np.linspace(frame, cols - frame - 1 - L / 2, col_nb)
    coordinates = []
    row_nb = (int)((rows - 2 * frame) / L)
    row_coord = np.linspace(frame, rows - frame - 1, row_nb)
    for i, r in enumerate(row_coord):
        for c in col_coord:
            if i % 2 == 0:
                coordinates.append((int(r), int(c + L / 2)))
            else:
                coordinates.append((int(r), int(c)))
    return coordinates


def get_grid_coordinates(rows, cols, L, frame):
    col_nb = (int)((cols - 2 * frame - L / 2) / L)
    col_coord = np.linspace(frame, cols - frame - 1, col_nb)
    coordinates = []
    row_nb = (int)((rows - 2 * frame) / L)
    row_coord = np.linspace(frame, rows - frame - 1, row_nb)
    for i, r in enumerate(row_coord):
        for c in col_coord:
            coordinates.append((int(r), int(c)))
    return coordinates


class Cluster(object):
    cluster_index = 0

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()

    def clear(self):
        self.pixels = []


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

    def get_slices(self, clusters, S, data):
        # mat = np.ones((len(self.clusters), 4 * self.S, 4 * self.S, 5), dtype='float32')
        # for i, cluster in enumerate(self.clusters):
        #     for h in range(0, 4 * self.S):
        #         for w in range(0, 4 * self.S):
        #             h_ = h + (cluster.h - 2 * self.S)
        #             w_ = w + (cluster.w - 2 * self.S)
        #             L, A, B = self.data[h_][w_]
        #             mat[i, h, w] = [h_, w_, L, A, B]
        p = np.empty((len(clusters), 2))
        for i, cluster in enumerate(clusters):
            p[i] = [cluster.h, cluster.w]

        mat2 = np.ones((len(clusters), 4 * S, 4 * S, 5), dtype='float32')
        xv, yv = np.meshgrid(np.arange(0, 4 * S), np.arange(0, 4 * S), indexing='ij')
        for i in range(len(clusters)):
            temp_x = xv + p[i, 0] - 2 * S
            temp_y = yv + p[i, 1] - 2 * S
            temp_x = temp_x.astype(int)
            temp_y = temp_y.astype(int)
            mat2[i, :, :, 0] = temp_x
            mat2[i, :, :, 1] = temp_y
            mat2[i, :, :, 2:] = data[temp_x, temp_y]
        return mat2

    def get_slices_d(self, clusters, S, M, data):
        slices = self.get_slices(clusters, S, data)
        # mat = np.empty((len(self.clusters), 4 * self.S, 4 * self.S), dtype='float32')
        # matDc = np.empty((len(self.clusters), 4 * self.S, 4 * self.S), dtype='float32')
        # matDs = np.empty((len(self.clusters), 4 * self.S, 4 * self.S), dtype='float32')
        # for i, cluster in enumerate(self.clusters):
        #     for h in range(0, 4 * self.S):
        #         for w in range(0, 4 * self.S):
        #             h_, w_, L, A, B = slices[i, h, w]
        #             Dc = math.sqrt(
        #                 math.pow(cluster.l - L, 2) +
        #                 math.pow(cluster.a - A, 2) +
        #                 math.pow(cluster.b - B, 2))
        #             matDc[i,h,w] = Dc
        #             Ds = math.sqrt(
        #                 math.pow(cluster.h - h_, 2) +
        #                 math.pow(cluster.w - w_, 2))
        #             matDs[i,h,w] = Ds
        #             mat[i, h, w] = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))

        cluster_pos = np.empty((len(clusters), 2))
        for i, cluster in enumerate(clusters):
            cluster_pos[i] = [cluster.h, cluster.w]

        clusters = np.empty((len(clusters), 5), dtype='float32')
        clusters[:, 0] = cluster_pos[:, 0]
        clusters[:, 1] = cluster_pos[:, 1]
        cluster_pos = cluster_pos.astype(int)
        clusters[:, 2:] = data[cluster_pos[:, 0], cluster_pos[:, 1]]

        temp = copy.deepcopy(slices)
        temp = temp.astype(float)
        slices_dist = np.empty((len(clusters), 4 * S, 4 * S), dtype='float32')
        for i, cluster in enumerate(clusters):
            Ds = np.linalg.norm((temp[i, :, :, 0:2] - clusters[i, 0:2]), axis=2) / S
            Dc = np.linalg.norm((temp[i, :, :, 2:] - clusters[i, 2:]), axis=2) / M
            slices_dist[i] = np.power(np.power(Ds, 2) + np.power(Dc, 2), 0.5)

        return slices_dist

    def get_slices_mask(self, clusters, S, M, data, image_height, image_width, border):
        slices_d = self.get_slices_d(clusters, S, M, data)
        mat = np.ones((len(clusters), 4 * S, 4 * S), dtype=bool) * False
        temp_dis = np.full((image_height, image_width), -1.0)
        temp_dis[border:image_height - border, border:image_width - border] = np.inf
        # temp_D = np.full((self.image_height, self.image_width), -1.0)
        # temp_D[self.border:self.image_height - self.border, self.border:self.image_width - self.border] = np.inf
        # for i, cluster in enumerate(self.clusters):
        #     temp_D = slices_d[i]
        #     for h in range(0, 4 * self.S):
        #         for w in range(0, 4 * self.S):
        #             h_ = h + (cluster.h - 2 * self.S)
        #             w_ = w + (cluster.w - 2 * self.S)
        #             if temp_D[h, w] < temp_dis[h_][w_]:
        #                 mat[i, h, w] = True
        #                 temp_dis[h_][w_] = temp_D[h, w]


        temp_mat = np.ones((len(clusters), 4 * S, 4 * S), dtype=bool) * False
        temp_dis2 = np.full((image_height, image_width), -1.0)
        temp_dis2[border:image_height - border, border:image_width - border] = np.inf
        dis_slices = np.empty((len(clusters), 4 * S, 4 * S))

        cluster_pos = np.empty((len(clusters), 2))
        for i, cluster in enumerate(clusters):
            cluster_pos[i] = [cluster.h, cluster.w]

        xv, yv = np.meshgrid(np.arange(0, 4 * S), np.arange(0, 4 * S), indexing='ij')

        for i in range(len(clusters)):
            temp_x = xv + cluster_pos[i, 0] - 2 * S
            temp_y = yv + cluster_pos[i, 1] - 2 * S
            temp_x = temp_x.astype(int)
            temp_y = temp_y.astype(int)
            mask = slices_d[i] < temp_dis2[temp_x, temp_y]
            mat[i] = mask
            x0 = int(cluster_pos[i, 0] - 2 * S)
            x1 = int(cluster_pos[i, 0] + 2 * S)
            y0 = int(cluster_pos[i, 1] - 2 * S)
            y1 = int(cluster_pos[i, 1] + 2 * S)
            temp_dis2[x0:x1, y0:y1][mask] = slices_d[i][mask]
        #
        #
        # for i, cluster in enumerate(self.clusters):
        #     mask = slices_d[i] < dis_slices[i]
        #     mat[i] = mask
        #     slices_d[i][mask]
            # for h in range(0, 4 * self.S):
            #     for w in range(0, 4 * self.S):
            #         h_ = h + (cluster.h - 2 * self.S)
            #         w_ = w + (cluster.w - 2 * self.S)
            #         if temp_D[h, w] < temp_dis[h_][w_]:
            #             mat[i, h, w] = True
            #             temp_dis[h_][w_] = temp_D[h, w]

        return mat, temp_dis2

    def get_final_labels(self, clusters, S, M, data, image_height, image_width, border, D):
        slices_mask, final_dis = self.get_slices_mask(clusters, S, M, data, image_height, image_width, border)
        temp = np.ones(D.shape) * -1
        mask2 = np.full(D.shape, False, dtype=bool)
        for i, cluster in enumerate(clusters):
            mask = slices_mask[i]
            mask2[cluster.h - 2 * S:cluster.h + 2 * S, cluster.w - 2 * S:cluster.w + 2 * S] = mask
            temp[mask2] = int(i)
            # Reset
            mask2[cluster.h - 2 * S:cluster.h + 2 * S, cluster.w - 2 * S:cluster.w + 2 * S] = False
        return temp

    def get_new_clusters(self, labels, clusters):
        # labels = self.get_final_labels(clusters, S, M, data, image_height, image_width, border, D)
        # temp = np.empty((len(self.new_clusters), 2), dtype=float)
        temp = np.empty((len(clusters), 2), dtype=float)
        # sub_labels = labels[self.border:self.image_height - self.border, self.border:self.image_width - self.border]
        # for i in range(len(self.new_clusters)):
        for i in range(len(clusters)):
            mask = labels == i
            center_of_mass = np.mean(np.where(mask), axis=1, dtype=int)
            temp[i, :] = center_of_mass
        return temp

    def test_slices_fct(self):
        np.testing.assert_allclose(self.test_slices, self.get_slices(self.clusters, self.S, self.data))

    def test_slices_d_fct(self):
        np.testing.assert_allclose(self.test_slices_d, self.get_slices_d(self.clusters, self.S, self.M, self.data), rtol=1e-6, atol=1e-6)

    def test_slices_mask_fct(self):
        mat2, _ = self.get_slices_mask(self.clusters, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        self.assertTrue((self.test_slices_mask == mat2).all())

    def test_dis_fct(self):
        mat = self.dis
        _, mat2 = self.get_slices_mask(self.clusters, self.S, self.M, self.data, self.image_height, self.image_width, self.border)
        np.testing.assert_allclose(mat, mat2, rtol=1e-6, atol=1e-6)

    def test_final_labels_fct(self):
        mat = self.test_final_labels
        mat2 = self.get_final_labels(self.clusters, self.S, self.M, self.data, self.image_height, self.image_width, self.border, self.D)
        np.testing.assert_allclose(mat, mat2, rtol=1e-6, atol=1e-6)

    def test_new_clusters_fct(self):
        pos = self.clusters_pos
        labels = self.get_final_labels(self.clusters, self.S, self.M, self.data, self.image_height, self.image_width, self.border, self.D)
        pos2 = self.get_new_clusters(labels, self.clusters)
        np.testing.assert_allclose(pos, pos2, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
