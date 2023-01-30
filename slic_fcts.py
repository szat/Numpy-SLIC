import time
import unittest
# from myslic import *
import cv2
import math
import numpy as np
from skimage import io, color
from numpy.testing import assert_almost_equal
import copy


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


def get_slices(cluster_pos, S, data):
    mat2 = np.ones((len(cluster_pos), 4 * S, 4 * S, 5), dtype='float32')
    xv, yv = np.meshgrid(np.arange(0, 4 * S), np.arange(0, 4 * S), indexing='ij')
    for i in range(len(cluster_pos)):
        temp_x = xv + cluster_pos[i, 0] - 2 * S
        temp_y = yv + cluster_pos[i, 1] - 2 * S
        temp_x = temp_x.astype(int)
        temp_y = temp_y.astype(int)
        mat2[i, :, :, 0] = temp_x
        mat2[i, :, :, 1] = temp_y
        mat2[i, :, :, 2:] = data[temp_x, temp_y]
    return mat2


def get_slices_d(clusters, slices, S, M, data):
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


def get_slices_mask(slices_d, clusters, S, M, data, image_height, image_width, border):
    mat = np.ones((len(clusters), 4 * S, 4 * S), dtype=bool) * False
    temp_dis = np.full((image_height, image_width), -1.0)
    temp_dis[border:image_height - border, border:image_width - border] = np.inf

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
    return mat, temp_dis2


def get_final_labels(slices_mask, final_dis, clusters, S, M, data, image_height, image_width, border, D):
    temp = np.ones(D.shape) * -1
    mask2 = np.full(D.shape, False, dtype=bool)
    for i, cluster in enumerate(clusters):
        mask = slices_mask[i]
        mask2[cluster.h - 2 * S:cluster.h + 2 * S, cluster.w - 2 * S:cluster.w + 2 * S] = mask
        temp[mask2] = int(i)
        # Reset
        mask2[cluster.h - 2 * S:cluster.h + 2 * S, cluster.w - 2 * S:cluster.w + 2 * S] = False
    return temp


def get_new_clusters(labels, clusters):
    # temp = np.empty((len(self.new_clusters), 2), dtype=float)
    temp = np.empty((len(clusters), 2), dtype=float)
    # sub_labels = labels[self.border:self.image_height - self.border, self.border:self.image_width - self.border]
    # for i in range(len(self.new_clusters)):
    for i in range(len(clusters)):
        mask = labels == i
        center_of_mass = np.mean(np.where(mask), axis=1, dtype=int)
        temp[i, :] = center_of_mass
    return temp
