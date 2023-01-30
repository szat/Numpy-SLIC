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

cluster_pos = np.empty((len(clusters), 2))
for i, cluster in enumerate(clusters):
    cluster_pos[i] = [cluster.h, cluster.w]

label = {}
dis = np.full((image_height, image_width), -1.0)
dis[border:image_height - border, border:image_width - border] = np.inf
D = np.full((image_height, image_width), -1.0)
D[border:image_height - border, border:image_width - border] = np.inf
LABEL = np.full((image_height, image_width), -1.0)
clusters_temp = np.empty((clusters[0].cluster_index, 3))

test_final_labels2 = get_final_labels(clusters, S, M, data, image_height, image_width, border, D)
new_clusters = get_new_clusters(test_final_labels2, clusters)


image_arr = np.copy(data)
colors_temp = np.empty((len(clusters), 3))
for c in clusters:
    colors_temp[c.no, :] = [c.l, c.a, c.b]

temp = test_final_labels2[border:-border, border:-border]
temp = temp.astype(int)
image_arr = image_arr[border:-border, border:-border]
image_arr = colors_temp[temp, :]
image_arr = color.lab2rgb(image_arr)
image_arr = image_arr * 255
image_arr = image_arr.astype(dtype=np.uint8)

