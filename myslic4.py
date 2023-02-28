import cv2
import math
import numpy as np
from skimage import io, color
from numpy.testing import assert_almost_equal
import copy
from slic_fcts import *


K = 500
M = 60
data_for_color = io.imread('parrot0.jpg')
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
data_for_color = cv2.copyMakeBorder(data_for_color, border, border, border, border, cv2.BORDER_CONSTANT)
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

slices = get_slices(cluster_pos, S, data)
slices_d = get_slices_d(cluster_pos, slices, S, M, data)
slices_mask, final_dis = get_slices_mask(slices_d, cluster_pos, S, image_height, image_width, border)
test_final_labels2 = get_final_labels(slices_mask, cluster_pos, S, D)
new_clusters = get_new_clusters(test_final_labels2, cluster_pos)

cluster_pos = new_clusters

# Visualize
image_arr = np.copy(data)
colors_temp = np.empty((len(clusters), 3))
cluster_viz = cluster_pos - border
for i in range(len(cluster_pos)):
    colors_temp[i, :] = data_for_color[int(cluster_pos[i, 0]), int(cluster_pos[i, 1]), :]

temp = test_final_labels2[border:-border, border:-border]
temp = temp.astype(int)
image_arr = image_arr[border:-border, border:-border]
image_arr = colors_temp[temp, :]

image_arr = image_arr.astype(dtype=np.uint8)

