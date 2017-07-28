from disjoint_set import *
import math
import numpy as np
import random


# ---------------------------------------------------------
# Segment a graph:
# Returns a disjoint-set forest representing the segmentation.
#
# Inputs:
#           num_vertices: number of vertices in graph.
#           num_edges: number of edges in graph
#           edges: array of edges.
#           c: constant for threshold function.
#
# Output:
#           a disjoint-set forest representing the segmentation.
# ------------------------------------------------------------
def segment_graph(num_vertices, num_edges, edges, c):
    # sort edges by weight (3rd column)
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    # make a disjoint-set forest
    u = universe(num_vertices)
    # init thresholds
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)

    # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        pedge = edges[i, :]

        # components connected by this edge
        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)

    return u


def get_threshold(size, c):
    return c / size


# returns square of a number
def square(value):
    return value * value


# randomly creates RGB
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb


# dissimilarity measure between pixels
def diff(red_band, green_band, blue_band, x1, y1, x2, y2):
    result = math.sqrt(
        square(red_band[y1, x1] - red_band[y2, x2]) + square(green_band[y1, x1] - green_band[y2, x2]) + square(
            blue_band[y1, x1] - blue_band[y2, x2]))
    return result
