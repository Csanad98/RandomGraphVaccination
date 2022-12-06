from typing import List

import numpy as np
from matplotlib import colors, pylab


def create_community_color_map(communities: np.array) -> np.array:
    """
    Assigns a color to each node, such that nodes within one community have the same color.
    Note, method is adapted from:
    https://stackoverflow.com/questions/33596491/extract-matplotlib-colormap-in-hex-format/33597599#33597599
    :param communities: 1d array of integers, ith value indicates the community id of the ith node
    :return: color_map: a list of colors
    """
    n = communities.shape[0]
    cmap = pylab.cm.get_cmap('hsv', n)
    color_list = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    community_cmap = np.array([color_list[i] for i in communities])
    return community_cmap


def community_map_from_community_sizes(community_sizes: List[int]):
    communities = []
    i = 0
    for size in community_sizes:
        for j in range(size):
            communities.append(i)
        i += 1
    return np.array(communities)
