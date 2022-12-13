from typing import List

import numpy as np


def create_community_random_color_map(communities: np.array, seed: int = 0) -> np.array:
    """
    Assigns a random color to each node, such that nodes within one community have the same color.
    :param seed:
    :param communities: 1d array of integers, ith value indicates the community id of the ith node
    :return: color_map: a list of colors
    """
    np.random.seed(seed)
    n = communities.shape[0]
    color_list = [np.random.rand(3,) for _ in range(n)]
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


def correct_deg_sum_to_be_even(seq: np.array):
    # make sure degrees sum to an even number
    if np.sum(seq) % 2 != 0:
        seq[0] += 1
    return seq
