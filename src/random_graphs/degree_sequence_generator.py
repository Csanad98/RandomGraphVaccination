from typing import List

import networkx as nx
import numpy as np


def generate_power_law_degree_seq(n: int, tau: float):
    deg_seq = nx.random_powerlaw_tree_sequence(n=n, gamma=tau)
    return np.array(deg_seq)


def generate_power_law_degree_seq_community(community_sizes: List[int], tau: float):
    deg_seq = []
    for size in community_sizes:
        community_deg_seq = nx.random_powerlaw_tree_sequence(n=size, gamma=tau)
        deg_seq += community_deg_seq
    return np.array(deg_seq)
