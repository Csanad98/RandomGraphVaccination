from typing import List

import networkx as nx
import numpy as np
from networkx import is_graphical
from networkx.utils import powerlaw_sequence


def generate_power_law_degree_seq(n: int, p: float, seed: int = 0):
    # while True:  # Continue generating sequences until one of them is graphical
    #     # Round to nearest integer to obtain DISCRETE degree sequence
    #     seq = [int(round(d)) for d in powerlaw_sequence(n=n, exponent=tau, seed=seed)]
    #     if is_graphical(seq):
    #         break

    seq = [d for n, d in nx.erdos_renyi_graph(n=n, p=p, seed=seed).degree()]
    return np.array(seq)


def generate_power_law_degree_seq_community(community_sizes: List[int], tau: float, seed: int = 0):
    deg_seq = []
    for size in community_sizes:
        community_deg_seq = [size-1 for _ in range(size)]
        deg_seq += community_deg_seq
    return np.array(deg_seq)
