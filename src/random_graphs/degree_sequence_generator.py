from typing import List
import numpy as np
from networkx.utils import powerlaw_sequence


def generate_power_law_degree_seq(n: int, tau: float, seed: int = 0):
    # Note: the resulting sequence may or may not be graphical, but we use CM, which works with self-loops and multi
    # edges, so it's not an issue.
    seq = [int(round(d)) for d in powerlaw_sequence(n=n, exponent=tau, seed=seed)]

    # make sure degrees sum to an even number
    if np.sum(seq) % 2 != 0:
        seq[0] += 1
    return seq


def generate_power_law_degree_seq_community(community_sizes: List[int], tau: float, seed: int = 0):
    deg_seq = []
    for size in community_sizes:
        community_deg_seq = generate_power_law_degree_seq(n=size, tau=tau, seed=seed)
        deg_seq += community_deg_seq
    return np.array(deg_seq)
