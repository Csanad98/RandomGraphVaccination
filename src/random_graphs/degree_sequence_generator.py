from typing import List, Callable
import numpy as np
from networkx.utils import powerlaw_sequence


def generate_power_law_degree_seq(n: int, tau: float, seed: int = 0):
    # Note: the resulting sequence may or may not be graphical, but we use CM, which works with self-loops and multi
    # edges, so it's not an issue.
    seq = [int(round(d)) for d in powerlaw_sequence(n=n, exponent=tau, seed=seed)]
    return correct_deg_sum_to_be_even(seq)


def correct_deg_sum_to_be_even(seq: np.array):
    # make sure degrees sum to an even number
    if np.sum(seq) % 2 != 0:
        seq[0] += 1
    return seq


def generate_poisson_degree_seq(n: int, lam: float, seed: int = 0):
    np.random.seed(seed)
    seq = np.random.poisson(lam=lam, size=n)
    return correct_deg_sum_to_be_even(seq)


def generate_community_degree_seq(seq_generator: Callable, community_sizes: List[int], gen_param: float, seed: int = 0):
    deg_seq = np.array([])
    for size in community_sizes:
        community_deg_seq = seq_generator(size, gen_param, seed)
        deg_seq = np.concatenate((deg_seq, community_deg_seq)) if deg_seq.size else community_deg_seq
    return np.array(deg_seq)
