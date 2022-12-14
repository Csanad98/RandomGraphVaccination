import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, generate_community_degree_seq, \
    generate_poisson_degree_seq
from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1, hierarchical_configuration_model_algo2
from utils import community_map_from_community_sizes, create_community_random_color_map


def get_hcm_degree_distributions(hcm_g: nx.Graph, communities: np.array):
    """
    Gives the realized degree distributions of HCM, within each community and between the communities.
    :param hcm_g:
    :param deg_seq_in:
    :param deg_seq_out:
    :param communities:
    :return:
    """
    comm_ids = np.unique(communities)
    c_deg_sequences = []
    edges_within_cs = []
    # for each community get their degree sequence
    for c in comm_ids:
        # get vertex ids for current community
        vertex_ids_for_c = np.where(communities == c)[0]
        c_subgraph = hcm_g.subgraph(vertex_ids_for_c)
        c_deg_sequence = sorted((d for n, d in c_subgraph.degree()), reverse=True)
        c_deg_sequences.append(c_deg_sequence)

        # collect edges within the community - to be removed in the next part
        c_edges = list(c_subgraph.edges)
        edges_within_cs += c_edges

    # get inter community degree sequence
    # 1st remove all edges within communities:
    hcm_g.remove_edges_from(edges_within_cs)
    b_deg_sequence = sorted((d for n, d in hcm_g.degree()), reverse=True)
    return b_deg_sequence, c_deg_sequences


if "__main__" == __name__:
    seed = 1
    random.seed(seed)
    community_sizes = [random.randint(5, 15) for _ in range(100)]
    tau = 2.8
    lam = 15
    n = sum(community_sizes)
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)

    b_deg_dist, c_deg_dists = get_hcm_degree_distributions(hcm_g=g, communities=communities)
    fig, ax = plt.subplots()
    deg_seq_out = sorted(deg_seq_out, reverse=True)

    ax.plot(b_deg_dist, color='green', label='Realized degree sequence')
    ax.plot(deg_seq_out, color='red', label='True degree sequence')
    ax.legend(loc='upper left')
    plt.show()




