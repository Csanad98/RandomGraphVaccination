import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random

from random_graphs.hcm_utils import create_half_edges_between_communities, is_community_structure_possible, \
    cm_for_communities
from utils import create_community_random_color_map, community_map_from_community_sizes

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq


def hierarchical_configuration_model_algo1(deg_seq_in: np.array,
                                           deg_seq_out: np.array,
                                           communities: np.array,
                                           seed=0):
    """
    Generates HCM model, suitable for lower number of communities.
    :param deg_seq_in: a list of integers, ith item denotes the degree of the ith vertex within its community
    :param deg_seq_out: a list of integers, ith item denotes the degree of the ith vertex matching to other communities
    :param communities: a list of integers, ith item denotes the community id of the ith vertex
    :param seed:
    :return:
    """

    assert len(deg_seq_in) == len(deg_seq_out)
    is_community_structure_possible(np.array(deg_seq_out), communities)
    full_graph = nx.Graph()
    random.seed(seed)

    # create communities according to degree distributions within communities
    full_graph = cm_for_communities(deg_seq_in=deg_seq_in, communities=communities, graph=full_graph, seed=0)

    # algo 3: choose a remaining half edge uniformly, then choose another half edge uniformly, match them, update
    # remaining half edges
    # addition for hierarchical CM: choose 2nd h.e. uniformly from any other community

    half_edges = create_half_edges_between_communities(deg_seq_out, communities)

    # while we have half-edges left to match:
    while half_edges.shape[0] != 0:
        # count which community has the most h.e.-s
        comm_he_count = np.bincount(half_edges[:, 1].astype(int))
        # pick first half edge uniformly within community with most h.e.-s
        comm_most_he_index = np.argmax(comm_he_count)
        half_edge_indices_of_community_with_most_hes = np.where(half_edges[:, 1] == comm_most_he_index)[0]

        # pick uniform h.e. within the community with most h.e.-s
        first = random.randint(0, len(half_edge_indices_of_community_with_most_hes) - 1)
        first_he_id = half_edge_indices_of_community_with_most_hes[first]
        first_community = half_edges[first_he_id][1]
        first_vertex_id = half_edges[first_he_id][0]

        half_edge_indices_of_other_communities = np.where(half_edges[:, 1] != first_community)[0]

        second = random.randint(0, len(half_edge_indices_of_other_communities) - 1)
        second_he_id = half_edge_indices_of_other_communities[second]
        second_vertex_id = half_edges[second_he_id][0]

        full_graph.add_edge(first_vertex_id, second_vertex_id)

        # remove matched half edges from h.e. list
        half_edges = np.delete(half_edges, [first_he_id, second_he_id], axis=0)

    return full_graph


def hierarchical_configuration_model_algo2(deg_seq_in: np.array,
                                           deg_seq_out: np.array,
                                           communities: np.array,
                                           seed=0):
    """
    Assumes a high number of communities. Allows self loops back to communities from deg_seq_out.
    :param deg_seq_in:
    :param deg_seq_out:
    :param communities:
    :param seed:
    :return:
    """
    assert len(deg_seq_in) == len(deg_seq_out)
    is_community_structure_possible(np.array(deg_seq_out), communities)
    full_graph = nx.Graph()
    random.seed(seed)

    # create communities according to degree distributions within communities
    full_graph = cm_for_communities(deg_seq_in=deg_seq_in, communities=communities, graph=full_graph, seed=0)

    half_edges = create_half_edges_between_communities(deg_seq_out, communities)

    # match all inter community half edges uniformly via algo 1: permutation, then pair them
    matchings = np.random.permutation(half_edges)

    for m in matchings:
        full_graph.add_edge(m[0], m[1])

    return full_graph


if "__main__" == __name__:
    seed = 1
    random.seed(seed)
    community_sizes = [random.randint(5, 15) for _ in range(10)]
    tau = 2.8
    p = 0.05
    lam = 15
    n = sum(community_sizes)
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    color_map = create_community_random_color_map(communities)
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    pos = nx.kamada_kawai_layout(g)  # Seed layout for reproducibility
    nx.draw_spring(g, with_labels=True, node_color=color_map)
    plt.show()
