from typing import List, Dict, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random

from utils import create_community_random_color_map, community_map_from_community_sizes

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq


def hierarchical_configuration_model(deg_seq_in: np.array,
                                     deg_seq_out: np.array,
                                     communities: np.array,
                                     seed=0):
    """

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

    # Run configuration model for each community, use degree sequence meant for within communities
    for c in np.unique(communities):
        # get vertex ids for current community
        vertex_ids_for_c = np.where(communities == c)[0]  # index 0 since we have only one dimension
        # call nx.Graph to get a (non-erased) Configuration Model Multigraph
        community_sub_graph = nx.configuration_model(deg_sequence=deg_seq_in[vertex_ids_for_c], seed=seed)
        community_sub_graph.remove_edges_from(nx.selfloop_edges(community_sub_graph))  # remove self loops
        # Remove Parallel Edges by turning the Multigraph into a Graph
        community_sub_graph = nx.Graph(community_sub_graph)
        full_graph = nx.disjoint_union(full_graph, community_sub_graph)

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
        first = random.randint(0, len(half_edge_indices_of_community_with_most_hes)-1)
        first_he_id = half_edge_indices_of_community_with_most_hes[first]
        first_community = half_edges[first_he_id][1]
        first_vertex_id = half_edges[first_he_id][0]

        half_edge_indices_of_other_communities = np.where(half_edges[:, 1] != first_community)[0]

        second = random.randint(0, len(half_edge_indices_of_other_communities)-1)
        second_he_id = half_edge_indices_of_other_communities[second]
        second_vertex_id = half_edges[second_he_id][0]

        full_graph.add_edge(first_vertex_id, second_vertex_id)

        # remove matched half edges from h.e. list
        half_edges = np.delete(half_edges, [first_he_id, second_he_id], axis=0)

    return full_graph

def create_half_edges_between_communities(deg_seq_out, communities):
    num_half_edges = np.sum(deg_seq_out)
    half_edges = np.zeros((num_half_edges, 2))
    half_edge_index = 0

    # setup half edge map
    for v_index in range(len(deg_seq_out)):
        for h in range(deg_seq_out[v_index]):
            half_edges[half_edge_index][0] = v_index  # first item in the tuple indicates vertex index of h.e.
            # 2nd item in the tuple indicates community index of h.e.
            half_edges[half_edge_index][1] = communities[v_index]
            half_edge_index += 1
    return half_edges


def is_community_structure_possible(deg_seq_out, communities):
    # for each community the number of out h.e.-s cannot exceed the total number of h.e.-s from every other community
    # half_edges = create_half_edges_between_communities(deg_seq_out=deg_seq_out, communities=communities)
    community_ids = np.unique(communities)
    for c in community_ids:
        vertex_ids_in_c = np.where(communities == c)[0]
        vertex_ids_not_in_c = np.where(communities != c)[0]
        assert np.sum(deg_seq_out[vertex_ids_in_c.astype(int)]) <= np.sum(deg_seq_out[vertex_ids_not_in_c.astype(int)]), \
            "for each community the number of out h.e.-s cannot exceed the total number of h.e.-s from every other " \
            "community"


if "__main__" == __name__:
    seed = 1
    community_sizes = [10, 15, 11, 12, 30, 22]
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
    g = hierarchical_configuration_model(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    pos = nx.spring_layout(g, seed=seed)  # Seed layout for reproducibility
    nx.draw_spring(g, with_labels=True, node_color=color_map)
    plt.show()
