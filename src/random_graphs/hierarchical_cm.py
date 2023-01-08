import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random

from random_graphs.hcm_utils import create_half_edges_between_communities, is_community_structure_possible, \
    cm_for_communities, get_half_edges
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
    while sum([len(x) for x in half_edges]) != 0:
        first_community = np.argmax([len(x) for x in half_edges])

        # pick uniform h.e. within the community with most h.e.-s
        first = random.randint(0, len(half_edges[first_community]) - 1)
        first_id = half_edges[first_community][first]

        # pick uniform h.r. outside the community
        leftover_he = half_edges[:first_community] + half_edges[first_community + 1:]
        second_big = random.randint(0, len(sum(leftover_he, [])) - 1)

        # find in which community it belongs to
        second_community = 0
        checker = len(leftover_he[second_community])
        while checker - 1 < second_big:
            second_community += 1
            checker += len(leftover_he[second_community])
        # index of second he inside its community list
        second = second_big - checker + len(leftover_he[second_community])
        # second he
        second_id = leftover_he[second_community][second]

        # if the second community was later than the first one then add one
        if second_big >= len(sum(half_edges[:first_community], [])):
            second_community += 1

        # add the edge
        full_graph.add_edge(first_id, second_id)

        # remove matched half edges from h.e. list
        del half_edges[first_community][first]
        del half_edges[second_community][second]

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


def hierarchical_configuration_model_algo3(deg_seq_in: np.array,
                                           deg_seq_out: np.array,
                                           communities: np.array,
                                           seed=0):
    """
    No "community" for general population. Matches general population in and out degrees both with community out edges
     and general pop edges.
    :param deg_seq_in:
    :param deg_seq_out:
    :param communities:
    :param seed:
    :return:
    """
    assert len(deg_seq_in) == len(deg_seq_out)
    full_graph = nx.Graph()
    random.seed(seed)
    np.random.seed(seed)

    # create communities according to degree distributions within communities
    full_graph = cm_for_communities(deg_seq_in=deg_seq_in, communities=communities, graph=full_graph,
                                    community_for_general_pop=False, seed=0)

    half_edges = get_half_edges(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)

    # match all inter community half edges uniformly via algo 1: permutation, then pair them
    matchings = np.random.permutation(half_edges)

    for i in range(0, len(matchings)-1, 2):
        full_graph.add_edge(matchings[i], matchings[i+1])

    return full_graph


if "__main__" == __name__:
    seed = 1
    random.seed(seed)
    community_sizes = [random.randint(5, 15) for _ in range(20)]
    tau = 2.8
    p = 0.05
    lam = 3
    n = sum(community_sizes)
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    color_map = create_community_random_color_map(communities)
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    pos = nx.kamada_kawai_layout(g)  # Seed layout for reproducibility
    nx.draw_spring(g, with_labels=False, node_color=color_map, node_size=50)
    plt.show()
