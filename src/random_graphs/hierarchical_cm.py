from typing import List, Dict, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import random


def hierarchical_configuration_model(deg_seq_in: List[int],
                                     deg_seq_out: List[int],
                                     communities: List[int],
                                     seed=0):
    """

    :param deg_seq_in: a list of integers, ith item denotes the degree of the ith vertex within its community
    :param deg_seq_out: a list of integers, ith item denotes the degree of the ith vertex matching to other communities
    :param communities: a list of integers, ith item denotes the community id of the ith vertex
    :param seed:
    :return:
    """

    assert len(deg_seq_in) == len(deg_seq_out)

    # Run configuration model for each community, use degree sequence meant for within communities
    community_sub_graphs = {}
    for c in range(len(np.unique(communities))):
        # get vertex ids for current community
        vertex_ids_for_c = np.where(communities == c)
        community_sub_graphs[c] = nx.configuration_model(deg_sequence=deg_seq_in[vertex_ids_for_c], seed=seed)

    random.seed(seed)
    # algo 3: choose a remaining half edge uniformly, then choose another half edge uniformly, match them, update
    # remaining half edges
    # addition for hierarchical CM: choose pair h.e. uniformly from any other community

    # create list for half edges:
    # for each h.e. we need to know:
    # - its vertex
    # - its community
    # map id to a tuple of vertex id and community id
    # should be easy to remove h.e.
    half_edges: List[Tuple[int, int]] = []
    id = 0
    for i in range(len(deg_seq_out)):
        for j in range(deg_seq_out[i]):
            half_edges[id] = (i, communities)

    random.randint()


if "__main__" == __name__:
    seed = 0
    deg_seq_in = [5, 3, 3, 3]
    deg_seq_out = []
    community = []
    g = hierarchical_configuration_model(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=community)
    pos = nx.spring_layout(g, seed=seed)  # Seed layout for reproducibility
    nx.draw(g, pos=pos)
    plt.show()