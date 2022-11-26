from typing import List, Dict, Tuple

import networkx as nx
from matplotlib import pyplot as plt
import random


def hierarchical_configuration_model(deg_seq_in: List[int],
                                     deg_seq_out: List[int],
                                     communities: Dict[int:List[int]],
                                     seed=0):
    """

    :param deg_seq_in: a list of integers, ith item denotes the degree of the ith vertex within its community
    :param deg_seq_out: a list of integers, ith item denotes the degree of the ith vertex matching to other communities
    :param communities: Dictionary mapping community ids to list of vertices (disjoint communities)
    :param seed:
    :return:
    """

    assert len(deg_seq_in) == len(deg_seq_out)
    community_sub_graphs = {}
    for k, v in communities:
        community_sub_graphs[k] = nx.configuration_model(deg_sequence=deg_seq_in[v], seed=seed)

    random.seed(seed)
    # algo 3: choose a remaining half edge uniformly, then choose another half edge uniformly, mathc them, update
    # remaining half edges
    # addition for hierarchical CM: choose pair h.e. uniformly from any other community

    # create ids for half edges:
    # for each h.e. we need to know:
    # - which vertex it belongs
    # - which community it belongs
    # map id to a tuple of vertex id and community id
    half_edges: Dict[int: Tuple[int, int]] = {}
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