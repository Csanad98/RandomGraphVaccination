import networkx as nx
import numpy as np

from random_graphs.hierarchical_cm import hierarchical_configuration_model

from utils import create_community_color_map, community_map_from_community_sizes

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_power_law_degree_seq_community
from node_attributes import attr_assign


def time_step_simulation(g: nx.Graph, seed: int):
    np.random.seed(seed)
    for i in [x for x, y in g.nodes(data=True) if y['health'] > 0]:
        # Check all healthy neighbors of i
        for j in [x for x, y in g.subgraph(list(g.neighbors(i))).nodes(data=True) if y['health'] == 0]:
            # infect the neighbor with probability of node infectivity
            if np.random.binomial(1, nx.get_node_attributes(g, "infectivity")[i]) == 1:
                nx.set_node_attributes(g, {j: 1}, 'health')
        # Check if node is not at the end of illness
        if nx.get_node_attributes(g, "health")[i] < nx.get_node_attributes(g, "outcome")[i]:
            # If not add one day to the health timeline
            nx.set_node_attributes(g, {i: nx.get_node_attributes(g, "health")[i] + 1}, 'health')
        # Otherwise check if outcome is death
        elif nx.get_node_attributes(g, "outcome")[i] == 18:
            nx.set_node_attributes(g, {i: -2}, 'health')
        # Otherwise we have recovery
        else:
            nx.set_node_attributes(g, {i: -1}, 'health')

    return g


if "__main__" == __name__:
    seed = 1
    # deg_seq_in = np.array([1, 3, 3, 3, 4, 4, 4, 4])
    # deg_seq_out = np.array([1, 3, 3, 3, 2, 2, 2, 2])
    # communities = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    community_sizes = [10, 15, 11, 12, 30, 22]
    tau = 2.8
    p = 0.05
    n = sum(community_sizes)
    deg_seq_out = generate_power_law_degree_seq(n=n, p=p)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_power_law_degree_seq_community(community_sizes=community_sizes, tau=tau)
    color_map = create_community_color_map(communities)
    g = hierarchical_configuration_model(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    g = attr_assign(g, deg_seq_out, deg_seq_in, communities, seed)
    nx.set_node_attributes(g, {1: 1, 3: 1, 4: 1, 10: 1, 12: 1}, 'health')

    for i in range(0, 90):
        g = time_step_simulation(g, seed)
        seed += 1
    print(list(nx.get_node_attributes(g, "health").values()).count(-2))
    print(list(nx.get_node_attributes(g, "health").values()).count(-1))