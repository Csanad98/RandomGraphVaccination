import networkx as nx
import numpy as np


#for testing
from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1

from matplotlib import pyplot as plt

from utils import create_community_random_color_map, community_map_from_community_sizes, community_sizes_generator

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq
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
    n = 700
    community_sizes = community_sizes_generator(n=n, prop_lr_com_size=0.45, seed=seed)
    tau = 2.8
    p = 0.05
    lam = 15
    # proportion of high risk individuals in high risk groups
    prop_hr_hr = 0.7
    # proportion of high risk individuals in low risk groups
    prop_hr_lr = 0
    # Generate sequences of degree and community distributions
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    color_map = create_community_random_color_map(communities)
    # generate hierarchical configuration model
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    pos = nx.spring_layout(g, seed=seed)  # Seed layout for reproducibility
    nx.draw_spring(g, with_labels=False, width=0.3, edgecolors="k", alpha=0.9, node_color=color_map, node_size = 100)
    # plot graph
    plt.show()
    g = attr_assign(g=g,
                    deg_seq_out=deg_seq_out,
                    deg_seq_in=deg_seq_in,
                    communities=communities,
                    prop_hr_hr=prop_hr_hr,
                    prop_hr_lr=prop_hr_lr,
                    seed=seed)
    nx.set_node_attributes(g, {1: 1, 3: 1, 4: 1, 10: 1, 100: 1, 200: 1, 300: 1, 400: 1, 500: 1, 604: 1, 640: 1, 603: 1,
                               622: 1, 677: 1, 690: 1, 688: 1}, 'health')

    for i in range(0, 365):
        g = time_step_simulation(g, seed)
        seed += 1
    print(list(nx.get_node_attributes(g, "health").values()).count(-2))
    print(list(nx.get_node_attributes(g, "health").values()).count(-1))
