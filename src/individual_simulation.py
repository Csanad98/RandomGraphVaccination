import networkx as nx
import numpy as np

# for testing
from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1

from matplotlib import pyplot as plt

from utils import create_community_random_color_map, community_map_from_community_sizes, \
    community_sizes_generator, init_infected

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq
from node_attributes import attr_assign

from pylab import plot, array


def time_step_simulation(g: nx.Graph, seed: int):
    """
    Simulates one day of the graph. For each infected individual, healthy
    neighbors get infected with respective probability.
    If the infected individual has not reached the day of outcome then their health
    gets larger by one.
    If the infected individual has reached the day of outcome then their health
    changes according to the outcome.
    """
    np.random.seed(seed)
    deaths = {"high_risk": 0, "low_risk": 0}
    recoveries = {"high_risk": 0, "low_risk": 0}
    infections = {"high_risk": 0, "low_risk": 0}
    for i in [x for x, y in g.nodes(data=True) if y['health'] > 0]:
        # Check all healthy neighbors of i
        for j in [x for x, y in g.subgraph(list(g.neighbors(i))).nodes(data=True) if y['health'] == 0]:
            # infect the neighbor with probability of node infectivity
            if np.random.binomial(1, nx.get_node_attributes(g, "infectivity")[i]) == 1:
                nx.set_node_attributes(g, {j: 1}, 'health')
                infections[nx.get_node_attributes(g, "risk_group")[j]] += 1
        # Check if node is not at the end of illness
        if nx.get_node_attributes(g, "health")[i] < nx.get_node_attributes(g, "outcome")[i]:
            # If not add one day to the health timeline
            nx.set_node_attributes(g, {i: nx.get_node_attributes(g, "health")[i] + 1}, 'health')
        # Otherwise check if outcome is death
        elif nx.get_node_attributes(g, "outcome")[i] == 18:
            nx.set_node_attributes(g, {i: -2}, 'health')
            deaths[nx.get_node_attributes(g, "risk_group")[i]] += 1
        # Otherwise we have recovery
        else:
            nx.set_node_attributes(g, {i: -1}, 'health')
            recoveries[nx.get_node_attributes(g, "risk_group")[i]] += 1

    return g, (deaths["high_risk"], deaths["low_risk"]), \
           (recoveries["high_risk"], recoveries["low_risk"]), \
           (infections["high_risk"], infections["low_risk"])


def singe_graph_simulation(seed: int,
                           n: int = 1000,
                           tau: float = 2.8,
                           lam: float = 15,
                           prop_lr_com_size: float = 0.45,
                           prop_int_inf: float = 0.05,
                           prop_int_inf_hr: float = 0.5,
                           prop_hr_hr: float = 0.7,
                           prop_hr_lr: float = 0,
                           n_days: int = 365):
    """
    Creates a graph and simulates n_days days of the graph.
    :param n: number of people
    :param seed: seed
    :param tau: parameter for outward degree distribution
    :param lam: parameter for inter-community degree distribution
    :param prop_lr_com_size: proportion of nodes in the low risk community
    :param prop_int_inf: proportion of initially infected nodes
    :param prop_int_inf_hr: proportion of high risk in the initially infected nodes
    :param prop_hr_hr: proportion of high risk individuals in high risk groups
    :param prop_hr_lr: proportion of high risk individuals in low risk groups
    :param n_days: number of days simulated
    :return:
    """

    # generate sequence of community sizes
    community_sizes = community_sizes_generator(n=n, prop_lr_com_size=prop_lr_com_size, seed=seed)
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
    nx.draw_spring(g, with_labels=False, width=0.3, edgecolors="k", alpha=0.9, node_color=color_map, node_size=100)
    # plot graph
    plt.show()

    # assign attributes to graph nodes
    g = attr_assign(g=g,
                    deg_seq_out=deg_seq_out,
                    deg_seq_in=deg_seq_in,
                    communities=communities,
                    prop_hr_hr=prop_hr_hr,
                    prop_hr_lr=prop_hr_lr,
                    seed=seed)
    # set the initially infected individuals
    infected= init_infected(n=n, prop_lr_com_size=prop_lr_com_size,
                                         prop_int_inf=prop_int_inf, prop_int_inf_hr=prop_int_inf_hr)
    nx.set_node_attributes(g, dict(zip(infected, len(infected) * [1])), 'health')

    # time simulation
    deaths_hr = []
    deaths_lr = []
    recoveries_hr = []
    recoveries_lr = []
    infections_hr = []
    infections_lr = []
    for i in range(0, n_days):
        # run one step of the simulation
        g, d, r, inf = time_step_simulation(g, seed)
        # keep track of deaths, recoveries and new infections in that day.
        deaths_hr += [d[0]]
        deaths_lr += [d[1]]
        recoveries_hr += [r[0]]
        recoveries_lr += [r[1]]
        infections_hr += [inf[0]]
        infections_lr += [inf[1]]
        seed += 1
    print(list(nx.get_node_attributes(g, "health").values()).count(-2))
    print(list(nx.get_node_attributes(g, "health").values()).count(-1))
    plt.plot(range(n_days), deaths_hr)
    # plt.plot(range(n_days), deaths_lr)
    plt.plot(range(n_days), recoveries_hr)
    # plt.plot(range(n_days), recoveries_lr)
    plt.plot(range(n_days), infections_hr)
    # plt.plot(range(n_days), infections_lr)

    plt.show()



if "__main__" == __name__:
    singe_graph_simulation(n=600, seed=1, prop_int_inf_hr=0.2, n_days=120)
