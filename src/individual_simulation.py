import networkx as nx
import numpy as np
import time
import random

from matplotlib import pyplot as plt

from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1

from utils import create_community_random_color_map, community_map_from_community_sizes, \
    community_sizes_generator, init_infected

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq

from node_attributes import attr_assign

from vaccination_strategies import random_vaccination


def single_graph_generator(seed: int,
                           n: int = 1000,
                           tau: float = 2.8,
                           lam: float = 15,
                           prop_lr_com_size: float = 0.45,
                           prop_int_inf: float = 0.05,
                           prop_int_inf_hr: float = 0.5,
                           prop_hr_hr: float = 0.7,
                           prop_hr_lr: float = 0):
    """
    Creates a hierarchical configuration model graph with randomly generated degree distributions and assigns the
    attributes of community, health risk group, infectivity and outcome. It also starts with a starting percentage of
    infected nodes
    :param n: number of people
    :param seed: seed
    :param tau: parameter for outward degree distribution
    :param lam: parameter for inter-community degree distribution
    :param prop_lr_com_size: proportion of nodes in the low risk community
    :param prop_int_inf: proportion of initially infected nodes
    :param prop_int_inf_hr: proportion of high risk in the initially infected nodes
    :param prop_hr_hr: proportion of high risk individuals in high risk groups
    :param prop_hr_lr: proportion of high risk individuals in low risk groups
    :return: graph: the generated hierarchical configuration model graph with the node attributes
    """
    t0 = time.time()
    # generate sequence of community sizes
    community_sizes = community_sizes_generator(n=n, prop_lr_com_size=prop_lr_com_size, seed=seed)
    print("community size generation")
    print(time.time() - t0)

    # generate sequences of degree and community distributions
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    print("hcm parameters generation")
    print(time.time() - t0)

    # generate hierarchical configuration model
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    print("hcm generation")
    print(time.time() - t0)

    # color_map = create_community_random_color_map(communities)
    # nx.draw_spring(g, with_labels=False, width=0.3, edgecolors="k", alpha=0.9, node_color=color_map, node_size=80)
    # # plot graph
    # plt.show()
    # print("graph plotting")
    # print(time.time() - t0)

    # assign attributes to graph nodes
    g = attr_assign(g=g,
                    communities=communities,
                    prop_hr_hr=prop_hr_hr,
                    prop_hr_lr=prop_hr_lr,
                    seed=seed)
    print("attribute assignment")
    print(time.time() - t0)

    # set the initially infected individuals
    infected = init_infected(n=n, prop_lr_com_size=prop_lr_com_size,
                             prop_int_inf=prop_int_inf, prop_int_inf_hr=prop_int_inf_hr)
    nx.set_node_attributes(g, dict(zip(infected, len(infected) * [1])), 'health')

    return g


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
    health_dict = {}
    for x1, y1 in [(x, y) for x, y in g.nodes(data=True) if y['health'] > 0]:
        # Check all healthy neighbors of i
        for x2, y2 in [(x, y) for x, y in g.subgraph(list(g.neighbors(x1))).nodes(data=True) if y['health'] == 0]:
            # infect the neighbor with probability of node infectivity
            if np.random.binomial(1, y2["infectivity"]) == 1:
                health_dict[x2] = 1
                infections[y2["risk_group"]] += 1
        # Check if node is not at the end of illness
        if y1["health"] < y1["outcome"]:
            # If not add one day to the health timeline
            health_dict[x1] = y1["health"] + 1
        # Otherwise check if outcome is death
        elif y1["outcome"] == 18:
            health_dict[x1] = -2
            deaths[y1["risk_group"]] += 1
        # Otherwise we have recovery
        else:
            health_dict[x1] = -1
            recoveries[y1["risk_group"]] += 1

    nx.set_node_attributes(g, health_dict, 'health')

    return g, (deaths["high_risk"], deaths["low_risk"]), \
           (recoveries["high_risk"], recoveries["low_risk"]), \
           (infections["high_risk"], infections["low_risk"])


def single_graph_simulation(seed: int,
                            n: int = 1000,
                            tau: float = 2.8,
                            lam: float = 15,
                            prop_lr_com_size: float = 0.45,
                            prop_int_inf: float = 0.05,
                            prop_int_hr_inf: float = 0.5,
                            prop_hr_hr: float = 0.7,
                            prop_hr_lr: float = 0,
                            n_days: int = 365,
                            vaccination_strategy: int = 0,
                            vac_stop: float = 1):
    """
    Creates a graph and simulates n_days days of the graph.
    :param n: number of people
    :param seed: seed
    :param tau: parameter for outward degree distribution
    :param lam: parameter for inter-community degree distribution
    :param prop_lr_com_size: proportion of nodes in the low risk community
    :param prop_int_inf: proportion of initially infected nodes
    :param prop_int_hr_inf: proportion of high risk in the initially infected nodes
    :param prop_hr_hr: proportion of high risk individuals in high risk groups
    :param prop_hr_lr: proportion of high risk individuals in low risk groups
    :param n_days: number of days simulated
    :param vaccination_strategy: the vaccination strategy to be used: takes values:
                                0: no vaccination
                                1: random vaccination
                                2: risk group based vaccination
                                3: risk community based vaccination
    :return: g: the random graph at the end of the simulation
             deaths_hr: times series of deaths of high risk individuals
             deaths_lr: times series of deaths of low risk individuals
             recoveries_hr: times series of recoveries of high risk individuals
             recoveries_lr: times series of recoveries of low risk individuals
             infections_hr: times series of infections of high risk individuals
             infections_lr: times series of infections of low risk individuals
             vaccinations_hr: times series of vaccinations of high risk individuals
             vaccinations_lr: times series of vaccinations of low risk individuals
    """
    t0 = time.time()
    # generation of our hierarchical configuration model random graph
    g = single_graph_generator(seed=seed,
                               n=n,
                               tau=tau,
                               lam=lam,
                               prop_lr_com_size=prop_lr_com_size,
                               prop_int_inf=prop_int_inf,
                               prop_int_inf_hr=prop_int_hr_inf,
                               prop_hr_hr=prop_hr_hr,
                               prop_hr_lr=prop_hr_lr)
    print("graph generation & preperation")
    print(time.time() - t0)

    # time simulation
    deaths_hr = []
    deaths_lr = []
    recoveries_hr = []
    recoveries_lr = []
    infections_hr = []
    infections_lr = []
    vaccinations_hr = []
    vaccinations_lr = []
    # start of simulation
    if vaccination_strategy == 0:  # simulation for no vaccination stategy
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
        vaccinations_hr = n_days * [0]
        vaccinations_lr = n_days * [0]
    elif vaccination_strategy == 1:  # random vaccination strategy
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
            # run daily vaccinations
            if vac_stop * g.number_of_nodes() > sum(recoveries_lr + recoveries_hr + vaccinations_hr + vaccinations_lr):
                g, vac = random_vaccination(g=g, seed=seed, vacc_percentage=0.004)
                vaccinations_hr += [vac[0]]
                vaccinations_lr += [vac[1]]
            else:
                vaccinations_hr += [0]
                vaccinations_lr += [0]
            seed += 1
    print("simulation of all days")
    print(time.time() - t0)

    return g, deaths_hr, deaths_lr, recoveries_hr, recoveries_lr, infections_hr, infections_lr, vaccinations_hr, \
           vaccinations_lr


if "__main__" == __name__:
    seed = 1
    n = 10000
    prop_int_hr_inf = 0.2
    n_days = 365
    g, deaths_hr, deaths_lr, recoveries_hr, recoveries_lr, infections_hr, infections_lr, vaccinations_hr, \
    vaccinations_lr = single_graph_simulation(n=n, seed=seed, prop_int_hr_inf=prop_int_hr_inf, n_days=n_days,
                                              vaccination_strategy=1, vac_stop=1)

    print(list(nx.get_node_attributes(g, "health").values()).count(-2))
    print(list(nx.get_node_attributes(g, "health").values()).count(-1))
    plt.plot(range(n_days), deaths_hr)
    # plt.plot(range(n_days), deaths_lr)
    plt.plot(range(n_days), recoveries_hr)
    # plt.plot(range(n_days), recoveries_lr)
    plt.plot(range(n_days), infections_hr)
    # plt.plot(range(n_days), infections_lr)
    plt.plot(range(n_days), vaccinations_hr)
    # plt.plot(range(n_days), vaccinations_lr)
    plt.show()
