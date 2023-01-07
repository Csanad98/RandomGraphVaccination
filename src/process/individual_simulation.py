import networkx as nx
import numpy as np
import time
import random

from matplotlib import pyplot as plt

import consts
from plots.plot_simulation import plot_ts_data_for_each_group
from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1

from utils import create_community_random_color_map, community_map_from_community_sizes, \
    community_sizes_generator, init_infected

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq

from node_attributes import attr_assign

from vaccination_strategies import no_vaccination, random_vaccination, max_vaccination_level_reached, \
    risk_group_biased_random_vaccination, high_degree_first_vaccination


def single_graph_generator(seed: int,
                           n: int = 1000,
                           tau: float = 2.8,
                           lam: float = 15,
                           prop_lr_com_size: float = 0.45,
                           prop_int_inf: float = 0.05,
                           prop_int_inf_hr: float = 0.025,
                           prop_hr_hr: float = 0.7,
                           prop_hr_lr: float = 0,
                           vacc_app_prob: float = 0.7):
    """
    Creates a hierarchical configuration model graph with randomly generated degree distributions and assigns the
    attributes of community, health risk group, infectivity and outcome. It also starts with a starting percentage of
    infected nodes
    :param vacc_app_prob: vaccine approval probability
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
    print("community size generation: {:.2f}s".format(time.time() - t0))
    # generate sequences of degree and community distributions
    deg_seq_out = generate_power_law_degree_seq(n=n, tau=tau, seed=seed)
    communities = community_map_from_community_sizes(community_sizes)
    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam)
    print("hcm parameters generation: {:.2f}s".format(time.time() - t0))

    # generate hierarchical configuration model
    g = hierarchical_configuration_model_algo1(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    print("hcm generation: {:.2f}s".format(time.time() - t0))

    # if graph is small enough, plot it
    if n <= 1000:
        color_map = create_community_random_color_map(communities)
        nx.draw_spring(g, with_labels=False, width=0.1, edgecolors="k", alpha=0.9, node_color=color_map, node_size=10)
        plt.show()
        print("graph plotting: {:.2f}s".format(time.time() - t0))

    # assign attributes to graph nodes
    g = attr_assign(g=g,
                    communities=communities,
                    prop_hr_hr=prop_hr_hr,
                    prop_hr_lr=prop_hr_lr,
                    vacc_app_prob=vacc_app_prob,
                    seed=seed)
    print("attribute assignment: {:.2f}s".format(time.time() - t0))
    # set the initially infected individuals
    init_infected(g=g, prop_int_inf=prop_int_inf, prop_int_inf_hr=prop_int_inf_hr, seed=seed)
    print("initial infections added: {:.2f}s".format(time.time() - t0))
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
    # 4: deaths. recoveries, infections, vaccinations
    stats = {"high_risk": np.zeros(shape=(4,)), "low_risk": np.zeros(shape=(4,))}
    # iterate through nodes that are infected (and still alive)
    for node, node_data in filter(lambda xy: xy[1]['health'] > 0, g.nodes.items()):
        # Check all healthy neighbors of i
        for n_node, n_node_data in \
                filter(lambda xy: xy[1]['health'] == 0, g.subgraph(list(g.neighbors(node))).nodes.items()):
            # infect the neighbor with probability of node infectivity
            if np.random.binomial(1, n_node_data["infectivity"]) == 1:
                n_node_data["health"] = 1
                stats[n_node_data["risk_group"]][2] += 1
        # Check if node is not at the end of illness
        if node_data["health"] < node_data["outcome"]:
            # If not add one day to the health timeline
            node_data["health"] += 1
        # Otherwise node reached last day of illness, determine outcome
        elif node_data["risk_group"] == "low_risk":
            if node_data["health"] == consts.DAYS_LR_DEATH:
                node_data["health"] = -2
                stats[node_data["risk_group"]][0] += 1
            elif node_data["health"] == consts.DAYS_LR_RECOVERY:
                node_data["health"] = -1
                stats[node_data["risk_group"]][1] += 1
            else:
                raise Exception("Inconsistency in LR days consts")

        elif node_data["risk_group"] == "high_risk":
            if node_data["health"] == consts.DAYS_HR_DEATH:
                node_data["health"] = -2
                stats[node_data["risk_group"]][0] += 1
            elif node_data["health"] == consts.DAYS_HR_RECOVERY:
                node_data["health"] = -1
                stats[node_data["risk_group"]][1] += 1
            else:
                raise Exception("Inconsistency in HR days consts")
        else:
            raise NotImplementedError("Unsupported risk level")

    return g, stats


def single_graph_simulation(seed: int,
                            n: int = 1000,
                            tau: float = 2.8,
                            lam: float = 15,
                            prop_lr_com_size: float = 0.45,
                            prop_int_inf: float = 0.05,
                            prop_int_hr_inf: float = 0.025,
                            prop_hr_hr: float = 0.7,
                            prop_hr_lr: float = 0,
                            n_days: int = 365,
                            vaccination_strategy: int = 0,
                            max_vacc_threshold: float = 1):
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
                               prop_hr_lr=prop_hr_lr,
                               vacc_app_prob=max_vacc_threshold)
    print("graph generation & preperation: {:.2f}".format(time.time() - t0))

    # time simulation
    # the 8 spots are for: deaths_hr, recoveries_hr, infections_hr, vaccinations_hr,
    # deaths_lr, recoveries_lr, infections_lr, vaccinations_lr
    ts_data = np.zeros((n_days, 8))

    # start of simulation
    for i in range(0, n_days):
        # run one step of the simulation
        g, stats_dict = time_step_simulation(g, seed)
        # keep track of deaths, recoveries and new infections in that day.
        daily_data = np.concatenate((stats_dict["high_risk"], stats_dict["low_risk"]))
        if not max_vaccination_level_reached(max_threshold=max_vacc_threshold,
                                             num_nodes=g.number_of_nodes(),
                                             vaccinated_count=np.sum(ts_data[:, 3]) + np.sum(ts_data[:, 7])):
            if vaccination_strategy == 0:
                vacc_dict = no_vaccination()
            elif vaccination_strategy == 1:
                vacc_dict = random_vaccination(g=g, vacc_percentage=0.004, seed=seed)
            elif vaccination_strategy == 2:
                vacc_dict = risk_group_biased_random_vaccination(g=g, vacc_percentage=0.004, hr_bias=0.9, seed=seed)
            elif vaccination_strategy == 3:
                vacc_dict = high_degree_first_vaccination(g=g, vacc_percentage=0.004)
            else:
                raise NotImplementedError
            daily_data[3], daily_data[7] = vacc_dict["high_risk"], vacc_dict["low_risk"]
        ts_data[i] = daily_data
        seed += 1

    print("simulation of all days: {:.2f}".format(time.time() - t0))

    return g, ts_data


if "__main__" == __name__:
    seed = 1
    n = 1000
    prop_int_hr_inf = 0.005
    prop_int_inf = 0.01
    n_days = 365
    g, ts_data = single_graph_simulation(n=n, seed=seed, prop_int_inf=prop_int_inf, prop_int_hr_inf=prop_int_hr_inf,
                                         n_days=n_days, vaccination_strategy=3, max_vacc_threshold=0.8)

    print("Number of deaths: {}".format(list(nx.get_node_attributes(g, "health").values()).count(-2)))
    print("Number of immune people: {}".format(list(nx.get_node_attributes(g, "health").values()).count(-1)))
    plot_ts_data_for_each_group(ts_data=ts_data, n_days=n_days)
