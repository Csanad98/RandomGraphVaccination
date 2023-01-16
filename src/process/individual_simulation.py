import networkx as nx
import numpy as np
import time
import random

from matplotlib import pyplot as plt

import consts
from analysis.stats import count_nodes_in_states, collect_health_attr_stats, get_end_time_of_pandemic, \
    get_max_infected_ratio
from plots.plot_simulation import plot_ts_data_for_each_group
from random_graphs.hierarchical_cm import hierarchical_configuration_model_algo1, hierarchical_configuration_model_algo3

from utils import create_community_random_color_map, community_map_from_community_sizes, \
    community_sizes_generator, init_infected, get_max_community_id

from random_graphs.degree_sequence_generator import generate_power_law_degree_seq, \
    generate_community_degree_seq, generate_poisson_degree_seq

from node_attributes import attr_assign

from vaccination_strategies import max_vaccination_level_reached, VaccinationStrategy, high_degree_first_vaccination


def single_graph_generator(seed: int,
                           t0: float,
                           n: int = 1000,
                           lam_out: float = 2.8,
                           lam_in: float = 15,
                           prop_lr_com_size: float = 0.45,
                           prop_com_size: float = 0.04,
                           prop_int_inf: float = 0.01,
                           prop_int_inf_hr: float = 0.5,
                           prop_hr_hr: float = 0.7,
                           prop_hr_lr: float = 0,
                           vacc_app_prob: float = 0.7,
                           degree_distribution: str = "power_law"):
    """
    Creates a hierarchical configuration model graph with randomly generated degree distributions and assigns the
    attributes of community, health risk group, infectivity and outcome. It also starts with a starting percentage of
    infected nodes
    :param degree_distribution:
    :param prop_com_size: mean proportion of nodes in the high risk communities
    :param vacc_app_prob: vaccine approval probability
    :param n: number of people
    :param seed: seed
    :param lam_out: parameter for outward degree distribution
    :param lam_in: parameter for inter-community degree distribution
    :param prop_lr_com_size: proportion of nodes in the low risk community
    :param prop_int_inf: proportion of initially infected nodes
    :param prop_int_inf_hr: proportion of high risk in the initially infected nodes
    :param prop_hr_hr: proportion of high risk individuals in high risk groups
    :param prop_hr_lr: proportion of high risk individuals in low risk groups
    :return: graph: the generated hierarchical configuration model graph with the node attributes
    """
    # generate sequence of community sizes
    community_sizes = community_sizes_generator(n=n, prop_lr_com_size=prop_lr_com_size, prop_com_size=prop_com_size,
                                                seed=seed)
    # print("community size generation: {:.2f}s".format(time.time() - t0))
    # generate sequences of degree and community distributions
    communities = community_map_from_community_sizes(community_sizes)
    if degree_distribution == "power_law":
        deg_seq_out = generate_power_law_degree_seq(n=n, tau=lam_out, seed=seed)
    elif degree_distribution == "poisson":
        deg_seq_out = generate_poisson_degree_seq(n=n, lam=lam_out, seed=seed)
    else:
        raise NotImplementedError("only power law and poisson degree distribution is supported")

    deg_seq_in = generate_community_degree_seq(seq_generator=generate_poisson_degree_seq,
                                               community_sizes=community_sizes,
                                               gen_param=lam_in)

    # print("hcm parameters generation: {:.2f}s".format(time.time() - t0))

    # generate hierarchical configuration model
    g = hierarchical_configuration_model_algo3(deg_seq_in=deg_seq_in, deg_seq_out=deg_seq_out, communities=communities)
    # print("hcm generation: {:.2f}s".format(time.time() - t0))

    # if graph is small enough, plot it
    if n <= 1000:
        color_map = create_community_random_color_map(communities)
        nx.draw_spring(g, with_labels=False, width=0.1, edgecolors="k", alpha=0.9, node_color=color_map, node_size=10)
        plt.savefig("network.png")
        plt.show()
        # print("graph plotting: {:.2f}s".format(time.time() - t0))

    # assign attributes to graph nodes
    g = attr_assign(g=g,
                    communities=communities,
                    prop_hr_hr=prop_hr_hr,
                    prop_hr_lr=prop_hr_lr,
                    vacc_app_prob=vacc_app_prob,
                    seed=seed)
    # print("attribute assignment: {:.2f}s".format(time.time() - t0))
    # set the initially infected individuals
    init_infected(g=g, prop_int_inf=prop_int_inf, prop_int_inf_hr=prop_int_inf_hr, seed=seed)
    # print("initial infections added: {:.2f}s".format(time.time() - t0))
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
    # iterate through nodes that are infected (and not dead or immune)
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
                            g: nx.Graph,
                            t0: float,
                            n_days: int = 365,
                            vaccination_strategy: int = 0,
                            daily_vacc_prop: float = 0.004,
                            max_vacc_threshold: float = 1,
                            ):
    """
    simulates n_days days of the graph.
    :param daily_vacc_prop: proportion of population that can be vaccinated in one day
    :param seed: seed
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

    # time simulation
    # the 8 spots are for: deaths_hr, recoveries_hr, infections_hr, vaccinations_hr,
    # deaths_lr, recoveries_lr, infections_lr, vaccinations_lr
    ts_data = np.zeros((n_days, 8))

    max_comm_id = get_max_community_id(g=g)
    vacc_strat = VaccinationStrategy(strategy_id=vaccination_strategy, max_comm_id=max_comm_id, g=g)

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
                vacc_dict = vacc_strat.apply_daily_vaccination()
            elif vaccination_strategy == 1:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop, seed=seed)
            elif vaccination_strategy == 2:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop,
                                                               hr_bias=0.9, seed=seed)
            elif vaccination_strategy == 3:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop)
            elif vaccination_strategy == 4:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop)
            elif vaccination_strategy == 5:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop)
            elif vaccination_strategy == 6:
                vacc_dict = vacc_strat.apply_daily_vaccination(g=g, vacc_percentage=daily_vacc_prop)
            else:
                raise NotImplementedError
            daily_data[3], daily_data[7] = vacc_dict["high_risk"], vacc_dict["low_risk"]
        ts_data[i] = daily_data
        seed += 1

    # print("simulation of all days: {:.2f}s".format(time.time() - t0))

    return g, ts_data


if "__main__" == __name__:
    t0: float = time.time()
    seed = 0
    n = 500
    prop_int_inf = 0.005  # total proportion of nodes that are initially infected (both low and high risk ppl)
    prop_int_hr_inf = 0.5  # proportion of initially infected ppl that are high risk
    n_days = 365
    vacc_strategy = 6
    prop_lr_com_size = 0.2
    prop_com_size = 0.05  # HR com size
    lam_out: float = 3.0  # poisson or power law
    lam_in: float = 40  # poisson parameter
    prop_hr_hr: float = 0.9
    prop_hr_lr: float = 0
    max_vacc_threshold = 0.8
    g = single_graph_generator(seed=seed,
                               n=n,
                               lam_out=lam_out,
                               lam_in=lam_in,
                               prop_lr_com_size=prop_lr_com_size,
                               prop_com_size=prop_com_size,
                               prop_int_inf=prop_int_inf,
                               prop_int_inf_hr=prop_int_hr_inf,
                               prop_hr_hr=prop_hr_hr,
                               prop_hr_lr=prop_hr_lr,
                               vacc_app_prob=max_vacc_threshold,
                               t0=t0)
    g, ts_data = single_graph_simulation(seed=seed, n_days=n_days, vaccination_strategy=vacc_strategy,
                                         max_vacc_threshold=0.8, g=g, t0=t0)
    collect_health_attr_stats(g=g)
    plot_ts_data_for_each_group(ts_data=ts_data, n_days=n_days)
    print("Pandemic end day: {}".format(get_end_time_of_pandemic(ts_data)))
    peaks = get_max_infected_ratio(time_series_data=ts_data, num_nodes=n)
    print("Pandemic peak ratios; everyone: {}, hr: {}, lr: {}".format(peaks[0], peaks[1], peaks[2]))
    print("experiment took: {:.2f}s".format(time.time() - t0))
