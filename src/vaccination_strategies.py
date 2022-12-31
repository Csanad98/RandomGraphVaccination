import networkx as nx
import numpy as np
import random


def random_vaccination(g: nx.Graph,
                       seed: int,
                       vacc_percentage: float = 0.004):
    """

    :param g: random graph
    :param seed: seed
    :param vacc_percentage: percentage of individuals vaccinated each day
    :return: g: graph that was gone through the vaccination process
    (vaccinations["high_risk"], vaccinations["low_risk"]): tuple containing the amount of vaccinations in low risk and
    high risk groups
    """
    np.random.seed(seed)
    # create list with healthy nodes
    temp_list = [(x, y) for x, y in g.nodes(data=True) if y['health'] == 0]
    vaccinations = {"high_risk": 0, "low_risk": 0}
    to_be_vaccinated = random.sample(temp_list, min(int(vacc_percentage * g.number_of_nodes()), len(temp_list)))
    # create dictionary for attribute assignment
    new_vacc_dict = {x: -1 for x, y in to_be_vaccinated}
    # check amount of vaccinations per risk group
    for x1, y1 in to_be_vaccinated:  # for all healthy nodes
        vaccinations[y1["risk_group"]] += 1
    nx.set_node_attributes(g, new_vacc_dict, "health")

    return g, (vaccinations["high_risk"], vaccinations["low_risk"])
