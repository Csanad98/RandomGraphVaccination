from typing import Tuple

import networkx as nx
import numpy as np


def attr_assign(g: nx.Graph,
                communities: np.array,
                prop_hr_hr: float,
                prop_hr_lr: float,
                reproduction_number: int = 2.9,
                seed: int = 0,
                death_prob_hr: float = 0.083,
                death_prob_lr: float = 0.012,
                days_hr_death: int = 18,
                days_hr_recovery: int = 22,
                days_lr_death: int = 18,
                days_lr_recovery: int = 17,
                risk_level_choices: Tuple[str] = ("high_risk", "low_risk")):
    """
    g: Random Graph
    communities: list containing the indexes refering the to the community of the nodes
    prop_hr_hr: proportion of high-risk people inside high risk communities
    prop_hr_lr: proportion of high-risk people inside low risk communities
    reproduction number: the average number of individuals infected by an infected individual
    seed: seed

    Assigns the following attributes to each of the nodes of the random graph:
    community: the community which the node belongs to
    deg_out: the outward degree of the node
    deg_in: the inward degree of the node
    health: the health state of the node, initially 0 (healthy) for every node
    risk_group: the risk group which the node belongs to. Chosen randomly but based on communities
    outcome: the outcome of the disease -> death, recovery. Denoted as number of days of infection (which are in 1-1 relation to the outcome)
    infectivity: probability of the node, if infected, to affect a neighboring node in a single day
    """
    np.random.seed(seed)
    n = len(communities)
    # Adding Community attribute
    nx.set_node_attributes(g, dict(zip(range(n), communities)), "community")

    # Adding Health attribute
    nx.set_node_attributes(g, dict(zip(range(n), [0] * n)), "health")

    # Adding Risk Group, Outcome and infectivity attributes
    risk_group_dict = {}
    outcome_dict = {}
    infectivity_dict = {}
    for node, node_data in g.nodes.items():
        # risk group attribute
        if node_data["community"] == 0:  # if in low risk community
            risk_group_dict[node] = np.random.choice(a=risk_level_choices, size=1, p=[prop_hr_lr, 1-prop_hr_lr])[0]
        else:  # if in high risk community
            risk_group_dict[node] = np.random.choice(a=risk_level_choices, size=1, p=[prop_hr_hr, 1-prop_hr_hr])[0]
        # outcome and infectivity attributes
        if risk_group_dict[node] == "low_risk":  # if low risk individual
            outcome_dict[node] = \
                np.random.choice(a=[days_lr_death, days_lr_recovery], size=1, p=[death_prob_lr, 1 - death_prob_lr])[0]
            infectivity_dict[node] = \
                np.random.choice(a=[get_infect_prob(reproduction_number, days_lr_death, g.degree[node]),
                                    get_infect_prob(reproduction_number, days_lr_recovery, g.degree[node])],
                                 size=1, p=[death_prob_lr, 1 - death_prob_lr])[0]
        else:  # if high risk individual
            outcome_dict[node] = \
                np.random.choice(a=[days_hr_death, days_hr_recovery], size=1, p=[death_prob_hr, 1 - death_prob_hr])[0]
            infectivity_dict[node] = \
                np.random.choice(a=[get_infect_prob(reproduction_number, days_hr_death, g.degree[node]),
                                    get_infect_prob(reproduction_number, days_hr_recovery, g.degree[node])],
                                 size=1, p=[death_prob_hr, 1 - death_prob_hr])[0]

    nx.set_node_attributes(g, risk_group_dict, "risk_group")
    nx.set_node_attributes(g, outcome_dict, "outcome")
    nx.set_node_attributes(g, infectivity_dict, "infectivity")

    return g


def get_infect_prob(reproduction_number: int, days: int, node_degree: int):
    """
    Calculates the probability of infection via any of the links of a node.
    :param reproduction_number: the average number of individuals infected by an infected individual
    :param days: number of days the node will be infected
    :param node_degree: number of links of the node
    :return: infectivity: probability of the node, if infected, to affect a neighboring node in a single day
    """
    return reproduction_number / (days * node_degree)
