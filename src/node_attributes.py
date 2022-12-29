import networkx as nx
import numpy as np


def rbin(p: float, out_true, out_false):
    return out_true if np.random.binomial(1, p) == 1 else out_false


def attr_assign(g: nx.Graph,
                deg_seq_out: np.array,
                deg_seq_in: np.array,
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
                days_lr_recovery: int = 17):
    """
    g: Random Graph
    deg_seq_out: The inter-community degree distribution
    deg_seq_in: The inside-community degree distribution
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
    np.random.seed(1)
    for node, node_data in g.nodes.items():
        # risk group attribute
        if node_data["community"] == 0:  # if in low risk community
            risk_group_dict[node] = rbin(prop_hr_lr, "high_risk", "low_risk")
        else:  # if in high risk community
            risk_group_dict[node] = rbin(prop_hr_hr, "high_risk", "low_risk")
        # outcome and infectivity attributes
        if risk_group_dict[node] == "low_risk":  # if low risk individual
            outcome_dict[node], infectivity_dict[node] = rbin(death_prob_lr,
                                                              (days_lr_death, reproduction_number / (
                                                                      days_lr_death * (
                                                                          deg_seq_out[node] + deg_seq_in[node]))),
                                                              (days_lr_recovery,
                                                               reproduction_number / (days_lr_recovery * (
                                                                       deg_seq_out[node] + deg_seq_in[node]))))
        else:  # if high risk individual
            outcome_dict[node], infectivity_dict[node] = rbin(death_prob_hr,
                                                              (days_hr_death, reproduction_number / (
                                                                      days_hr_death * (
                                                                          deg_seq_out[node] + deg_seq_in[node]))),
                                                              (days_hr_recovery,
                                                               reproduction_number / (days_hr_recovery * (
                                                                       deg_seq_out[node] + deg_seq_in[node]))))
    nx.set_node_attributes(g, risk_group_dict, "risk_group")
    nx.set_node_attributes(g, outcome_dict, "outcome")
    nx.set_node_attributes(g, infectivity_dict, "infectivity")

    return g
