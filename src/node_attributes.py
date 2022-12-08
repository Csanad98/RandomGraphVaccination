import networkx as nx
import numpy as np


def attr_assign(g: nx.Graph, deg_seq_out: np.array, deg_seq_in:np.array, communities: np.array, reproduction_number: int = 2.9, seed: int = 0):
    np.random.seed(seed)
    n = len(communities)
    # Adding Community attribute
    nx.set_node_attributes(g, dict(zip(range(n), communities)), "community")

    # Adding deg_out attribute
    nx.set_node_attributes(g, dict(zip(range(n), deg_seq_out)), "deg_out")

    # Adding deg_in attribute
    nx.set_node_attributes(g, dict(zip(range(n), deg_seq_in)), "deg_in")

    # Adding Health attribute
    nx.set_node_attributes(g, dict(zip(range(n), [0] * n)), "health")

    # Adding Risk Group attribute
    risk_group_dict = {}
    np.random.seed(1)
    for i in range(n):
        # if element in Low Risk group then choose a different distribution of risk
        if nx.get_node_attributes(g, "community")[i] == 0:
            risk_group_dict[i] = "high_risk" if np.random.binomial(1, 0.16666666) == 1 else "low_risk"
        else:
            risk_group_dict[i] = "high_risk" if np.random.binomial(1, 0.83333333) == 1 else "low_risk"
    nx.set_node_attributes(g, risk_group_dict, "risk_group")

    # Adding Outcome and infectivity attributes
    outcome_dict = {}
    infectivity_dict = {}

    p1 = reproduction_number/(18*(deg_seq_out[i]+deg_seq_in[i]))
    p2 = reproduction_number/(22*(deg_seq_out[i]+deg_seq_in[i]))
    p3 = reproduction_number / (17 * (deg_seq_out[i] + deg_seq_in[i]))
    for i in range(n):
        # if element in Low Risk group then choose different probability of death and recovery and infectivity
        if nx.get_node_attributes(g, "risk_group")[i] == "low_risk":
            outcome_dict[i], infectivity_dict[i] = (18, p1) if np.random.binomial(1, 0.083) == 1 else (17, p3)
        else:
            outcome_dict[i], infectivity_dict[i] = (18, p1) if np.random.binomial(1, 0.012) == 1 else (22, p2)
    nx.set_node_attributes(g, outcome_dict, "outcome")
    nx.set_node_attributes(g, infectivity_dict, "infectivity")

    return g
