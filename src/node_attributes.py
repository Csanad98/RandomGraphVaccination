import networkx as nx
import numpy as np


def attr_assign(g: nx.Graph, communities: np.array, seed: int = 0):
    np.random.seed(seed)
    n = len(communities)
    # Adding Community attribute
    nx.set_node_attributes(g, dict(zip(range(n), communities)), "community")

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

    # Adding Outcome attribute
    outcome_dict = {}
    for i in range(n):
        # if element in Low Risk group then choose different probability of death and recovery
        if nx.get_node_attributes(g, "risk_group")[i] == "low_risk":
            outcome_dict[i] = "death" if np.random.binomial(1, 0.083) == 1 else "recovery"
        else:
            outcome_dict[i] = "death" if np.random.binomial(1, 0.012) == 1 else "recovery"
    nx.set_node_attributes(g, outcome_dict, "outcome")

    return g
