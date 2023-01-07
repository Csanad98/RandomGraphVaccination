from typing import List, Tuple

import networkx as nx
import numpy as np

from utils import get_conditional_nodes


def count_nodes_in_states(g: nx.Graph, attributes: List[str], values: List):
    return len(get_conditional_nodes(g=g, attributes=attributes, values=values))


def collect_health_attr_stats(g: nx.Graph):
    dead = get_health_stats(g=g, health_code=-2)
    recovered = get_health_stats(g=g, health_code=-1)
    vaccinated = get_health_stats(g=g, health_code=-3)
    never_virus = get_health_stats(g=g, health_code=0)
    num_immune = tuple(np.array(vaccinated) + np.array(recovered))

    print("Health stats:")
    health_stats_printer(counts=dead, type_str="deaths")
    health_stats_printer(counts=recovered, type_str="recoveries")
    health_stats_printer(counts=vaccinated, type_str="vaccinated")
    health_stats_printer(counts=num_immune, type_str="immune")
    health_stats_printer(counts=never_virus, type_str="people who never got the virus")


def get_health_stats(g: nx.Graph, health_code: int):
    num_hr = count_nodes_in_states(g=g, attributes=["health", "risk_group"], values=[health_code, "high_risk"])
    num_lr = count_nodes_in_states(g=g, attributes=["health", "risk_group"], values=[health_code, "low_risk"])
    num_total = num_hr + num_lr
    return num_total, num_hr, num_lr


def health_stats_printer(counts: Tuple[int, int, int], type_str: str):
    print("Number of {}: {}".format(type_str, counts[0]))
    print("Number of hr {}: {}".format(type_str, counts[1]))
    print("Number of lr {}: {}".format(type_str, counts[2]))
