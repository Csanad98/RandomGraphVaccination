from typing import List, Tuple, Dict

import networkx as nx
from utils import get_nodes_by_degree_order


def get_vaccine_doses_count(g: nx.Graph, vacc_percentage: float, nodes_count: int):
    return min(int(g.number_of_nodes() * vacc_percentage), nodes_count)


def vaccinate_selected_high_degree_nodes_first(g: nx.Graph, nodes_dict: dict, vacc_percentage: float):
    """
    Helper function for high degree first vaccinations.
    Vaccinates as many passed nodes as possible in decreasing order of their degrees.
    :param g: graph
    :param nodes_dict: node_id -> node_data
    :param vacc_percentage:
    :return: vacc stats
    """
    deg_sorted_h_nodes = get_nodes_by_degree_order(g=g, node_ids=list(nodes_dict.keys()))
    available_doses = get_vaccine_doses_count(g=g, vacc_percentage=vacc_percentage, nodes_count=len(deg_sorted_h_nodes))
    to_be_vaccinated = [(node_id, nodes_dict[node_id]) for node_id, _ in deg_sorted_h_nodes[:available_doses]]
    vaccinations = _apply_vaccination_on_selected_nodes(to_be_vaccinated=to_be_vaccinated)
    return vaccinations


def _apply_vaccination_on_selected_nodes(to_be_vaccinated: List[Tuple[int, dict]]) -> Dict[str, int]:
    """
    Vaccinates selected nodes, ratios the number of high and low risk vaccinations.
    :param to_be_vaccinated:
    :return: count of hr and lr vaccinations that have occurred
    """
    vaccinations = {"high_risk": 0, "low_risk": 0}
    # vaccinate chosen nodes
    for node, node_data in to_be_vaccinated:
        if node_data["vaccine_approval"]:
            node_data["health"] = -3
            vaccinations[node_data["risk_group"]] += 1
        else:
            raise Exception("Should not try vaccinating people who are not willing.")
    return vaccinations
