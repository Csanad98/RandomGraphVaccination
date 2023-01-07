from typing import List, Tuple, Dict

import networkx as nx
import random

from utils import get_conditional_nodes


def max_vaccination_level_reached(max_threshold: float, num_nodes: int, vaccinated_count: int):
    if vaccinated_count < num_nodes * max_threshold:
        return False
    else:
        return True


def no_vaccination():
    return {"high_risk": 0, "low_risk": 0}


def random_vaccination(g: nx.Graph, vacc_percentage: float = 0.004, seed: int = 0):
    """
    Vaccinate vacc_percentage ratio of healthy nodes in the graph.
    :param g: random graph
    :param seed: seed
    :param vacc_percentage: percentage of individuals vaccinated each day
    :return: vaccinations, a dict containing the amount of vaccinations in low risk and high risk groups
    """
    random.seed(seed)
    healthy_nodes = get_conditional_nodes(g=g, attributes=["health", "vaccine_approval"], values=[0, True])
    to_be_vaccinated = random.sample(healthy_nodes, min(int(vacc_percentage * g.number_of_nodes()), len(healthy_nodes)))
    vaccinations = _apply_vaccination_on_selected_nodes(to_be_vaccinated=to_be_vaccinated)
    return vaccinations


def risk_group_biased_random_vaccination(g: nx.Graph, vacc_percentage: float = 0.004, hr_bias: float = 0.5,
                                         seed: int = 0):
    """
    Each day vaccinate hr_bias proportion of (healthy) high risk people from the daily available doses.
    If not enough high risk people are available, use the remaining doses for low risk people.
    :param g: graph
    :param vacc_percentage: percentage of individuals (nodes) vaccinated each day
    :param hr_bias:
    :param seed: seed
    :return: vaccinations, a dict containing the amount of vaccinations in low risk and high risk groups
    """
    random.seed(seed)
    hr_h_nodes = get_conditional_nodes(g=g,
                                       attributes=["health", "risk_group", "vaccine_approval"],
                                       values=[0, "high_risk", True])
    lr_h_nodes = get_conditional_nodes(g=g,
                                       attributes=["health", "risk_group", "vaccine_approval"],
                                       values=[0, "low_risk", True])
    to_be_vaccinated_hr = random.sample(hr_h_nodes, min(int(hr_bias * vacc_percentage * g.number_of_nodes()),
                                                        len(hr_h_nodes)))
    remaining_doses = int(vacc_percentage * g.number_of_nodes()) - len(to_be_vaccinated_hr)
    to_be_vaccinated_lr = random.sample(lr_h_nodes, min(remaining_doses, len(hr_h_nodes)))
    all_to_be_vaccinated = to_be_vaccinated_hr + to_be_vaccinated_lr
    vaccinations = _apply_vaccination_on_selected_nodes(to_be_vaccinated=all_to_be_vaccinated)
    return vaccinations


def high_degree_first_vaccination(g: nx.Graph, vacc_percentage: float = 0.004):
    """
    Vaccinate healthy nodes in a non-increasing order of their degree.
    :param g:
    :param vacc_percentage:
    :param seed:
    :return: vaccine stats
    """
    h_nodes_dict = {node_id: node_data for node_id, node_data in
                    get_conditional_nodes(g=g, attributes=["health", "vaccine_approval"], values=[0, True])}
    deg_sorted_h_nodes = sorted(g.degree(list(h_nodes_dict.keys())), key=lambda x: x[1], reverse=True)
    available_doses = min(int(g.number_of_nodes() * vacc_percentage), len(deg_sorted_h_nodes))
    to_be_vaccinated = [(node_id, h_nodes_dict[node_id]) for node_id, _ in deg_sorted_h_nodes[:available_doses]]
    vaccinations = _apply_vaccination_on_selected_nodes(to_be_vaccinated=to_be_vaccinated)
    return vaccinations


def community_ring_vaccination():
    pass


def high_out_degree_communities_first():
    pass


def _apply_vaccination_on_selected_nodes(to_be_vaccinated: List[Tuple[int, dict]]) -> Dict[str, int]:
    """
    Vaccinates selected nodes, counts the number of high and low risk vaccinations.
    :param to_be_vaccinated:
    :return: count of hr and lr vaccinations that have occurred
    """
    vaccinations = {"high_risk": 0, "low_risk": 0}
    # vaccinate chosen nodes
    for node, node_data in to_be_vaccinated:
        if node_data["vaccine_approval"]:
            node_data["health"] = -1
            vaccinations[node_data["risk_group"]] += 1
        else:
            raise Exception("Should not try vaccinating people who are not willing.")
    return vaccinations
