from typing import List, Tuple, Dict

import networkx as nx
import random


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
    healthy_nodes = get_conditional_nodes(g=g, attributes=["health"], values=[0])
    to_be_vaccinated = random.sample(healthy_nodes, min(int(vacc_percentage * g.number_of_nodes()), len(healthy_nodes)))
    vaccinations = apply_vaccination_on_selected_nodes(to_be_vaccinated=to_be_vaccinated)
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
    hr_h_nodes = get_conditional_nodes(g=g, attributes=["health", "risk_group"], values=[0, "high_risk"])
    lr_h_nodes = get_conditional_nodes(g=g, attributes=["health", "risk_group"], values=[0, "low_risk"])
    to_be_vaccinated_hr = random.sample(hr_h_nodes, min(int(hr_bias * vacc_percentage * g.number_of_nodes()),
                                                        len(hr_h_nodes)))
    remaining_doses = int(vacc_percentage * g.number_of_nodes()) - len(to_be_vaccinated_hr)
    to_be_vaccinated_lr = random.sample(lr_h_nodes, min(remaining_doses, len(hr_h_nodes)))
    all_to_be_vaccinated = to_be_vaccinated_hr + to_be_vaccinated_lr
    vaccinations = apply_vaccination_on_selected_nodes(to_be_vaccinated=all_to_be_vaccinated)
    return vaccinations


def get_conditional_nodes(g: nx.Graph, attributes: List[str], values: List) -> List[Tuple[int, dict]]:
    """
    Creates list with nodes and their attributes that satisfy a condition.
    :param attributes: node_data attributes
    :param values: the desired values the attributes must satisfy for a node to be selected
    :param g: graph
    :return: list of nodes satisfying conditions
    """
    result = []
    for node, node_data in g.nodes.items():
        for attr, val in zip(attributes, values):
            if node_data[attr] != val:
                break
        # only adds node to result if all conditions were satisfied
        else:
            result += [(node, node_data)]
    return result


def apply_vaccination_on_selected_nodes(to_be_vaccinated: List[Tuple[int, dict]]) -> Dict[str, int]:
    """
    Vaccinates selected nodes, counts the number of high and low risk vaccinations.
    :param to_be_vaccinated:
    :return:
    """
    vaccinations = {"high_risk": 0, "low_risk": 0}
    # vaccinate chosen nodes
    for node, node_data in to_be_vaccinated:
        node_data["health"] = -1
        vaccinations[node_data["risk_group"]] += 1
    return vaccinations
