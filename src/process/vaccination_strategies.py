import networkx as nx
import random


def max_vaccination_level_reached(max_threshold: float, num_nodes: int, vaccinated_count: int):
    if vaccinated_count < num_nodes*max_threshold:
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
    :return: vaccinations, a dict containing the amount of vaccinations in low risk and
    high risk groups
    """
    random.seed(seed)
    # create list with healthy nodes
    healthy_nodes = [(node, node_data) for node, node_data in g.nodes.items() if node_data['health'] == 0]
    to_be_vaccinated = random.sample(healthy_nodes, min(int(vacc_percentage * g.number_of_nodes()), len(healthy_nodes)))
    vaccinations = {"high_risk": 0, "low_risk": 0}
    # vaccinate chosen nodes
    for node, node_data in to_be_vaccinated:
        node_data["health"] = -1
        vaccinations[node_data["risk_group"]] += 1
    return vaccinations


def risk_group_biased_random_vaccination(g: nx.Graph, vacc_percentage: float = 0.004, hr_bias: float = 0.5, seed: int = 0):
    # todo
    pass

