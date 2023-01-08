import networkx as nx
import random

from utils import get_conditional_nodes
from process.vaccine_utils import vaccinate_selected_high_degree_nodes_first, _apply_vaccination_on_selected_nodes


def max_vaccination_level_reached(max_threshold: float, num_nodes: int, vaccinated_count: int):
    if vaccinated_count < num_nodes * max_threshold:
        return False
    else:
        return True


class VaccinationStrategy:
    def __init__(self, strategy_id: int):
        self.strategy_id = strategy_id
        if strategy_id == 0:
            self.apply_daily_vaccination = no_vaccination
        elif strategy_id == 1:
            self.apply_daily_vaccination = random_vaccination
        elif strategy_id == 2:
            self.apply_daily_vaccination = risk_group_biased_random_vaccination
        elif strategy_id == 3:
            self.apply_daily_vaccination = high_degree_first_vaccination
        elif strategy_id == 4:
            self.apply_daily_vaccination = self.community_ring_vaccination
            self.current_community = 1
            self.current_community_nodes = None
        else:
            raise NotImplementedError

    def community_ring_vaccination(self, g: nx.Graph, last_community_id: int, vacc_percentage: float = 0.004):
        """
        Vaccination community by community. Within a community, only vaccinate nodes that connect the community to the
        outside of the community. Vaccinate high degree connector nodes first. It's not realistic to vaccinate everyone
        on the border - this is taken care of by random nodes refusing vaccines.
        One day vaccination can only happen in one community, hence sometimes there are left over doses that cannot be
        used - this emulates how the vaccine staff needs to travel and setup at each community they vist.
        :return: vaccine stats
        """
        # always update to get latest data
        self.current_community_nodes = get_conditional_nodes(g=g,
                                                             attributes=["community", "health", "vaccine_approval"],
                                                             values=[self.current_community, 0, True])
        # once no more people to vaccinate in the current community, switch to the next one
        while len(self.current_community_nodes) == 0 and self.current_community < last_community_id:
            self.current_community += 1
            self.current_community_nodes = get_conditional_nodes(g=g,
                                                                 attributes=["community", "health", "vaccine_approval"],
                                                                 values=[self.current_community, 0, True])

        # filter to get nodes that connect to outside
        connector_nodes = {}
        for node, node_data in self.current_community_nodes:
            for neighbor in g.neighbors(node):
                if g.nodes[neighbor]["community"] != self.current_community:
                    connector_nodes[node] = node_data
                    break
        # order by degree and vaccinate
        return vaccinate_selected_high_degree_nodes_first(g=g, nodes_dict=connector_nodes,
                                                          vacc_percentage=vacc_percentage)


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
    return vaccinate_selected_high_degree_nodes_first(g=g, nodes_dict=h_nodes_dict, vacc_percentage=vacc_percentage)


def high_out_degree_communities_first():
    pass


def nodes_connecting_most_communities_first():
    pass


def high_betweenness_nodes_first():
    pass
