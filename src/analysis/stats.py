from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from utils import get_conditional_nodes


def count_nodes_in_states(g: nx.Graph, attributes: List[str], values: List):
    return len(get_conditional_nodes(g=g, attributes=attributes, values=values))


def collect_health_attr_stats(g: nx.Graph, to_print: bool = True):
    dead = get_health_stats(g=g, health_code=-2)
    recovered = get_health_stats(g=g, health_code=-1)
    vaccinated = get_health_stats(g=g, health_code=-3)
    never_virus = get_health_stats(g=g, health_code=0)
    num_immune = tuple(np.array(vaccinated) + np.array(recovered))

    if to_print:
        print("Health stats:")
        health_stats_printer(ratios=dead, type_str="deaths")
        health_stats_printer(ratios=recovered, type_str="recoveries")
        health_stats_printer(ratios=vaccinated, type_str="vaccinated")
        health_stats_printer(ratios=num_immune, type_str="immune")
        health_stats_printer(ratios=never_virus, type_str="people who never got the virus")

    return dead, recovered, vaccinated, never_virus, num_immune


def get_health_stats(g: nx.Graph, health_code: int):
    n = g.number_of_nodes()
    num_hr = count_nodes_in_states(g=g, attributes=["health", "risk_group"], values=[health_code, "high_risk"])
    num_lr = count_nodes_in_states(g=g, attributes=["health", "risk_group"], values=[health_code, "low_risk"])
    num_total = num_hr + num_lr
    return num_total / n, num_hr / n, num_lr / n


def health_stats_printer(ratios: Tuple[float, float, float], type_str: str):
    print("Ratio of {}: {}".format(type_str, ratios[0]))
    print("Ratio of hr {}: {}".format(type_str, ratios[1]))
    print("Ratio of lr {}: {}".format(type_str, ratios[2]))


def get_end_time_of_pandemic(time_series_data: np.ndarray) -> int:
    """

    :param time_series_data: (n_days, 8), where the 8 spots are for: deaths_hr, recoveries_hr, infections_hr,
    vaccinations_hr, deaths_lr, recoveries_lr, infections_lr, vaccinations_lr
    :return: last day of the pandemic (max of: date of last recovery or death) - as a proportion of a year
    """
    hr_deaths = np.nonzero(time_series_data[:, 0])[0]
    last_hr_death = hr_deaths[-1] if hr_deaths.size != 0 else 0
    hr_recoveries = np.nonzero(time_series_data[:, 1])[0]
    last_hr_recovery = hr_recoveries[-1] if hr_recoveries.size != 0 else 0
    lr_deaths = np.nonzero(time_series_data[:, 4])[0]
    last_lr_death = lr_deaths[-1] if lr_deaths.size != 0 else 0
    lr_recoveries = np.nonzero(time_series_data[:, 5])[0]
    last_lr_recovery = lr_recoveries[-1] if lr_recoveries.size != 0 else 0
    return max((last_hr_death, last_hr_recovery, last_lr_death, last_lr_recovery)) / time_series_data.shape[0]


def get_max_infected_ratio(time_series_data: np.ndarray, num_nodes: int) -> Tuple[float, float, float]:
    """

    :param num_nodes: total number of nodes in the graph
    :param time_series_data: (n_days, 8), where the 8 spots are for: deaths_hr, recoveries_hr, infections_hr,
    vaccinations_hr, deaths_lr, recoveries_lr, infections_lr, vaccinations_lr
    :return: max ratio of infected people: total, high risk, low risk
    """
    lr = 0
    hr = 0
    max_total = 0
    max_lr = 0
    max_hr = 0
    for d in range(time_series_data.shape[0]):
        hr += time_series_data[d, 2]
        hr -= time_series_data[d, 0]
        hr -= time_series_data[d, 1]
        lr += time_series_data[d, 6]
        lr -= time_series_data[d, 4]
        lr -= time_series_data[d, 5]
        total = lr + hr
        if lr > max_lr:
            max_lr = lr
        if hr > max_hr:
            max_hr = hr
        if total > max_total:
            max_total = total
    return max_total / num_nodes, max_hr / num_nodes, max_lr / num_nodes


def combine_data_from_experiment(g: nx.Graph, ts_data: np.array) -> dict:
    """
    assumes: columns: ['n', 'hr_com_size', 'lr_com_size', "lr_prop_per_com", "degree_dist", "deg_dist_param",
               "com_dist_param", "max_vaccine_threshold", "vacc_strategy", "seed", "end",
               "peak", "peak_hr", "peak_lr", "dead", "dead_hr", "dead_lr", "rec",
               "rec_hr", "rec_lr", "vacc", "vacc_hr", "vacc_lr", "imu",
               "imu_hr", "imu_lr", "never_v", "never_v_hr", "never_v_lr"]

    :param g:
    :param ts_data:

    :return:
    """
    dead, recovered, vaccinated, never_virus, immune = collect_health_attr_stats(g=g, to_print=False)
    num_nodes = g.number_of_nodes()
    peaks = get_max_infected_ratio(time_series_data=ts_data, num_nodes=num_nodes)
    end = get_end_time_of_pandemic(time_series_data=ts_data)
    result = {
        "end": end, "peak": peaks[0], "peak_hr": peaks[1], "peak_lr": peaks[2], "dead": dead[0], "dead_hr": dead[1],
        "dead_lr": dead[2], "rec": recovered[0], "rec_hr": recovered[1], "rec_lr": recovered[2], "vacc": vaccinated[0],
        "vacc_hr": vaccinated[1], "vacc_lr": vaccinated[2], "imu": immune[0], "imu_hr": immune[1], "imu_lr": immune[2],
        "never_v": never_virus[0], "never_v_hr": never_virus[1], "never_v_lr": never_virus[2],
    }
    return result



