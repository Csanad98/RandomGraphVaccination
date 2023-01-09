import time
from typing import List

import pandas as pd

from analysis.stats import collect_health_attr_stats, get_max_infected_ratio, combine_data_from_experiment
from process.individual_simulation import single_graph_simulation, single_graph_generator


# things to vary:
# 1 seeds (change seed)
# 2 vaccine strategies (on each graph)
# 3 degree distribution: power-law, poisson
# 4 degree distribution parameter
# 5 proportion of initially infected people
# 6 proportion of initially infected people who are high/low risk (skip: just do 50-50)
# 7 lr community size (proportion of total population)
# 8 community sizes
# 9 max vaccination threshold of population
# 10 proportion of lr people in hr communities (how many nurses per patient?)
# 11 graph size

def run_experiments(graph_sizes: List[int],
                    hr_community_sizes: List[float],
                    lr_community_sizes: List[float],
                    lr_ppl_per_hr_communities: List[float],
                    degree_distributions: List[str],
                    degree_distribution_params: dict[str: List[float]],
                    community_deg_dist_params: List[float],
                    max_vaccine_thresholds: List[float],
                    vaccine_strategies: List[int],
                    seeds: List[int]):
    t00: float = time.time()
    num_exp = len(graph_sizes) * len(hr_community_sizes) * len(lr_community_sizes) * \
              len(lr_ppl_per_hr_communities) * len(degree_distributions) * \
              len(degree_distribution_params["power_law"]) * len(community_deg_dist_params) * \
              len(max_vaccine_thresholds) * len(vaccine_strategies) * len(seeds)
    i = 1

    columns = ['n', 'hr_com_size', 'lr_com_size', "lr_prop_per_com", "degree_dist", "deg_dist_param",
               "com_dist_param", "max_vaccine_threshold", "vacc_strategy", "seed", "end",
               "peak", "peak_hr", "peak_lr", "dead", "dead_hr", "dead_lr", "rec",
               "rec_hr", "rec_lr", "vacc", "vacc_hr", "vacc_lr", "imu",
               "imu_hr", "imu_lr", "never_v", "never_v_hr", "never_v_lr"]

    row_list = []

    for n in graph_sizes:
        for hr_com_size in hr_community_sizes:
            for lr_com_size in lr_community_sizes:
                for lr_prop_per_com in lr_ppl_per_hr_communities:
                    for degree_dist in degree_distributions:
                        for deg_dist_param in degree_distribution_params[degree_dist]:
                            for com_dist_param in community_deg_dist_params:
                                for max_vaccine_threshold in max_vaccine_thresholds:
                                    for vacc_strategy in vaccine_strategies:
                                        for seed in seeds:
                                            t0: float = time.time()
                                            g = single_graph_generator(seed=seed,
                                                                       n=n,
                                                                       lam_out=deg_dist_param,
                                                                       lam_in=com_dist_param,
                                                                       degree_distribution=degree_dist,
                                                                       prop_lr_com_size=lr_com_size,
                                                                       prop_com_size=hr_com_size,
                                                                       prop_int_inf=0.005,  # 0.5% of population
                                                                       prop_int_inf_hr=0.5,
                                                                       prop_hr_hr=1 - lr_prop_per_com,
                                                                       prop_hr_lr=0,
                                                                       vacc_app_prob=max_vaccine_threshold,
                                                                       t0=t0)
                                            g, ts_data = single_graph_simulation(seed=seed,
                                                                                 g=g,
                                                                                 n_days=365,
                                                                                 vaccination_strategy=vacc_strategy,
                                                                                 max_vacc_threshold=max_vaccine_threshold,
                                                                                 t0=t0)
                                            stats = combine_data_from_experiment(g=g, ts_data=ts_data)
                                            add_exp_prams(row_dict=stats, n=n, hr_com_size=hr_com_size,
                                                          lr_com_size=lr_com_size, lr_prop_per_com=lr_prop_per_com,
                                                          degree_dist=degree_dist, deg_dist_param=deg_dist_param,
                                                          com_dist_param=com_dist_param,
                                                          max_vaccine_threshold=max_vaccine_threshold,
                                                          vacc_strategy=vacc_strategy, seed=seed)
                                            row_list.append(stats)
                                            print("experiment took: {:.2f}s, experiment: {}/{}, mean exp time: {:.2f}s"
                                                  .format(time.time() - t0, i, num_exp, (time.time() - t00) / i))
                                            i += 1
    df = pd.DataFrame(row_list, columns=columns)
    df.to_csv("experiment_data.csv", index=False)
    print("all experiments took: {:.2f}s".format(time.time() - t00))


def add_exp_prams(row_dict: dict, n: int, hr_com_size: float, lr_com_size: float, lr_prop_per_com: float,
                  degree_dist: str, deg_dist_param: float, com_dist_param: float, max_vaccine_threshold: float,
                  vacc_strategy: int, seed: int):
    row_dict["n"] = n
    row_dict["hr_com_size"] = hr_com_size
    row_dict["lr_com_size"] = lr_com_size
    row_dict["lr_prop_per_com"] = lr_prop_per_com
    row_dict["degree_dist"] = degree_dist
    row_dict["deg_dist_param"] = deg_dist_param
    row_dict["com_dist_param"] = com_dist_param
    row_dict["max_vaccine_threshold"] = max_vaccine_threshold
    row_dict["vacc_strategy"] = vacc_strategy
    row_dict["seed"] = seed


if "__main__" == __name__:
    run_experiments(
        graph_sizes=[500],
        hr_community_sizes=[0.05, 0.1],
        lr_community_sizes=[0.2, 0.4],
        lr_ppl_per_hr_communities=[0.05, 0.1],  # 20 ppl for one nurse, 10 ppl/ nurse
        degree_distributions=["power_law"],
        degree_distribution_params={"power_law": [2.0, 2.5, 3.0]},
        community_deg_dist_params=[20, 40],
        max_vaccine_thresholds=[0.5, 0.65, 0.8],
        vaccine_strategies=[i for i in range(7)],
        seeds=[i for i in range(3)]
    )
