import time
from typing import List

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
                                            print("experiment took: {:.2f}s".format(time.time() - t0))


if "__main__" == __name__:
    run_experiments(
        graph_sizes=[500],
        hr_community_sizes=[0.01, 0.05, 0.1],
        lr_community_sizes=[0.2],
        lr_ppl_per_hr_communities=[0.05, 0.025, 0.1],  # 20 ppl for one nurse, 10 ppl/ nurse, 5 ppl/ nurse
        degree_distributions=["power_law", "poisson"],
        degree_distribution_params={"power_law": [2.0, 2.5, 3.0], "poisson": [10, 25, 40]},
        community_deg_dist_params=[25],
        max_vaccine_thresholds=[0.7],
        vaccine_strategies=[i for i in range(7)],
        seeds=[i for i in range(5)]
    )
