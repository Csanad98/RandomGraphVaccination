import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def regression_without_interaction(data: pd.DataFrame, var: str):
    model = ols(
        "{} ~ C(n) + C(hr_com_size) + C(lr_com_size) + C(lr_prop_per_com) + C(deg_dist_param) "
        "+ C(com_dist_param) + C(max_vaccine_threshold) + C(vacc_strategy)".format(var),
        data=data).fit()

    # table = sm.stats.anova_lm(model)
    # print(table)
    print(model.summary(alpha=0.01))
    return model


def load_experiment_data(filter: int):
    df = pd.read_csv("data/experiment_data.csv")
    # df.columns = ['n', 'hr_com_size', 'lr_com_size', "lr_prop_per_com", "degree_dist", "deg_dist_param",
    #               "com_dist_param", "max_vaccine_threshold", "vacc_strategy", "seed", "end",
    #               "peak", "peak_hr", "peak_lr", "dead", "dead_hr", "dead_lr", "rec",
    #               "rec_hr", "rec_lr", "vacc", "vacc_hr", "vacc_lr", "imu",
    #               "imu_hr", "imu_lr", "never_v", "never_v_hr", "never_v_lr"]

    # use the line below to get data when random vaccination is the baseline
    # df_with_vaccine_rows_only = df[(df["vacc_strategy"] > 0)]

    # compare high degree first with the other two centrality measure based ones
    # df_with_vaccine_rows_only = df[(df["vacc_strategy"] == 3) | (df["vacc_strategy"] == 5) | (df["vacc_strategy"] == 6)]

    # compare two vaccination strategies with selected ids
    df_with_vaccine_rows_only = df[(df["vacc_strategy"] == 3) | (df["vacc_strategy"] == 4)]
    return df, df_with_vaccine_rows_only


if __name__ == "__main__":
    filter = 4
    df, df_vaccine_rows_only = load_experiment_data(0)
    variables_with_risk_groups = ["end", "peak", "peak_hr", "peak_lr", "dead", "dead_hr", "dead_lr", "rec", "rec_hr",
                                  "rec_lr", "vacc",  "vacc_hr", "vacc_lr", "imu", "imu_hr", "imu_lr", "never_v",
                                  "never_v_hr", "never_v_lr"]
    variables = ["peak", "peak_hr", "dead", "dead_hr"]
    # compare against no vaccination
    # for var in variables:
    #     model = regression_without_interaction(data=df, var=var)

    # compare against random vaccination
    for var in variables:
        model = regression_without_interaction(data=df_vaccine_rows_only, var=var)
