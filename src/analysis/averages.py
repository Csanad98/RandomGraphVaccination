import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


if __name__ == "__main__":
    df = pd.read_csv("data/experiment_data.csv")
    variables_with_risk_groups = ["end", "peak", "peak_hr", "peak_lr", "dead", "dead_hr", "dead_lr", "rec", "rec_hr",
                                  "rec_lr", "vacc",  "vacc_hr", "vacc_lr", "imu", "imu_hr", "imu_lr", "never_v",
                                  "never_v_hr", "never_v_lr"]
    variables = ["peak", "peak_hr", "dead", "dead_hr"]
    ylabels = ["Peak infection ratio", "HR peak infection ratio", "Death ratio", "HR death ratio"]
    for i in range(len(variables)):
        combined_df = []
        for vacc_strategy in range(7):
            print("Strategy id: {}".format(vacc_strategy))
            cur_df = df[df["vacc_strategy"] == vacc_strategy]
            print(cur_df[variables[i]].describe())
            combined_df.append(cur_df[variables[i]])
        fig, ax = plt.subplots()
        ax.set_title("Vaccination strategy comparison: {}".format(ylabels[i]))
        ax.boxplot(combined_df)
        ax.set_xticklabels(["None", "Random", " Risk Group\nBiased", "High \n degree", "Ring", "Between-\nness", "Closeness"])
        ax.set_ylabel(ylabels[i])
        plt.savefig("boxplot_{}.png".format(ylabels[i]))
        plt.show()
