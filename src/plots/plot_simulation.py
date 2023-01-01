import numpy as np
from matplotlib import pyplot as plt


def plot_ts_for_group(ts_data: np.ndarray, n_days: int, group: str = "high risk people"):
    plt.title("Time series data for {}".format(group))
    plt.plot(range(n_days), ts_data[:, 0], label="death")
    plt.plot(range(n_days), ts_data[:, 1], label="recovery")
    plt.plot(range(n_days), ts_data[:, 2], label="infection")
    plt.plot(range(n_days), ts_data[:, 3], label="vaccination")
    plt.legend()
    plt.show()


def plot_ts_data_for_each_group(ts_data: np.ndarray, n_days: int):
    plot_ts_for_group(ts_data=ts_data[:, :4], n_days=n_days, group="high risk people")
    plot_ts_for_group(ts_data=ts_data[:, 4:], n_days=n_days, group="low risk people")
    for i in range(4):
        ts_data[:, i] += ts_data[:, i+4]
    plot_ts_for_group(ts_data=ts_data[:, :4], n_days=n_days, group="everyone (risk groups combined)")
