import numpy as np
from matplotlib import pyplot as plt


def plot_degree_seq_comparison(deg_seq_a: np.array, deg_seq_b: np.array, sort_sequences: bool = False):
    """
    :param sort_sequences:
    :param deg_seq_a:
    :param deg_seq_b:
    :return:
    """
    if sort_sequences:
        deg_seq_a = sorted(deg_seq_a, reverse=True)
        deg_seq_b = sorted(deg_seq_b, reverse=True)

    fig, ax = plt.subplots()
    ax.plot(deg_seq_a, color='green', label='Realized degree sequence')
    ax.plot(deg_seq_b, color='red', label='True degree sequence')
    ax.legend(loc='upper right')
    plt.show()
