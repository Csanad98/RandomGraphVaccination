import networkx as nx
from matplotlib import pyplot as plt


def main():
    seed = 0
    sizes = [10, 10, 10]
    probs = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    g = nx.stochastic_block_model(sizes, probs, seed=0)
    pos = nx.spring_layout(g, seed=seed)  # Seed layout for reproducibility
    nx.draw(g, pos=pos)
    plt.show()


if "__main__" == __name__:
    main()
