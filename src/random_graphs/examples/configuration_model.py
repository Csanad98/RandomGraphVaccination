import networkx as nx
from matplotlib import pyplot as plt


def main():
    seed = 0
    deg_seq = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    G = nx.configuration_model(deg_sequence=deg_seq, seed=seed)
    pos = nx.spring_layout(G, seed=seed)  # Seed layout for reproducibility
    nx.draw(G, pos=pos)
    plt.show()

    
if "__main__" == __name__:
    main()
