import numpy as np
import networkx as nx


def create_half_edges_between_communities(deg_seq_out, communities):
    # make list of n empty lists where n the number of communities
    half_edges = [[] for _ in range(len(np.unique(communities)))]

    # setup half edge map
    for v_index in range(len(deg_seq_out)):  # for all nodes
        for h in range(deg_seq_out[v_index]):  # for all half_edges of node
            half_edges[int(communities[v_index])].append(v_index)  # add node to list of its community
    return half_edges


def is_community_structure_possible(deg_seq_out, communities):
    # for each community the number of out h.e.-s cannot exceed the total number of h.e.-s from every other community
    # half_edges = create_half_edges_between_communities(deg_seq_out=deg_seq_out, communities=communities)
    community_ids = np.unique(communities)
    for c in community_ids:
        vertex_ids_in_c = np.where(communities == c)[0]
        vertex_ids_not_in_c = np.where(communities != c)[0]
        assert np.sum(deg_seq_out[vertex_ids_in_c.astype(int)]) <= np.sum(deg_seq_out[vertex_ids_not_in_c.astype(int)]), \
            "for each community the number of out h.e.-s cannot exceed the total number of h.e.-s from every other " \
            "community"


def cm_for_communities(deg_seq_in: np.array,
                       communities: np.array,
                       graph: nx.Graph,
                       seed=0):
    # Run configuration model for each community, use degree sequence meant for within communities
    for c in np.unique(communities):
        # get vertex ids for current community
        vertex_ids_for_c = np.where(communities == c)[0]  # index 0 since we have only one dimension
        # call nx.Graph to get a (non-erased) Configuration Model Multigraph
        community_sub_graph = nx.configuration_model(deg_sequence=deg_seq_in[vertex_ids_for_c], seed=seed)
        community_sub_graph.remove_edges_from(nx.selfloop_edges(community_sub_graph))  # remove self loops
        # Remove Parallel Edges by turning the Multigraph into a Graph
        community_sub_graph = nx.Graph(community_sub_graph)
        graph = nx.disjoint_union(graph, community_sub_graph)
    return graph
