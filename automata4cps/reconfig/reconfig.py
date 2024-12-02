"""
CPS reconfiguration is a process of finding a configuration that will bring/maintain the system in the desired state.
"""
import networkx as nx


class Reconfigurator:
    """

    """
    def __init__(self, system_model):
        self.system_model = system_model

    def reconfigure(self, state):
        pass


def example_two_tank_system():
    edges = [("bv01", "l1", 1), ("bp21", "l1", 1), ("bp12", "l2", 1),
                     ("bv02", "l2", 1), ("bh1", "v1", 1), ("bh2", "v1", 1),
                     ("bv10", "l1", -1), ("bp12", "l1", -1), ("bp21", "l2", -1),
                     ("bv20", "l2", -1), ("bp21", "v1", -1), ("bv02", "v2", -1),
                     ("bc1", "v2", -1), ("bc2", "v2", -1)]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    SM = example_two_tank_system()

    from automata4cps.vis import plot_bipartite_graph

    plot_bipartite_graph(SM)