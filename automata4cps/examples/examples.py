from automata4cps import Automaton
import numpy as np
import pandas as pd
import os


def timed_control():
    A = Automaton()
    A.add_state(["q1", "q2"])
    A.add_transition([("q1", "q2", "e1"),
                      ("q2", "q1", "e2")])
    return A


def simple_conveyor_system():
    A = Automaton()

    def timed_transition():
        return np.random.normal(5, 0.1)

    def output_fun(mean, cov):
        f = lambda: np.random.multivariate_normal(mean, cov)
        return f

    A.add_state("s1", output_fun=output_fun([0, 0], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s2", output_fun=output_fun([2, 0], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s3", output_fun=output_fun([0, 2], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s4", output_fun=output_fun([0, 2], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s5", output_fun=output_fun([1, 1], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s6", output_fun=output_fun([1, 1], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s7", output_fun=output_fun([0, 0], [[1, 0], [0, 1]]), sample_time=0.1)
    A.add_state("s8", output_fun=output_fun([2, 0], [[1, 0], [0, 1]]), sample_time=0.1)

    A.add_transition("s1", "s2", "Put item", weight=1, timed=timed_transition)
    A.add_transition("s2", "s1", "Remove item", weight=1, timed=timed_transition)
    A.add_transition("s7", "s8", "Put item", weight=1, timed=timed_transition)
    A.add_transition("s8", "s7", "Remove item", weight=1, timed=timed_transition)
    A.add_transition("s1", "s4", "Go down", weight=1, timed=timed_transition)
    A.add_transition("s4", "s7", "Stop", weight=1, timed=timed_transition)
    A.add_transition("s7", "s3", "Go up", weight=1, timed=timed_transition)
    A.add_transition("s3", "s1", "Stop", weight=1, timed=timed_transition)
    A.add_transition("s2", "s6", "Go down", weight=1, timed=timed_transition)
    A.add_transition("s6", "s8", "Stop", weight=1, timed=timed_transition)
    A.add_transition("s8", "s5", "Go up", weight=1, timed=timed_transition)
    A.add_transition("s5", "s2", "Stop", weight=1, timed=timed_transition)
    return A


def conveyor_system_sfowl(variable_type="all"):
    file_path = os.path.dirname(os.path.abspath(__file__))
    log1 = pd.read_csv(os.path.join(file_path, "data", "log1.csv"))
    log2 = pd.read_csv(os.path.join(file_path, "data", "log2.csv"))
    data = [log1, log2]
    discrete_cols = [c for c in data[0].columns if c.lower()[-5:] == '_ctrl']
    cont_cols = [c for c in data[0].columns if c.lower()[-5:] != '_ctrl' and c != "timestamp"]

    if variable_type == "discrete":
        discrete_data = [d[['timestamp'] + discrete_cols] for d in data]
        return discrete_data, "timestamp", discrete_cols
    elif variable_type == "continuous":
        cont_data = [d[['timestamp'] + cont_cols] for d in data]
        return cont_data, "timestamp", cont_cols
    else:
        return data, "timestamp", discrete_cols, cont_cols


if __name__ == "__main__":
    # A = timed_control()
    # A.simulate(finish_time=500)

    A = simple_conveyor_system()
    A.view_plotly().show()
    ddata = A.simulate(finish_time=500)

    A = Automaton()
    A.add_state(["s1", "s2", "s3"])
    A.add_transition([("s1", "s2", "e1"),
                      ("s2", "s3", "e1"),
                      ("s3", "s1", "e2")])

    print(A)
    A.view_plotly().show()
