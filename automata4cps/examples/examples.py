import datetime
from automata4cps import Automaton, tools
import numpy as np
import pandas as pd
import os
import math


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
    columns_16bit = ['O_w_HAL_Ctrl', 'O_w_HAR_Ctrl']

    file_path = os.path.dirname(os.path.abspath(__file__))
    log1 = pd.read_csv(os.path.join(file_path, "data", "log1.csv"))
    log2 = pd.read_csv(os.path.join(file_path, "data", "log2.csv"))
    data = [log1, log2]

    cont_cols = [c for c in data[0].columns if c.lower()[-5:] != '_ctrl' and c != "timestamp" and 'energy' not in c and
                 "energie" not in c]
    discrete_cols = [c for c in data[0].columns if '_Ctrl' in c]
    # num_bits = {c: max([math.ceil(math.log2(d[c].max())) for d in data]) for c in discrete_cols}

    for d in data:
        d.drop(columns=[c for c in d.columns if "energy" in c or "energie" in c], axis=1, inplace=True)

    # reformat timestamp
    for i, log in enumerate(data):
        log['timestamp_new'] = (datetime.datetime(1, 1, 1)) + log['timestamp'].apply(lambda x: datetime.timedelta(seconds=x))
        log['timestamp'] = pd.to_datetime(log['timestamp_new'])
        log.drop(['timestamp_new'], axis=1, inplace=True)


        # series_16bit = log[col].apply(lambda x: list(format(x, f'{num_bits[col]:03d}b')))
        # binary_df = pd.DataFrame(series_16bit.tolist(), columns=[f'{col}_bit_{i}' for i in range(num_bits[col])]).astype(int)
        # data[i].drop([col], axis=1, inplace=True)
        # data[i] = pd.concat([data[i], binary_df], axis=1)
    data = tools.encode_nominal_list_df(data, columns=discrete_cols)

    discrete_cols = [c for c in data[0].columns if '_Ctrl' in c]

    # remove constant bits
    # constant_cols = [c for c in discrete_cols if 1 == len(set(item for sublist in ([d[c].unique() for d in data]) for item in sublist))]
    # for d in data:
    #     d.drop(columns=constant_cols, axis=1, inplace=True)
    # discrete_cols = [d for d in discrete_cols if d not in constant_cols]


    # Adding the Path/Weg variable
    for d in data:
        control_sig_1 = d['O_w_BRU_Axis_Ctrl_1'].to_numpy()
        control_sig_3 = d['O_w_BRU_Axis_Ctrl_3'].to_numpy()
        ind = np.logical_and(control_sig_1[0:-1] == 1, control_sig_3[1:] == 1)
        ind = np.nonzero(ind)[0] + 1
        ind = [0] + list(ind) + [d.shape[0]]
        d["Weg"] = 0.
        for n in range(len(ind) - 1):
            # cc = c.iloc[ind[n]:ind[n + 1]].copy()
            time_diff = d['timestamp'].iloc[min(ind[n+1], d.shape[0]-1)] - d['timestamp'].iloc[ind[n]]
            if time_diff < datetime.timedelta(seconds=13.5):
                d.iloc[ind[n]:ind[n + 1], d.columns.get_loc("Weg")] = 1
    discrete_cols.append("Weg")

    if variable_type == "discrete":
        discrete_data = [d[['timestamp'] + discrete_cols] for d in data]
        return discrete_data, "timestamp", discrete_cols
    elif variable_type == "continuous":
        cont_data = [d[['timestamp'] + cont_cols] for d in data]
        return cont_data, "timestamp", cont_cols
    else:
        return data, "timestamp", discrete_cols, cont_cols


if __name__ == "__main__":

    data = conveyor_system_sfowl()
    exit()
    # A = timed_control()
    # A.simulate(finish_time=500)

    A = simple_conveyor_system()
    A.plot_cps().show()
    ddata = A.simulate(finish_time=500)

    A = Automaton()
    A.add_state(["s1", "s2", "s3"])
    A.add_transition([("s1", "s2", "e1"),
                      ("s2", "s3", "e1"),
                      ("s3", "s1", "e2")])

    print(A)
    A.plot_cps().show()
