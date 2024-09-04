"""
    Various methods to transform data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import torch
from datetime import datetime
import dash_cytoscape as cyto
import pprint


def remove_timestamps_without_change(data, sig_names):
    """Removes timestamps where no values changed in comparison to the previous timestamp."""
    new_data = []
    for d in data:
        ind = (d[sig_names].diff() != 0).any(axis=1)
        dd = d.loc[ind]
        new_data.append(dd.copy(deep=True))
    return new_data


def filter_signals(data, sig_names):
    new_data = []
    for d in data:
        d = d[sig_names]
        new_data.append(d)
    return data


def create_events_from_signal_vectors(data, sig_names):
    for d in data:
        d.loc[:, "event"] = d[sig_names].diff().apply(lambda x: ' '.join(x.astype(str)).replace(".0", ""), 1)
        # d.drop(columns=signals)
        # d["dt"] = d[time].diff().shift(-1)
        # new_data.append(d)
    return data


def group_data_on_discrete_state(data, state_column, reset_time=False, time_col=None):
    sequences = dict()
    for d in data:
        for g, dd in d.groupby(d[state_column].ne(d[state_column].shift()).cumsum()):
            state = dd[state_column].iloc[0]
            if state not in sequences:
                sequences[state] = list()
            if reset_time:
                dd[time_col] -= dd[time_col].iloc[0]
            sequences[state].append(dd)
    return sequences


def split_data_on_signal_value(data, sig_name, new_value):
    new_data = []
    for d in data:
        d["splitting_group"] = ((d[sig_name] == new_value) & (d[sig_name].shift(1) != new_value)).cumsum()
        for g, dd in d.groupby(d["splitting_group"]):
            new_data.append(dd)
    return new_data


def split_train_valid(time, data, other, split):
    # if lists are passed than returns lists, otherwise numpy arrays
    train_other = None
    valid_other = None

    if type(data) is list:
        split_ind = int(split * len(data))

        train_time = time[:split_ind]
        valid_time = time[split_ind:]

        train_data = data[:split_ind]
        valid_data = data[split_ind:]

        if other is not None:
            train_other = other[:split_ind]
            valid_other = other[split_ind:]
    else:
        split_ind = int(split * data.size(dim=0))

        train_time = time[:split_ind]
        valid_time = time[split_ind:]

        train_data = data[:split_ind, :]
        valid_data = data[split_ind:, :]

        if other is not None:
            train_other = other.iloc[:split_ind, :].to_numpy()
            valid_other = other.iloc[split_ind:, :].to_numpy()

    return train_time, valid_time, train_data, valid_data, train_other, valid_other


def filter_na_and_constant(data):
    for i in range(len(data)):
        data[i] = data[i][:,
                  ~torch.any(data[i].isnan() | data[i].isinf(), dim=0).cpu() & (data[i].std(dim=0) != 0).cpu()]
    return data


def flatten_dict(dict_of_lists):
    return [item for sublist in dict_of_lists.values() for item in sublist]


def get_binary_cols(df):
    binary_columns = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]
    return binary_columns


def melt_dataframe(df, timestamp=None):
    if timestamp is None:
        timestamp = df.columns[0]

    # Set timestamp as the index
    df = df.set_index(timestamp)

    # Initialize an empty list to store the changes
    changes = []

    # Iterate through columns and detect changes
    for col in df.columns:
        # Get the boolean series where the current value is different from the previous value
        diff = df[col].ne(df[col].shift())
        # Collect the changes in a DataFrame
        changes_col = df.loc[diff, [col]].reset_index()
        changes_col['variable'] = col
        changes_col = changes_col.rename(columns={col: 'value'})
        changes.append(changes_col)

    # Concatenate all changes
    result = pd.concat(changes).sort_values(by=[timestamp, 'variable']).reset_index(drop=True)

    return result


def compute_purity(cluster_assignments, class_assignments):
    num_samples = len(cluster_assignments)
    valid_values = np.asarray(pd.notna(cluster_assignments) & pd.notna(class_assignments))
    cluster_assignments = cluster_assignments[valid_values]
    class_assignments = class_assignments[valid_values]
    # cluster_class_counts = confusion_matrix(class_assignments[valid_values], cluster_assignments[valid_values])

    cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                            for cluster_ in np.unique(cluster_assignments)}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    total_intersection = sum([max(class_dict.values()) for cluster_, class_dict in cluster_class_counts.items()])

    purity = total_intersection / num_samples

    return purity


def composite_f1_score(anom_labels, start_event_idx, true_anom_idx):
    true_anomalies = np.array(true_anom_idx) != 0
    pred_anomalies = np.array(anom_labels) != 0
    # True Positives (TP): True anomalies correctly predicted as anomalies
    tp = np.sum([pred_anomalies[start_event_idx:].any()])
    # False Negatives (FN): True anomalies missed by the prediction
    fn = 1 - tp
    # Recall for events (Rec_e): Proportion of true anomalies correctly identified
    rec_e = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Precision for the entire time series (Prec_t)
    prec_t = precision_score(true_anomalies, pred_anomalies)
    # Composite F-score
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    else:
        fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    return fscore_c


def binary_ordinal_encode(column, order):
    """
    Encodes a pandas Series with binary ordinal encoding based on the specified order.

    Args:
        column (pd.Series): The column to encode.
        order (list): The ordered list of unique values in the column.

    Returns:
        pd.DataFrame: The binary ordinal encoded DataFrame for the given column.
    """
    num_levels = len(order)
    num_bits = num_levels.bit_length()

    # Create a dictionary mapping each level to its binary representation
    encoding_map = {value: list(map(int, format(i, f'0{num_bits}b'))) for i, value in enumerate(order)}

    # Apply the encoding to the column
    encoded_df = pd.DataFrame(column.map(encoding_map).tolist(),
                              columns=[f"{column.name}_bit_{i}" for i in range(num_bits)])
    return encoded_df


def encode_ordinal(x, columns, order=None):
    if columns is None:
        columns = x.columns
    if len(columns) == 1 and type(order) is not dict:
        order = {columns[0]: order}
    new_data = []
    for c in columns:
        new_data.append(binary_ordinal_encode(x[c], order[c]))

    # Concatenate the encoded columns with the original DataFrame (excluding the original ordinal columns)
    return pd.concat([x.drop(columns, axis=1)] + new_data, axis=1)


def encode_nominal(x, columns=None, categories=None):
    if columns is None:
        columns = x.columns
    if categories is None:
        categories = dict()
    for c in columns:
        x[c] = pd.Categorical(x[c], categories=categories.get(c, None))
    new_dataset = pd.get_dummies(x, columns=columns, dtype=float)

    # Get the original columns
    original_columns = x.columns

    # Get the new columns after encoding
    new_columns = new_dataset.columns

    # Determine the newly created columns by `pd.get_dummies`
    newly_created_columns = set(new_columns) - set(original_columns)

    # Create a mapping from original columns to their new dummy columns
    mapping = {col: [dummy_col for dummy_col in newly_created_columns if dummy_col.startswith(f"{col}_")]
               for col in columns}
    return new_dataset, mapping


def encode_nominal_list_df(dfs, columns=None, categories=None):
    if columns is None:
        columns = list(pd.concat(dfs.columns).unique())
    if categories is None:
        categories = dict.fromkeys(columns)
        for c in columns:
            categories[c] = np.sort(list(np.unique(np.concatenate([df[c].unique() for df in dfs]))))
    return [encode_nominal(x, columns, categories=categories)[0] for x in dfs]


def dict_to_df(d):
    max_rows = max(len(v) for v in d.values())
    return pd.DataFrame({k: v + [None] * (max_rows - len(v)) for k, v in d.items()})


def dict_to_csv(d, name="csv.csv"):
    max_rows = max(len(v) for v in d.values())
    pd.DataFrame({k: v + [None] * (max_rows - len(v)) for k, v in d.items()}).to_csv(name, sep=";", index=False)


def data_list_to_dataframe(element, data, signal_names, prefix=None, last_var=None):
    data = pd.DataFrame(data)
    if data.shape[1] == 0:
        return data
    if signal_names is None:
        signal_names = [prefix + str(i + 1) for i in range(len(data.columns) - 1)]
        if last_var is not None and len(signal_names) > 0:
            signal_names[-1] = last_var
    else:
        if last_var is not None:
            signal_names += [last_var + str(i + 1) for i in range(len(data.columns) - len(signal_names) - 1)]
    signal_names.insert(0, 'Time')

    if len(data.columns) != len(signal_names):
        raise Exception('Wrong number of column names: length of {} != {}'.format(signal_names, len(data.columns)))

    data.columns = signal_names
    data.set_index(data.columns[0], inplace=True)
    if element is not None:
        data.columns = element + '.' + data.columns
    return data


def flatten_dict_data(stateflow, reduce_keys_if_possible=True):
    d = pd.json_normalize(stateflow).to_dict('records')[0]
    if reduce_keys_if_possible:
        for k in list(d.keys()):
            k_new = k.split('.')[-1]
            d[k_new] = d.pop(k)
    return d


def group_components(comp, *states):
    if type(comp) is str:
        return
    res = ([k for k in comp if comp[k] == state]
           for state in states)
    if len(states) == 1:
        res = next(res)
        return res
    else:
        return tuple(res)


def signal_vector_to_state(sig_vec):
    return pprint.pformat(sig_vec, compact=True)


def signal_vector_to_event(previous_vec, sig_vec):
    return


if __name__ == "__main__":
    # from automata4cps.examples import examples
    #
    # data = examples.high_rack_storage_system_sfowl()
    # data = split_data_on_signal_value(data, sig_name="O_w_BRU_Axis_Ctrl", new_value=3)
    X = pd.DataFrame({"Col1": [1, 2, 3], "Col2": [2, 3, 2]})
    Xe = encode_nominal(X, columns=["Col1", "Col2"])