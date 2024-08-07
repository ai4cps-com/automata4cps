import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from plotly import colors as clr
from sklearn.metrics import precision_score, confusion_matrix
import torch
from datetime import datetime
import dash_cytoscape as cyto


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


def plot_data(data, title=None, timestamp=None, use_columns=None, discrete=False, height=None, plotStates=False,
              limit_num_points=None, names=None, xaxis_title=None,
              customdata=None, iterate_colors=True, y_title_font_size=14, opacity=1, vertical_spacing=0.005,
              sharey=False, bounds=None, plot_only_changes=False, yAxisLabelOffset=False, marker_size=4,
              showlegend=False, mode='lines+markers', **kwargs):
    """
    Plots all variables in the dataframe as subplots.

    plotStates: if True, changes interpolation to 'hv', therefore keeping the last value
    yAxisLabelOffset: if True, adds a offset to the plots y axis labels. Improves readability on long subplot names.
    """

    if limit_num_points is None or limit_num_points < 0:
        limit_num_points = np.inf
    if customdata is not None:
        customdata = customdata.fillna('')
    if type(data) is not list:
        data = [data]

    if len(data) == 0:
        return None

    # if not panda data frame
    for i in range(len(data)):
        if not isinstance(data[i], pd.DataFrame):
            data[i] = pd.DataFrame(data[i])

    # if no timestamp is in the data
    if timestamp is not None:
        if type(timestamp) == str or type(timestamp) == int:
            for i in range(len(data)):
                data[i] = data[i].set_index(timestamp)

    if height is None:
        height = max(800, len(data[0].columns) * 60)

    if use_columns is None:
        columns = data[0].columns
    else:
        columns = use_columns

    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
                        shared_yaxes=sharey)

    # select line_shape:
    if plotStates is True:
        lineShape = 'hv'
    else:
        lineShape = 'linear'

    # Add traces
    i = 0

    for col_ind in range(len(columns)):
        i += 1
        k = -1
        for trace_ind, d in enumerate(data):
            col_name = columns[col_ind]
            col = d.columns[col_ind]
            if names:
                trace_name = names[trace_ind]
            else:
                trace_name = str(trace_ind)
            if use_columns is not None and col_name not in use_columns:
                continue

            hovertemplate = f"<b>Time:</b> %{{x}}<br><b>Event:</b> %{{y}}"
            if customdata is not None:
                hovertemplate += "<br><b>Context:</b>"
                for ind, c in enumerate(customdata.columns):
                    hovertemplate += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<em>{c}:</em> %{{customdata[{ind}]}}"

            k += 1
            if iterate_colors:
                color = clr.DEFAULT_PLOTLY_COLORS[k % len(clr.DEFAULT_PLOTLY_COLORS)]
            else:
                color = clr.DEFAULT_PLOTLY_COLORS[0]

            color = f'rgba{color[3:-1]}, {str(opacity)})'
            if len(d.index.names) > 1:
                t = d.index.get_level_values(d.index.names[-1]).to_numpy()
            else:
                t = d.index.to_numpy()
            if d[col].dtype == tuple:
                sig = d[col].astype(str).to_numpy()
            else:
                sig = d[col].to_numpy()
            if discrete:
                ind = min(limit_num_points, d.shape[0])
                if plot_only_changes:
                    ind = np.nonzero(np.not_equal(sig[0:ind - 1], sig[1:ind]))[0] + 1
                    # sig = __d[col][0:min(limit_num_points, __d.shape[0])]
                    ind = np.insert(ind, 0, 0)
                    t = t[ind]
                    sig = sig[ind]
                    if customdata is not None:
                        customdata = customdata[ind]
                else:
                    t = t[0:ind]
                    sig = sig[0:ind]
                    if customdata is not None:
                        customdata = customdata[0:ind]

                fig.add_trace(go.Scatter(x=t, y=sig, mode='markers',
                                         name=trace_name, marker=dict(line_color=color, color=color,
                                                                    line_width=2, size=marker_size),
                                         customdata=customdata,
                                         hovertemplate=hovertemplate, **kwargs), row=i, col=1)
            else:
                ind = min(limit_num_points, d.shape[0])
                fig.add_trace(go.Scatter(x=t[0:ind], y=sig[0:ind], mode=mode, name=trace_name, customdata=customdata,
                                         line=dict(color=color), line_shape=lineShape, **kwargs), row=i, col=1)
            fig.update_yaxes(title_text=str(col_name), row=i, col=1, title_font=dict(size=y_title_font_size),
                             categoryorder='category ascending')
        if i % 2 == 0:
            fig.update_yaxes(side="right", row=i, col=1)
        if yAxisLabelOffset == True:
            fig.update_yaxes(title_standoff=10 * i, row=i, col=1)
        if xaxis_title is not None:
            fig.update_xaxes(title=xaxis_title)
        if bounds is not None:
            upper_col = bounds[0].iloc[:, col_ind]
            lower_vol = bounds[1].iloc[:, col_ind]
            upper_bound = go.Scatter(
                name='Upper Bound',
                x=bounds[0].index.get_level_values(-1),
                y=upper_col,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False)
            lower_bound = go.Scatter(
                name='Lower Bound',
                x=bounds[1].index.get_level_values(-1),
                y=lower_vol,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False)
            fig.add_trace(upper_bound, row=i, col=1)
            fig.add_trace(lower_bound, row=i, col=1)
    fig.update_layout(title={'text': title, 'x': 0.5}, autosize=True, height=height + 180, showlegend=showlegend)
    return fig


def plot_execution_tree(graph, nodes_to_color, color, font_size=30):
    # The function plots system execution in form of a graph, where horizontal position of the nodes corresponds to the
    # node's timestamp. The tree branches vertically.

    # for ntd in nodes_to_delete:
    #     if ntd in graph:
    #         prenodes = list(graph.predecessors(ntd))
    #         sucnodes = list(graph.successors(ntd))
    #         preedges = list(graph.in_edges(ntd))
    #         sucedges = list(graph.out_edges(ntd))
    #         edgestodelete = preedges + sucedges
    #         if ((len(preedges) > 0) and (len(sucedges) > 0)):
    #             for prenode in prenodes:
    #                 for sucnode in sucnodes:
    #                     graph.add_edge(prenode, sucnode)
    #         if (len(edgestodelete) > 0):
    #             graph.remove_edges_from(edgestodelete)

    startstring = list(graph.nodes)[0]
    arr_elements = []
    num_of_nodes = graph.number_of_nodes()
    # vertical_height = num_of_states
    visited = set()
    stack = [startstring]
    while stack:
        node = stack.pop()
        if node not in visited:
            elemid = str(node)
            elemlabel = graph.nodes[node].get('label')
            datepos1 = datetime.strptime(startstring, "%d/%m/%Y, %H:%M:%S")
            datepos2 = datetime.strptime(node, "%d/%m/%Y, %H:%M:%S")
            nodeweight = graph.nodes[node].get('weight')
            ypos = 0
            if nodeweight == 0:
                ypos = num_of_states * 100
            else:
                ypos = (nodeweight - 1) * 200
            element = {
                'data': {
                    'id': elemid,
                    'label': elemlabel
                },
                'position': {
                    'x': (datepos2 - datepos1).total_seconds() / 7200,
                    'y': ypos
                },
                # 'locked': True
            }
            arr_elements.append(element)
            visited.add(node)
            stack.extend(neighbor for neighbor in graph.successors(node) if neighbor not in visited)
    for u, v in list(graph.edges):
        edge_element = {
            'data': {
                'source': u,
                'target': v
            }
        }
        arr_elements.append(edge_element)


    colorcode = ['gray'] * num_of_nodes
    for n in nodes_to_color:
        if n in graph:
            n_ind = list(graph.nodes).index(n)
            if (n_ind < num_of_nodes):
                colorcode[n_ind] = color
    new_stylesheet = []
    for i in range(0, num_of_nodes):
        new_stylesheet.append({
            'selector': f'node[id = "{list(graph.nodes)[i]}"]',
            'style': {
                'font-size': f'{font_size}px',
                'content': 'data(label)',
                'background-color': colorcode[i],
                'text-valign': 'top',
                'text-halign': 'center',
                # 'animate': True
            }
        })

    cytoscapeobj = cyto.Cytoscape(
        id='org-chart',
        layout={'name': 'preset'},
        style={'width': '2400px', 'height': '1200px'},
        elements=arr_elements,
        stylesheet=new_stylesheet
    )
    return cytoscapeobj


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
    mapping = {col: [dummy_col for dummy_col in newly_created_columns if dummy_col.startswith(f"{col}_")] for col in
               columns}

    return new_dataset, mapping

def encode_nominal_list_df(dfs, columns=None, categories=None):
    if columns is None:
        columns = list(pd.concat(dfs.columns).unique())
    if categories is None:
        categories = dict.fromkeys(columns)
        for c in columns:
            categories[c] = np.sort(list(np.unique(np.concatenate([df[c].unique() for df in dfs]))))
    return [encode_nominal(x, columns, categories=categories) for x in dfs]


def dict_to_df(d):
    max_rows = max(len(v) for v in d.values())
    return pd.DataFrame({k: v + [None] * (max_rows - len(v)) for k, v in d.items()})


def dict_to_csv(d, name="csv.csv"):
    max_rows = max(len(v) for v in d.values())
    pd.DataFrame({k: v + [None] * (max_rows - len(v)) for k, v in d.items()}).to_csv(name, sep=";", index=False)


if __name__ == "__main__":
    from automata4cps.examples import examples

    data = examples.high_rack_storage_system_sfowl()
    data = split_data_on_signal_value(data, sig_name="O_w_BRU_Axis_Ctrl", new_value=3)