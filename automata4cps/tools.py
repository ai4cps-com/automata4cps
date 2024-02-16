import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from plotly import colors as clr


def remove_timestamps_without_change(data, sig_names):
    """Removes timestamps where no values changed in comparison to the previous timestamp."""
    new_data = []
    for d in data:
        ind = (d[sig_names].diff() != 0).any(axis=1)
        d = d.loc[ind]
        new_data.append(d.copy())
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
                dd[time_col] = dd[time_col] - dd[time_col].iloc[0]
            sequences[state].append(dd)
    return sequences


def split_data_on_signal_value(data, sig_name, new_value):
    new_data = []
    for d in data:
        d["splitting_group"] = ((d[sig_name] == new_value) & (d[sig_name].shift(1) != new_value)).cumsum()
        for g, dd in d.groupby(d["splitting_group"]):
            new_data.append(dd)
    return new_data


def plot_data(data, title=None, timestamp=None, discrete=False, height=None, plotStates=False, limit_num_points=None,
              customdata=None,
              iterate_colors=True, y_title_font_size=14, opacity=1, vertical_spacing=0.005, sharey=False, bounds=None,
              plot_only_changes=False, yAxisLabelOffset=False, marker_size=4, showlegend=False, mode='lines+markers',
              **kwargs):
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
        height = len(data[0].columns) * 60

    fig = make_subplots(rows=len(data[0].columns), cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
                        shared_yaxes=sharey)

    # select line_shape:
    if plotStates is True:
        lineShape = 'hv'
    else:
        lineShape = 'linear'

    # Add traces
    i = 0
    columns = data[0].columns
    for col_ind in range(len(columns)):
        i += 1
        k = -1
        for d in data:
            col_name = columns[col_ind]
            col = d.columns[col_ind]

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
                                         name=str(col), marker=dict(line_color=color, color=color,
                                                                    line_width=2, size=marker_size),
                                         customdata=customdata,
                                         hovertemplate=hovertemplate, **kwargs), row=i, col=1)
            else:
                ind = min(limit_num_points, d.shape[0])
                fig.add_trace(go.Scatter(x=t[0:ind], y=sig[0:ind], mode=mode,
                                         name=str(col), customdata=customdata,
                                         line=dict(color=color), line_shape=lineShape, **kwargs), row=i, col=1)
            fig.update_yaxes(title_text=str(col_name), row=i, col=1, title_font=dict(size=y_title_font_size),
                             categoryorder='category ascending')
        if i % 2 == 0:
            fig.update_yaxes(side="right", row=i, col=1)
        if yAxisLabelOffset == True:
            fig.update_yaxes(title_standoff=10 * i, row=i, col=1)
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
                showlegend=False
            )
            lower_bound = go.Scatter(
                name='Lower Bound',
                x=bounds[1].index.get_level_values(-1),
                y=lower_vol,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
            fig.add_trace(upper_bound, row=i, col=1)
            fig.add_trace(lower_bound, row=i, col=1)
    fig.update_layout(title={'text': title, 'x': 0.5},
                      autosize=True, height=height + 180, showlegend=showlegend)
    return fig


if __name__ == "__main__":
    from examples import examples

    data = examples.high_rack_storage_system_sfowl()
    data = split_data_on_signal_value(data, sig_name="O_w_BRU_Axis_Ctrl", new_value=3)