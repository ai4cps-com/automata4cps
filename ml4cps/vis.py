"""
    The module provides methods to visualize various kinds of data, such as time series or automata graphs.

    Authors:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
    - Tom Westermann, tom.westermann@hsu-hh.de, tom@ai4cps.com
"""
from __future__ import annotations

from ml4cps.cps import CPS
from plotly import graph_objects as go
import pandas as pd
import datetime
from plotly import colors
import numpy as np
import pydotplus as pdp
from plotly import subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from itertools import chain
import networkx as nx
import dash_cytoscape as cyto
from dash import html, Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import time, webbrowser, threading
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence
import imageio.v2 as imageio
import plotly.io as pio

def plot_timeseries(
    data,
    timestamp=None,
    mode_data=None,
    discrete=False,
    title=None,
    use_columns=None,
    height=None,
    limit_num_points=None,
    names=None,
    xaxis_title=None,
    customdata=None,
    iterate_colors=True,
    y_title_font_size=14,
    opacity=1.0,
    vertical_spacing=0.005,
    sharey=False,
    bounds=None,
    plot_only_changes=False,
    yAxisLabelOffset=False,
    marker_size=4,
    showlegend=False,
    mode_height=0.2,
    x_title=None,
    **kwargs,
):
    """
    Plot one or more time-series datasets as vertically stacked Plotly subplots.

    Each selected column is plotted in its own subplot. If multiple datasets are
    provided, each subplot contains one trace per dataset for that column. An
    optional `mode_data` subplot can be added above the signal subplots.

    Parameters
    ----------
    data : pandas.DataFrame | dict | array-like | list[pandas.DataFrame | dict | array-like]
        One dataset or a list of datasets to plot. Non-DataFrame inputs are
        converted to pandas DataFrames.

    timestamp : str | int | None, default None
        Column name or column index to set as the index for each dataset before
        plotting. Ignored if `None`.

    mode_data : pandas.DataFrame | pandas.Series | numpy.ndarray | list[...] | None, default None
        Optional mode/categorical signal(s) to plot in a dedicated top subplot.
        For DataFrame inputs, a ``Mode`` column is required. If a ``Time`` column
        exists, it is used for the x-axis; otherwise the index is used.

    discrete : bool, default False
        If True, plot signal traces as markers only. If False, use the Plotly
        mode specified by `mode`.

    title : str | None, default None
        Figure title.

    use_columns : sequence[str] | None, default None
        Columns to plot. If None, all columns from the first dataset are used.

    height : int | None, default None
        Figure height in pixels. If None, a default height is computed from the
        number of subplots.

    limit_num_points : int | None, default None
        Maximum number of points to plot from each trace. If None or negative,
        all available points are used.

    names : sequence[str] | None, default None
        Dataset names used in legend entries. If omitted, traces are named by
        their dataset index.

    xaxis_title : str | None, default None
        Title applied to all x-axes.

    customdata : pandas.DataFrame | array-like | None, default None
        Extra per-point hover data. This is attached to each signal trace and
        added to the hover tooltip. Row count should align with plotted points.

    iterate_colors : bool, default True
        If True, cycle through Plotly default colors per dataset. If False, use
        the first default color for every dataset.

    y_title_font_size : int, default 14
        Font size used for subplot y-axis titles.

    opacity : float, default 1.0
        Opacity applied to generated trace colors.

    vertical_spacing : float, default 0.005
        Vertical spacing between subplot rows.

    sharey : bool, default False
        If True, share y-axes across subplot rows.

    bounds : tuple[pandas.DataFrame, pandas.DataFrame] | None, default None
        Pair of `(upper_bounds, lower_bounds)` DataFrames used to draw a filled
        band for each plotted column.

    plot_only_changes : bool, default False
        Only relevant when `discrete=True`. If True, only points where the signal
        changes are plotted, along with the first point.

    yAxisLabelOffset : bool, default False
        If True, progressively increases y-axis title standoff on lower subplots.

    marker_size : int, default 4
        Marker size for discrete and mode traces.

    showlegend : bool, default False
        Whether to display the figure legend.

    mode_height : float, default 0.2
        Relative height of the `mode_data` subplot, when present.

    x_title : str | None, default None
        Title applied only to the bottom x-axis.

    **kwargs
        Additional keyword arguments forwarded to `go.Scatter`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.

    Raises
    ------
    ValueError
        If input data is missing, `mode_height` is invalid, or a mode trace is
        missing its required columns.

    KeyError
        If one of the requested columns is not found in a dataset or bounds data.

    Notes
    -----
    - The returned figure includes the selected columns only.
    - If `mode_data` is provided, it occupies the first subplot row.
    - `customdata` is filtered per trace when `plot_only_changes=True`.
    """
    datasets = _normalize_datasets(data, timestamp=timestamp)
    if not datasets or all(d.empty for d in datasets):
        return go.Figure()

    columns = _resolve_columns(datasets, use_columns)
    customdata_df = _normalize_customdata(customdata)
    limit_num_points = _normalize_point_limit(limit_num_points)
    names = list(names) if names is not None else None

    if height is None:
        height = max(800, len(columns) * 60) + 180

    has_mode_row = mode_data is not None
    fig = _make_timeseries_subplots(
        num_signal_rows=len(columns),
        has_mode_row=has_mode_row,
        mode_height=mode_height,
        vertical_spacing=vertical_spacing,
        sharey=sharey,
    )

    row_offset = 0
    mode_categories = []

    if has_mode_row:
        mode_categories = _add_mode_traces(
            fig=fig,
            mode_data=mode_data,
            names=names,
            iterate_colors=iterate_colors,
            opacity=opacity,
            marker_size=marker_size,
            showlegend=showlegend,
            scatter_kwargs=kwargs,
        )
        row_offset = 1

    for col_idx, column in enumerate(columns, start=1):
        row = col_idx + row_offset
        for trace_idx, df in enumerate(datasets):
            trace_name = _trace_name(trace_idx, names)
            color = _trace_color(trace_idx, iterate_colors, opacity)

            x, y = _extract_xy(df, column, limit_num_points)
            trace_customdata = _slice_customdata(customdata_df, len(x))

            if discrete and plot_only_changes:
                keep_idx = _changed_point_indices(y)
                x = x[keep_idx]
                y = y[keep_idx]
                if trace_customdata is not None:
                    trace_customdata = trace_customdata.iloc[keep_idx]

            hovertemplate = _build_hovertemplate(trace_customdata)

            _add_signal_trace(
                fig=fig,
                row=row,
                x=x,
                y=y,
                trace_name=trace_name,
                color=color,
                discrete=discrete,
                marker_size=marker_size,
                customdata=trace_customdata,
                hovertemplate=hovertemplate,
                showlegend=(showlegend and col_idx == 1 and mode_data is None),
                scatter_kwargs=kwargs.copy(),
            )

        fig.update_yaxes(
            title_text=str(column),
            row=row,
            col=1,
            title_font=dict(size=y_title_font_size),
            categoryorder="category ascending",
        )

        if row % 2 == 0:
            fig.update_yaxes(side="right", row=row, col=1)

        if yAxisLabelOffset:
            fig.update_yaxes(title_standoff=10 * row, row=row, col=1)

        if bounds is not None:
            _add_bounds(fig, row, column, bounds)

    _apply_layout(
        fig=fig,
        title=title,
        height=height,
        xaxis_title=xaxis_title,
        showlegend=showlegend,
    )

    if has_mode_row and mode_categories:
        categories = pd.concat(mode_categories).drop_duplicates().to_list()
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=categories,
            row=1,
            col=1,
        )

    if x_title:
        _set_bottom_xaxis_title(fig, x_title)

    return fig


def _normalize_datasets(data, timestamp=None):
    """Convert input data into a list of DataFrames and optionally set the index."""
    if data is None:
        raise ValueError("`data` must not be None.")

    if not isinstance(data, list):
        data = [data]

    datasets = []
    for item in data:
        df = item if isinstance(item, pd.DataFrame) else pd.DataFrame(item)
        if timestamp is not None and isinstance(timestamp, (str, int)):
            df = df.set_index(timestamp)
        datasets.append(df)

    return datasets


def _resolve_columns(datasets, use_columns=None):
    """Resolve columns to plot and validate they exist in every dataset."""
    columns = list(datasets[0].columns) if use_columns is None else list(use_columns)

    for column in columns:
        for idx, df in enumerate(datasets):
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in dataset at index {idx}.")
    return columns


def _normalize_customdata(customdata):
    """Convert custom hover data to a DataFrame and fill missing values."""
    if customdata is None:
        return None
    if not isinstance(customdata, pd.DataFrame):
        customdata = pd.DataFrame(customdata)
    return customdata.fillna("")


def _normalize_point_limit(limit_num_points):
    """Convert point limit to a usable numeric cap."""
    if limit_num_points is None or limit_num_points < 0:
        return np.inf
    return limit_num_points


def _make_timeseries_subplots(num_signal_rows, has_mode_row, mode_height, vertical_spacing, sharey):
    """Create the subplot layout for the figure."""
    if has_mode_row:
        if not (0 < mode_height < 1):
            raise ValueError("`mode_height` must be between 0 and 1.")
        total_rows = num_signal_rows + 1
        row_heights = [mode_height] + [(1 - mode_height) / num_signal_rows] * num_signal_rows
        return make_subplots(
            rows=total_rows,
            cols=1,
            row_heights=row_heights,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            shared_yaxes=sharey,
        )

    return make_subplots(
        rows=num_signal_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        shared_yaxes=sharey,
    )


def _trace_name(index, names=None):
    """Return a legend/trace name for a dataset index."""
    if names is not None and index < len(names):
        return names[index]
    return str(index)


def _trace_color(index, iterate_colors=True, opacity=1.0):
    """Return an RGBA Plotly default color for a trace."""
    base = (
        DEFAULT_PLOTLY_COLORS[index % len(DEFAULT_PLOTLY_COLORS)]
        if iterate_colors
        else DEFAULT_PLOTLY_COLORS[0]
    )
    return f"rgba{base[3:-1]}, {opacity})"


def _index_to_numpy(df):
    """Return x-values from a DataFrame index, supporting MultiIndex."""
    if len(df.index.names) > 1:
        return df.index.get_level_values(df.index.names[-1]).to_numpy()
    return df.index.to_numpy()


def _series_to_numpy(series):
    """Return y-values as a NumPy array, stringifying tuple-valued entries."""
    if series.dtype == tuple:
        return series.astype(str).to_numpy()
    return series.to_numpy()


def _extract_xy(df, column, limit_num_points):
    """Extract x/y arrays for a single dataset column."""
    n = int(min(limit_num_points, len(df)))
    x = _index_to_numpy(df)[:n]
    y = _series_to_numpy(df[column])[:n]
    return x, y


def _slice_customdata(customdata_df, length):
    """Slice hover customdata to the same length as the plotted trace."""
    if customdata_df is None:
        return None
    return customdata_df.iloc[:length].copy()


def _changed_point_indices(values):
    """Return indices for the first point and subsequent changes in value."""
    if len(values) == 0:
        return np.array([], dtype=int)
    changed_idx = np.nonzero(np.not_equal(values[:-1], values[1:]))[0] + 1
    return np.insert(changed_idx, 0, 0)


def _build_hovertemplate(customdata_df):
    """Build a hovertemplate including optional customdata columns."""
    hovertemplate = "<b>Time:</b> %{x}<br><b>Event:</b> %{y}"
    if customdata_df is not None:
        hovertemplate += "<br><b>Context:</b>"
        for idx, col in enumerate(customdata_df.columns):
            hovertemplate += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<em>{col}:</em> %{{customdata[{idx}]}}"
    return hovertemplate


def _add_signal_trace(
    fig,
    row,
    x,
    y,
    trace_name,
    color,
    discrete,
    marker_size,
    customdata,
    hovertemplate,
    showlegend,
    scatter_kwargs,
):
    """Add a signal trace to the figure."""
    if discrete:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=scatter_kwargs.pop("mode", "markers"),
                name=trace_name,
                legendgroup=trace_name,
                marker=dict(
                    line_color=color,
                    color=color,
                    line_width=2,
                    size=marker_size,
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
                showlegend=showlegend,
                **scatter_kwargs,
            ),
            row=row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=scatter_kwargs.pop("mode", "lines+markers"),
                name=trace_name,
                legendgroup=trace_name,
                line=dict(color=color, shape="linear"),
                customdata=customdata,
                hovertemplate=hovertemplate if customdata is not None else None,
                showlegend=showlegend,
                **scatter_kwargs,
            ),
            row=row,
            col=1,
        )


def _normalize_mode_item(mode_item):
    """Convert one mode input into a DataFrame with a required 'Mode' column."""
    if isinstance(mode_item, np.ndarray):
        return pd.DataFrame({"Time": np.arange(mode_item.shape[0]), "Mode": mode_item})

    if isinstance(mode_item, pd.Series):
        return pd.DataFrame({"Mode": mode_item})

    if not isinstance(mode_item, pd.DataFrame):
        mode_item = pd.DataFrame(mode_item)

    if "Mode" not in mode_item.columns:
        raise ValueError("Each `mode_data` item must contain a 'Mode' column.")

    return mode_item


def _add_mode_traces(
    fig,
    mode_data,
    names,
    iterate_colors,
    opacity,
    marker_size,
    showlegend,
    scatter_kwargs,
):
    """Add top-row mode traces and return their category series."""
    mode_items = mode_data if isinstance(mode_data, list) else [mode_data]
    categories = []

    for idx, item in enumerate(mode_items):
        md = _normalize_mode_item(item)
        trace_name = _trace_name(idx, names)
        color = _trace_color(idx, iterate_colors, opacity)
        x = md["Time"] if "Time" in md.columns else md.index

        categories.append(md["Mode"].drop_duplicates())

        fig.add_trace(
            go.Scatter(
                x=x,
                y=md["Mode"],
                mode="lines+markers",
                name=trace_name,
                legendgroup=trace_name,
                line_shape="hv",
                marker=dict(
                    line_color=color,
                    color=color,
                    line_width=2,
                    size=marker_size,
                ),
                showlegend=showlegend,
                **scatter_kwargs,
            ),
            row=1,
            col=1,
        )

    return categories


def _add_bounds(fig, row, column, bounds):
    """Add upper/lower filled bound traces for one subplot row."""
    upper_df, lower_df = bounds

    if column not in upper_df.columns or column not in lower_df.columns:
        raise KeyError(f"Bounds missing column '{column}'.")

    upper_trace = go.Scatter(
        name="Upper Bound",
        x=upper_df.index.get_level_values(-1),
        y=upper_df[column],
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False,
    )

    lower_trace = go.Scatter(
        name="Lower Bound",
        x=lower_df.index.get_level_values(-1),
        y=lower_df[column],
        mode="lines",
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
        showlegend=False,
    )

    fig.add_trace(upper_trace, row=row, col=1)
    fig.add_trace(lower_trace, row=row, col=1)


def _apply_layout(fig, title, height, xaxis_title, showlegend):
    """Apply common layout settings to the figure."""
    if title is not None:
        fig.update_layout(title={"text": title, "x": 0.5})

    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)

    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(b=20, t=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.02,
        ),
        showlegend=showlegend,
    )


def _set_bottom_xaxis_title(fig, x_title):
    """Set the title of the bottom-most x-axis."""
    for axis_num in reversed(range(1, 100)):
        axis_key = f"xaxis{axis_num if axis_num > 1 else ''}"
        if axis_key in fig.layout:
            fig.layout[axis_key].title = x_title
            return


def plot_stateflow(stateflow, color_mapping=None, state_col='State', task_col='Task', bar_height=12,
                   start_column='Start', finish_column='Finish', return_figure=False, description_col='Description',
                   idle_states=None):
    """
    Visualizes state transitions over time for one or more tasks/stations as a Gantt-like interactive timeline.

    Parameters:
    - stateflow (DataFrame or dict): DataFrame with state transitions, or a dictionary of DataFrames per station.
    - color_mapping (dict, optional): Mapping of state names to colors. If None, default colors are used.
    - state_col (str): Column name indicating the state (default: 'State').
    - bar_height (int): Height of the timeline bars (default: 12).
    - start_column (str): Column name with start timestamps (default: 'Start').
    - finish_column (str): Column name with end timestamps (default: 'Finish').
    - return_figure (bool): If True, returns a Plotly Figure. Otherwise, returns a list of Plotly traces.
    - description_col (str or list): Column(s) to include in the hover tooltip (default: 'Description').
    - idle_states (str or list): State(s) to exclude from the plot (e.g., 'IDLE').

    Returns:
    - Plotly Figure or list of traces, depending on `return_figure`.

    Example:
        fig = plot_stateflow(df, state_col='Mode', start_column='StartTime', finish_column='EndTime', return_figure=True)
        fig.show()

    This function is ideal for visualizing process flows, machine states, or event-based logs with time intervals.
    """

    if idle_states is None:
        idle_states = []
    if type(idle_states) is str:
        idle_states = [idle_states]

    if isinstance(stateflow, dict):
        stateflow_df_list = []
        for station, s in stateflow.items():
            if s.size:
                sf = s[(~s[state_col].isin(idle_states))]
                # ((start_plot <= s.Time) &
                #  (s.Time <= finish_plot)) |
                # ((start_plot <= s.Finish) & (s.Finish <= finish_plot)))
                sf[task_col] = station
                # if sf.size > 0 and pd.isnull(sf['Finish'].iloc[-1]):
                #     sf['Finish'].iloc[-1] = pd.to_datetime(finish_plot)
                # s['Finish'] = pd.to_datetime(s['Finish'])
                stateflow_df_list.append(sf)
            else:
                stateflow_df_list.append(pd.DataFrame([]))
        stateflow_df = pd.concat(stateflow_df_list)
    else:
        stateflow_df = stateflow

    if stateflow_df.shape[0] == 0:
        if return_figure:
            return go.Figure()
        else:
            return []

    if description_col is not None and type(description_col) is str:
        description_col = [description_col]
    if color_mapping is None:
        color_mapping = {}
        items = list(stateflow_df[state_col].unique())
        for k, i in enumerate(items):
            color_mapping[i] = colors.qualitative.Dark24[k % 24]

    stateflow_df['Duration'] = stateflow_df[finish_column] - stateflow_df[start_column]
    if state_col not in stateflow_df:
        stateflow_df[state_col] = None
    stateflow_df[state_col] = stateflow_df[state_col].replace([None], [''])

    traces = []
    for name, g in stateflow_df.groupby(state_col):
        if name is None or name == '':
            continue
        x = []
        y = []
        hovertext = []
        custom_data = []
        text = []
        for k, row in g.iterrows():  # , g[item_col], g.Source, g.Destination):
            x1, x2, tsk = row[start_column], row[finish_column], row[task_col]
            x.append(x1)
            x.append(x2)
            x.append(None)
            y.append(tsk)
            y.append(tsk)
            y.append(None)
            dauer = x2 - x1
            if type(x1) in [datetime.datetime, pd.Timestamp]:
                x1_str = x1.strftime("%d.%m %H:%M:%S")
            else:
                x1_str = x1
            if type(x2) in [datetime.datetime, pd.Timestamp]:
                x2_str = x2.strftime("%d.%m %H:%M:%S")
            else:
                x2_str = x2

            ht = 'Start: {}<br>Finish: {}<br>Duration: {}'.format(x1_str, x2_str, dauer)
            if description_col is not None:
                for dc in description_col:
                    if dc in row:
                        ht += '<br>{}: {}'.format(dc, row[dc])
            for k, val in row.items():
                if not pd.isnull(val) and k not in [state_col, finish_column, start_column, task_col, 'Duration']:
                    ht += f'<br>{k}: {val}'
            hovertext.append(ht)

            custom_data.append(dict(Start=x1, Finish=x2, State=name, Task=tsk, Source=row.get('Quelle', None)))

        color = color_mapping.get(name, "black")
        traces.append(go.Scatter(x=x, y=y, line=dict(width=bar_height), name=name, line_color=color,
                                 hoverinfo='skip', mode='lines', legendgroup=name, showlegend=True, opacity=0.8))
        traces.append(go.Scatter(x=np.asarray(g[start_column] + g.Duration / 2), y=g.Task, mode='text+markers',
                                 marker=dict(size=5, color=color), name=name,
                                 showlegend=False, opacity=0.8, customdata=custom_data,
                                 hovertext=hovertext, text=text, textfont=dict(size=10, color='olive'),
                                 hovertemplate=f'<extra></extra><b>{name}</b><br>%{{hovertext}}'))

    if return_figure:
        fig = go.Figure(data=traces)
        return fig
    else:
        return traces


def plot_cps_component(cps, id=None, node_labels=False, center_node_labels=False, event_label=True,
                       show_transition_freq=False, show_transition_timing=False, font_size=6, edge_font_size=6,
                       edge_text_max_width=None, init_label=False, limit_interval_precision=2,
                       show_transition_data=False, transition_data_keys=True, node_size=20, output="cyto",
                       dash_port=8050, min_zoom=0.5, split_edges_diff_event=False,
                       max_zoom=2, min_edge_thickness=0.1, max_edge_thickness=4, freq_as_edge_thickness=False,
                       color="black", title_text=None, layout_name='breadthfirst', layout_spacingFactor=1,
                       hide_nodes=None):
    """    
    Visualizes a component of a Cyber-Physical System (CPS) as a graph using Dash Cytoscape.
    This function generates a graphical representation of the discrete states and transitions of a CPS,
    with various customization options for node and edge appearance, labels, and output format.
    The visualization can be rendered as Dash Cytoscape elements, in a Dash app, or as a notebook widget.
    Parameters:
        cps: object
            The CPS object containing discrete states, transitions, and related data.
        id: str, optional
            The unique identifier for the Cytoscape component (default: "graph").
        node_labels: bool, optional
            Whether to display labels on nodes (default: False).
        center_node_labels: bool, optional
            Whether to center node labels (default: False).
        edge_labels: bool, optional
            Whether to display labels on edges (default: True).
        show_transition_freq: bool, optional
            Whether to show transition frequency on edge labels (default: False).
        edge_font_size: int, optional
            Font size for edge labels (default: 6).
        edge_text_max_width: int or None, optional
            Maximum width for edge label text wrapping (default: None).
        init_label: bool, optional
            Whether to label initial state transitions as 'init' (default: False).
        show_transition_data: bool or list, optional
            Whether to display additional transition data on edge labels. If a list, only specified keys are shown (default: False).
        node_size: int, optional
            Size of the nodes (default: 20).
        output: str, optional
            Output format: "cyto" (Dash Cytoscape Div), "elements" (raw elements), "notebook" (inline Dash app), or "dash" (Dash app in browser) (default: "cyto").
        dash_port: int, optional
            Port for running the Dash app (default: 8050).
        min_zoom: float, optional
            Minimum zoom level for the Cytoscape component (default: 0.5).
        max_zoom: float, optional
            Maximum zoom level for the Cytoscape component (default: 1).
        min_edge_thickness: float, optional
            Minimum edge thickness for frequency-based scaling (default: 0.1).
        max_edge_thickness: float, optional
            Maximum edge thickness for frequency-based scaling (default: 4).
        freq_as_edge_thickness: bool, optional
            Whether to scale edge thickness based on transition frequency (default: False).
        color: str, optional
            Color for nodes and edges (default: "black"). If "hsu", uses a preset color.
        title_text: str or Dash component, optional
            Title text or component to display above the graph (default: None).
    Returns:
        Dash component, dict, or Dash app:
            - If output == "cyto": returns a Dash html.Div containing the Cytoscape graph.
            - If output == "elements": returns a dict with 'nodes' and 'edges'.
            - If output == "notebook": runs and displays a Dash app inline (for Jupyter).
            - If output == "dash": runs a Dash app in the browser and returns the app instance.
    Notes:
        - Requires Dash, dash_cytoscape, dash_bootstrap_components, and pandas.
        - The function supports interactive modals for displaying timing data on states and transitions.
        - Threading is used to launch the Dash app in browser mode without blocking the main program.
    # 1. 'grid'            → Places nodes in a simple rectangular grid.
# 2. 'random'          → Randomly positions nodes; useful for testing.
# 3. 'circle'          → Arranges nodes evenly around a circle.
# 4. 'concentric'      → Places nodes in concentric circles, often by degree or weight.
# 5. 'breadthfirst'    → Hierarchical layout (tree-like), good for state machines or DAGs.
#                        Optional params: directed=True, padding=<int>
# 6. 'cose'            → Force-directed layout (spring simulation). Great for organic graphs.
#                        Optional params: idealEdgeLength, nodeRepulsion, gravity, numIter
# 7. 'cose-bilkent'    → Improved force-directed layout with better stability and aesthetics.
#                        (Requires: cyto.load_extra_layouts())
# 8. 'cola'            → Constraint-based force-directed layout; handles larger graphs well.
#                        (Requires: cyto.load_extra_layouts())
# 9. 'euler'           → Physically simulated layout; looks natural and dynamic.
#                        (Requires: cyto.load_extra_layouts())
# 10. 'avsdf'          → Circular layout optimized to reduce edge crossings.
#                        (Requires: cyto.load_extra_layouts())
# 11. 'spread'         → Distributes disconnected components evenly across space.
#                        (Requires: cyto.load_extra_layouts())
# 12. 'klay'           → Layered (hierarchical) layout, excellent for flowcharts or process models.
#                        (Requires: cyto.load_extra_layouts())
# 13. 'dagre'          → Directed acyclic graph layout, ideal for workflows and automata.
#                        (Requires: cyto.load_extra_layouts())
#                        Optional params: rankDir='TB' (top-bottom), 'LR' (left-right), etc.
    """


    if id is None:
        id = "graph"

    if color == "hsu":
        color = "#B8234F"
    nodes = []
    edges = []
    for n in cps.discrete_states:
        if n in cps.final_q:
            classes = ['final']
            if hide_nodes and n in hide_nodes:
                classes.append('hidden')
            nodes.append(dict(data={'id': n, 'label': n.replace(' ','')}, classes=classes))
        else:
            classes = []
            if hide_nodes and n in hide_nodes:
                classes.append('hidden')
            nodes.append(dict(data={'id': n, 'label': n.replace(' ','')}, classes=classes))

        if n in cps.q0:
            nodes.append(dict(data={'id': f"q0{n}", 'label': f"q0{n}"}, classes='q0'))
            if init_label:
                edges.append(dict(data=dict(label='init', source=f"q0{n}", target=n, line_color=color, config='')))
            elif event_label:
                edges.append(dict(data=dict(label=cps.initial_r, source=f"q0{n}", target=n, line_color=color, config='')))
            else:
                edges.append(dict(data=dict(source=f"q0{n}", target=n, line_color=color, config='')))

    fallback_keys = ("*", "default", None)

    def _resolve_for_config(value, cfg, source=None, target=None, event_key=None, edge_data=None, default=None):
        if callable(value):
            try:
                return value(cfg, source, target, event_key, edge_data)
            except TypeError:
                try:
                    return value(cfg)
                except TypeError:
                    return value()

        if isinstance(value, dict):
            if cfg in value:
                return value[cfg]
            for key in fallback_keys:
                if key in value:
                    return value[key]
            return default

        if value is None:
            return default
        return value

    def _as_seconds(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, (int, float, np.number)):
            return float(raw_value)
        try:
            return pd.Timedelta(raw_value).total_seconds()
        except Exception:
            try:
                return float(raw_value)
            except Exception:
                return None

    def _resolve_timing_values(data, cfg, source, target, event_key):
        raw_timing = _resolve_for_config(data.get("timing"), cfg, source, target, event_key, data, default=[])
        if raw_timing is None:
            raw_timing = []
        if not isinstance(raw_timing, (list, tuple, np.ndarray)):
            raw_timing = [raw_timing]
        timings = [_as_seconds(x) for x in raw_timing]
        return [x for x in timings if x is not None]

    def _resolve_interval_bounds(data, cfg, source, target, event_key, timings):
        if timings:
            return min(timings), max(timings)

        interval = _resolve_for_config(data.get("interval"), cfg, source, target, event_key, data, default=None)
        if isinstance(interval, (list, tuple, np.ndarray)) and len(interval) >= 2:
            interval_min = _as_seconds(interval[0])
            interval_max = _as_seconds(interval[1])
            if interval_min is not None and interval_max is not None:
                return (interval_min, interval_max) if interval_min <= interval_max else (interval_max, interval_min)

        min_interval = None
        max_interval = None
        for key in ("min_interval", "minTiming", "min_time"):
            if key in data:
                min_interval = _as_seconds(_resolve_for_config(data.get(key), cfg, source, target, event_key, data, default=None))
                break
        for key in ("max_interval", "maxTiming", "max_time"):
            if key in data:
                max_interval = _as_seconds(_resolve_for_config(data.get(key), cfg, source, target, event_key, data, default=None))
                break

        if min_interval is not None and max_interval is not None:
            return (min_interval, max_interval) if min_interval <= max_interval else (max_interval, min_interval)
        if min_interval is not None:
            return min_interval, min_interval
        if max_interval is not None:
            return max_interval, max_interval

        time_value = _as_seconds(_resolve_for_config(data.get("time"), cfg, source, target, event_key, data, default=None))
        if time_value is not None:
            return time_value, time_value
        return None

    def _is_state_enabled(state, cfg):
        state_data = cps._G.nodes[state]
        enabled_data = state_data.get("enabled", None)
        if enabled_data is None:
            return True
        return bool(_resolve_for_config(enabled_data, cfg, source=state, target=state, default=0))

    def _is_transition_enabled(source, target, event_key, data, cfg):
        if not (_is_state_enabled(source, cfg) and _is_state_enabled(target, cfg)):
            return False
        enabled_data = data.get("enabled", None)
        if enabled_data is not None and not bool(_resolve_for_config(enabled_data, cfg, source, target, event_key, data, default=0)):
            return False
        guard = data.get("guard", None)
        if guard is not None and not bool(_resolve_for_config(guard, cfg, source, target, event_key, data, default=True)):
            return False
        return True

    inferred_configs = set()
    if hasattr(cps, "configurations") and getattr(cps, "configurations") is not None:
        for cfg in getattr(cps, "configurations"):
            if cfg not in fallback_keys:
                inferred_configs.add(cfg)

    for _, node_data in cps._G.nodes(data=True):
        for value in node_data.values():
            if isinstance(value, dict):
                for cfg in value.keys():
                    if cfg not in fallback_keys:
                        inferred_configs.add(cfg)

    all_transitions = list(cps.get_transitions())
    for _, _, _, data in all_transitions:
        for value in data.values():
            if isinstance(value, dict):
                for cfg in value.keys():
                    if cfg not in fallback_keys:
                        inferred_configs.add(cfg)

    inferred_configs = sorted(inferred_configs, key=lambda x: str(x))
    active_configuration = getattr(cps, "configuration", None)
    plot_all_configurations = active_configuration is None and len(inferred_configs) > 0
    if len(inferred_configs) <= len(DEFAULT_PLOTLY_COLORS):
        config_palette = DEFAULT_PLOTLY_COLORS
    else:
        # Expand the palette rather than recycling colors for larger systems.
        config_palette = [f"hsl({360 * i / len(inferred_configs):.3f},65%,45%)"
                          for i in range(len(inferred_configs))]
    config_color_map = dict(zip(inferred_configs, config_palette))

    for e in all_transitions:
        source, target, event_key, transition_data = e
        if hide_nodes and (source in hide_nodes or target in hide_nodes):
            continue

        configs_to_plot = inferred_configs if plot_all_configurations else [active_configuration]

        for cfg in configs_to_plot:
            if not _is_transition_enabled(source, target, event_key, transition_data, cfg):
                continue

            timings = _resolve_timing_values(transition_data, cfg, source, target, event_key)
            interval_bounds = _resolve_interval_bounds(transition_data, cfg, source, target, event_key, timings)

            if timings:
                freq = float(len(timings))
            else:
                freq = 0
            if freq == 0:
                continue

            edge_event = transition_data.get("event", event_key)
            edge_color = config_color_map.get(cfg, color) if plot_all_configurations else color
            config_tag = f"[{cfg}] " if plot_all_configurations else ""
            base_label = f"{config_tag}{edge_event}" if event_label else config_tag.strip()

            edge = dict(data={
                'source': source,
                'target': target,
                'label': base_label,
                'timing': timings,
                'freq': freq,
                'line_color': edge_color,
                'config': str(cfg) if cfg is not None else ''
            })

            existing_edge = None
            if not plot_all_configurations and not split_edges_diff_event:
                existing_edge = next((x for x in edges if x['data']['source'] == edge['data']['source'] and
                                      x['data']['target'] == edge['data']['target']), None)

            if existing_edge is None or split_edges_diff_event or plot_all_configurations:
                if show_transition_freq:
                    edge['data']['label'] += f' #{freq:g}'

                if show_transition_timing and interval_bounds is not None:
                    tmin, tmax = interval_bounds
                    if limit_interval_precision is None:
                        edge['data']['label'] += f' [{tmin},{tmax}]'
                    else:
                        edge['data']['label'] += (
                            f' [{tmin:.{limit_interval_precision}f},{tmax:.{limit_interval_precision}f}]'
                        )

                if show_transition_data:
                    edge_data = {}
                    for key, value in transition_data.items():
                        if key in {"event", "guard"}:
                            continue
                        if isinstance(show_transition_data, list) and key not in show_transition_data:
                            continue
                        edge_data[key] = _resolve_for_config(
                            value, cfg, source=source, target=target, event_key=event_key, edge_data=transition_data, default=value
                        )

                    if edge_data:
                        if transition_data_keys:
                            edge['data']['label'] += " " + " ".join(f"{key}: {value}" for key, value in edge_data.items())
                        else:
                            edge['data']['label'] += " " + " ".join(f"{value}" for _, value in edge_data.items())
                edges.append(edge)
            else:  # existing_edge
                if show_transition_freq:
                    existing_edge['data']['label'] += f' #{freq:g}'
                if show_transition_timing and interval_bounds is not None:
                    tmin, tmax = interval_bounds
                    if limit_interval_precision is None:
                        existing_edge['data']['label'] += f' [{tmin},{tmax}]'
                    else:
                        existing_edge['data']['label'] += (
                            f' [{tmin:.{limit_interval_precision}f},{tmax:.{limit_interval_precision}f}]'
                        )
                if show_transition_data or event_label:
                    edge_data = {}
                    for key, value in transition_data.items():
                        if key in {"event", "guard"}:
                            continue
                        if isinstance(show_transition_data, list) and key not in show_transition_data:
                            continue
                        edge_data[key] = _resolve_for_config(
                            value, cfg, source=source, target=target, event_key=event_key, edge_data=transition_data, default=value
                        )
                    if edge_data:
                        existing_edge['data']['label'] += (
                            f" ,{edge_event} " + "; ".join(f"{key} = {value}" for key, value in edge_data.items())
                        )

    # Normalize thickness to the range [1, 10]
    thickness_values = [edge["data"].get("freq", 1) for edge in edges]
    min_thickness = min(thickness_values) if thickness_values else 0
    max_thickness = max(thickness_values) if thickness_values else 0

    if max_thickness == min_thickness:
        max_thickness += 1

    for edge in edges:
        raw_thickness = edge["data"].get("freq", 1)
        edge["data"]["thickness"] = ((raw_thickness - min_thickness) / (max_thickness - min_thickness) *
                                     (max_edge_thickness - min_edge_thickness) + min_edge_thickness)

    elements = dict(nodes=nodes, edges=edges)

    if output == "elements":
        return elements

    node_style = {'width': node_size,
                  'height': node_size,
                  'border-width': 1,
                  'border-color': color,
                  'background-color': 'transparent',
                  "font-family": "serif",
                  'background-opacity': 0}
    if node_labels:
        node_style['label'] = 'data(label)'
        node_style['font-size'] = font_size
        node_style['font-style'] = "italic"
        node_style['text-wrap'] = 'wrap'
        node_style['text-max-width'] = 50
    if center_node_labels:
        node_style['text-halign'] = 'center'
        node_style['text-valign'] = 'center'


    edge_style = {
        'curve-style': 'bezier',
        'background-color': 'white',  # Inner fill
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'data(line_color)',
        'target-arrow-size': 3,
        'text-background-color': '#ffffff',
        'text-background-opacity': 1,
        'text-background-shape': 'roundrectangle',
        'color': 'data(line_color)',
        'width': 1,
        'font-style': 'italic',
        'font-family': "serif",
        'text-wrap': 'wrap',
        'font-size': edge_font_size,
        'text-max-width': edge_text_max_width,
        'line-color': 'data(line_color)'
    }

    if freq_as_edge_thickness:
        edge_style['width'] = 'data(thickness)'

    edge_style['label'] = 'data(label)'

    stylesheet = [
        {
            'selector': 'node',
            'style': node_style
        },
        {
            'selector': '.q0',
            'style': {
                'width': 1,  # Small width to make it look like a point
                'height': 1,  # Small height to make it look like a point
                'label': '',  # No label to keep it minimal
                'border-width': 0  # No border
            }
        },
        {
            'selector': '.final',
            'style': {
                'border-width': 3  # No border
            }
        },
        {
            'selector': 'edge',
            'style': edge_style
        },
        {
            "selector": ".hidden",
            "style": {"visibility": "hidden"}
        }
    ]

    network = cyto.Cytoscape(
        id=id,
        layout={'name': layout_name, "fit": True, "spacingFactor": layout_spacingFactor},
        maxZoom=max_zoom,
        minZoom=min_zoom,
        style={'width': '100%', 'height': '100%'}, stylesheet=stylesheet,
        elements=elements)

    modal_state_data = dbc.Modal(children=[dbc.ModalHeader("Timings"),
                                           dbc.ModalBody(html.Div(children=[]))],
                                 id=f"{id}-modal-state-data")
    modal_transition_data = dbc.Modal(children=[dbc.ModalHeader("Timings"),
                                                dbc.ModalBody(html.Div(children=[]))],
                                      id=f"{id}-modal-transition-data")
    network = html.Div([title_text, network, modal_state_data, modal_transition_data], style={'width': '100%', 'height': '100%'})

    if output == "notebook":
        app = Dash(__name__)
        app.layout = html.Div(children=[network], style={'width': '100%',
                                                         'height': '100vh',
                                                         'margin': '0',
                                                         'padding': '0'})
        app.run(mode='inline', port=dash_port)
        return None
    elif output == "dash":
        app = Dash(__name__)
        app.layout = html.Div(children=[network], style={'width': '100%',
                                                         'height': '100vh',
                                                         'margin': '0',
                                                         'padding': '0'})

        # Function to start the Dash server
        def run_dash():
            app.run(port=dash_port, debug=False, use_reloader=False)  # Start the Dash server

        # Function to open the browser
        def open_browser():
            time.sleep(1)  # Give the server a second to start
            webbrowser.open(f"http://127.0.0.1:{dash_port}/")  # Open the Dash app in the browser

        # Start the Dash server in a separate thread
        server_thread = threading.Thread(target=run_dash)
        server_thread.daemon = True  # Allows the program to exit even if this thread is running
        server_thread.start()

        # Open the Dash app in the default browser
        open_browser()
        server_thread.join(timeout=10)
        return app
    else:
        return network


def plot_cps(cps: CPS, dash_id=None, node_labels=False, edge_labels=True, node_size=40, node_font_size=20,
             edge_font_size=16, edge_text_max_width=None, output="cyto", dash_port=8050, height='100%',
             minZoom=0.5, maxZoom=2, **kwargs):
    """
    Plots all the components of a CPS in the same figure.
    :param cps: CPS to plot.
    :param node_labels: Should node labels be plotted.
    :param edge_labels: Should edge labels be plotted.
    :param node_size: What is the size of the nodes in the figure.
    :param node_font_size: The font size of the node labels.
    :param edge_font_size: The font size of the edge labels.
    :param edge_text_max_width: Max width of the edge labels.
    :param output: Should output be plotted as a dash.Cytoscape component ("cyto"), or should dash server be run
    ("dash").
    :param dash_port: If temporary dash server is run, what port to use.
    :param kwargs: Other paramters are forwarded to the Cytoscape component.
    :return:
    """
    elements = dict(nodes=[], edges=[])

    for comid, com in cps.items():
        els = dict(edges=[], nodes=com)
        elements['nodes'].append({'data': {'id': comid, 'label': comid}, 'classes': 'parent'})
        for x in els['nodes']:
            if type(x) is str or type(x) is list:
                x = {'data': {'id': x}}
            x['data']['group'] = comid
            x['data']['parent'] = comid
            x['data']['label'] = f"{x['data']['id']}"
            x['data']['id'] = f"{comid}-{x['data']['id']}"
            elements['nodes'].append(x)
        for x in els['edges']:
            x['data']['source'] = f"{comid}-{x['data']['source']}"
            x['data']['target'] = f"{comid}-{x['data']['target']}"
            elements['edges'].append(x)

    node_style = {'width': node_size,
                  'height': node_size}
    if node_labels:
        node_style['label'] = 'data(label)'
        node_style['font-size'] = node_font_size
        node_style['text-wrap'] = 'wrap'
        node_style['text-max-width'] = 50

    edge_style = {
        # The default curve style does not work with certain arrows
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-size': 3,
        'width': 1,
        'font-color': 'black',
        'text-wrap': 'wrap',
        'font-size': edge_font_size,
        'text-max-width': edge_text_max_width
    }
    if edge_labels:
        edge_style['label'] = 'data(label)'

    stylesheet = [
        {
            'selector': 'node',
            'style': node_style
        },
        {
            'selector': 'edge',
            'style': edge_style
        }]

    network = cyto.Cytoscape(
        id=dash_id if dash_id is not None else cps.id,
        layout={
            'name': 'grid',
            'padding': 10,  # Padding around the graph layout
            'nodeOverlap': 20,  # Adjust to reduce overlap
            'nodeRepulsion': 50,  # Increase repulsion for better separation
            'idealEdgeLength': 10,  # Increase edge length to spread nodes
            'componentSpacing': 50,  # Spacing between disconnected components
            'nodeDimensionsIncludeLabels': True,  # Include label sizes in layout
            'nestingFactor': 0.7,  # Factor to apply to compounds when calculating layout
            "spacingFactor": 1.5
        },
        # layout={
        #     "name": "breadthfirst",
        #     "directed": True,
        #     "spacingFactor": 1.5
        # },
        maxZoom=maxZoom,
        minZoom=minZoom,
        stylesheet=stylesheet,
        elements=elements, style={'width': '100%', 'height': height},
        **kwargs)

    modal_state_data = dbc.Modal(children=[dbc.ModalHeader("Timings"),
                                           dbc.ModalBody(html.Div(children=[]))],
                                 id=f"{id}-modal-state-data")
    modal_transition_data = dbc.Modal(children=[dbc.ModalHeader("Timings"),
                                                dbc.ModalBody(html.Div(children=[]))],
                                      id=f"{id}-modal-transition-data")
    # network = html.Div([network, modal_state_data, modal_transition_data])
    if output == "notebook":
        app = Dash(__name__)
        app.layout = html.Div(children=[network])
        app.run(mode='inline')
        return
    if output == "dash":
        app = Dash(__name__)
        app.layout = html.Div(children=[network], style={'width': '100%',
                                                         'height': '100%',
                                                         'margin': '0',
                                                         'padding': '0'})

        # Function to start the Dash server
        def run_dash():
            app.run(port=dash_port, debug=False, use_reloader=False)  # Start the Dash server

        # Function to open the browser
        def open_browser():
            time.sleep(1)  # Give the server a second to start
            webbrowser.open(f"http://127.0.0.1:{dash_port}/")  # Open the Dash app in the browser

        # Start the Dash server in a separate thread
        server_thread = threading.Thread(target=run_dash)
        server_thread.daemon = True  # Allows the program to exit even if this thread is running
        server_thread.start()

        # Open the Dash app in the default browser
        open_browser()
        server_thread.join(timeout=1)

    return network


def plot_cps_plotly(cps, layout="kamada_kawai", marker_size=20, node_positions=None, show_events=True, show_num_occur=False,
                    show_state_label=True, font_size=10, plot_self_transitions=True, use_previos_node_positions=False,
                    **kwargs):
    """
    Visualizes a Cyber-Physical System (CPS) state-transition graph using Plotly.
    This function generates an interactive Plotly figure representing the states and transitions of a CPS.
    Nodes represent system states, and edges represent transitions. Various layout algorithms and display options
    are supported.
    Args:
        cps: The CPS object containing the state-transition graph. Must have attributes `_G` (networkx graph),
            `get_transitions()`, `print_state()`, `num_occur()`, and `previous_node_positions`.
        layout (str, optional): Layout algorithm for node positioning. Options are "dot" (default), "spectral",
            "kamada_kawai", or "fruchterman_reingold".
        marker_size (int, optional): Size of the node markers. Default is 20.
        node_positions (dict, optional): Precomputed node positions as a dictionary {node: (x, y)}. If None,
            positions are computed using the selected layout.
        show_events (bool, optional): Whether to display event labels on transitions. Default is True.
        show_num_occur (bool, optional): Whether to display the number of occurrences for each transition. Default is False.
        show_state_label (bool, optional): Whether to display state labels on nodes. Default is True.
        font_size (int, optional): Font size for transition/event labels. Default is 10.
        plot_self_transitions (bool, optional): Whether to plot self-loop transitions. Default is True.
        use_previos_node_positions (bool, optional): If True and node_positions is None, reuse positions from
            `cps.previous_node_positions`. Default is False.
        **kwargs: Additional keyword arguments passed to the layout function (e.g., for networkx layouts).
    Returns:
        plotly.graph_objs.Figure: A Plotly figure object representing the CPS state-transition graph.
    Notes:
        - Requires Plotly, NetworkX, and pydotplus (for "dot" layout).
        - The CPS object must provide the required methods and attributes as described above.
        - Edge and node styling can be further customized by modifying the function.
    """
    # layout = 'kamada_kawai'  # TODO
    edge_scatter_lines = None
    if node_positions is None:
        if use_previos_node_positions:
            node_positions = cps.previous_node_positions
        else:
            g = cps._G
            if layout == "dot":
                graph = pdp.graph_from_edges([('"' + tr[0] + '"', '"' + tr[1] + '"')
                                              for tr in g.edges], directed=True)
                # graph.set_node_defaults(shape='point')
                for nnn in g.nodes:
                    graph.add_node(pdp.Node(nnn, shape='point'))
                graph.set_prog('dot')
                graph = graph.create(format="dot")

                graph = pdp.graph_from_dot_data(graph)
                node_positions = {n.get_name().strip('"'): tuple(float(x) for x in n.get_pos()[1:-1].split(','))
                                  for n in graph.get_nodes() if
                                  n.get_name().strip('"') not in ['\\r\\n', 'node', 'graph']}
                edges = {e.obj_dict['points']: e.get_pos()[3:-1].split(' ')
                         for e in graph.get_edges()}  # [3:].split(",")

                # edge_shapes = []
                # edge_scatter_lines = []
                # for points, edg in edges.items():
                #     edg = [tuple(float(eee.replace('\r', '').replace('\n', '').replace('\\', '').strip())
                #                  for eee in e.split(",")) for e in edg]
                #     node_pos_start = node_positions[points[0].replace('"', '')]
                # edg.insert(0, node_pos_finish)ääääääääääääääääääääääääääääääääääääääääääääääääääääääää
                # node_pos_finish = node_positions[points[1].replace('"', '')]
                # control_points = ' '.join(','.join(map(str, e)) for e in edg[1:])
                # {node_pos_start[0]}, {node_pos_start[1]}
                # Cubic Bezier Curves
                # edge_shapes.append(dict(
                #     type="path",
                #     path=f"M {node_pos_start[0]},{node_pos_start[1]} C {control_points}", #{node_pos_finish[0]}, {node_pos_finish[1]}",
                #     line_color="MediumPurple",
                # ))

                # edg.append(node_pos_start)

                # edg.append((None, None))
                # annotations.append(dict(ax=node_pos_finish[0], ay=node_pos_finish[1], axref='x', ayref='y',
                #     x=edg[-2][0], y=edg[-2][1], xref='x', yref='y',
                #     showarrow=True, arrowhead=1, arrowsize=2, startarrowhead=0))
                # edge_scatter_lines.append(edg)
                # parse_path(edges)
                # points_from_path(edges)
            elif layout == 'spectral':
                node_positions = nx.spectral_layout(g, **kwargs)
            elif layout == 'kamada_kawai':
                node_positions = nx.kamada_kawai_layout(g, **kwargs)
            elif layout == 'fruchterman_reingold':
                node_positions = nx.fruchterman_reingold_layout(g, **kwargs)
        cps.previous_node_positions = node_positions
    node_x = []
    node_y = []
    for node in cps._G.nodes:
        x, y = node_positions[node]
        node_x.append(x)
        node_y.append(y)
    texts = []
    for v in cps._G.nodes:
        try:
            texts.append(cps.print_state(v))
        except:
            texts.append('Error printing state: ')
    if show_state_label:
        mode = 'markers+text'
    else:
        mode = 'markers'
    node_trace = go.Scatter(x=node_x, y=node_y, text=list(cps._G.nodes), mode=mode, textposition="top center",
                            hovertext=texts, hovertemplate='%{hovertext}<extra></extra>',
                            marker=dict(size=marker_size, line_width=1), showlegend=False)

    annotations = [dict(ax=node_positions[tr[0]][0], ay=node_positions[tr[0]][1], axref='x', ayref='y',
                        x=node_positions[tr[1]][0], y=node_positions[tr[1]][1], xref='x', yref='y',
                        showarrow=True, arrowhead=1, arrowsize=2) for tr in cps._G.edges]

    # annotations = []

    def fun(tr):
        if show_events and show_num_occur:
            return '<i>{} ({})</i>'.format(tr[2], cps.num_occur(tr))
        elif show_events:
            return '<i>{}</i>'.format(tr[2])
        elif show_num_occur:
            return '<i>{}</i>'.format(cps.num_occur(tr))

    if show_num_occur or show_events:
        annotations_text = [dict(x=(0.4 * node_positions[tr[0]][0] + 0.6 * node_positions[tr[1]][0]),
                                 y=(0.4 * node_positions[tr[0]][1] + 0.6 * node_positions[tr[1]][1]),
                                 xref='x', yref='y', text=fun(tr), font=dict(size=font_size, color='darkblue'),
                                 yshift=0, showarrow=False)  # , bgcolor='white')
                            for tr in cps.get_transitions() if plot_self_transitions or tr[0] != tr[1]]

        annotations += annotations_text

    traces = [node_trace]
    if edge_scatter_lines:
        edge_scatter_lines = list(chain(*edge_scatter_lines))
        edge_trace = go.Scatter(x=[xx[0] for xx in edge_scatter_lines], y=[xx[1] for xx in edge_scatter_lines],
                                mode='lines', showlegend=False, line=dict(color='black', width=1), hoverinfo=None,
                                hovertext=None, name='Transitions')
        traces.insert(0, edge_trace)

    fig = go.Figure(data=traces, layout=go.Layout(annotations=annotations,
                                                  paper_bgcolor='rgba(0,0,0,0)',
                                                  plot_bgcolor='rgba(0,0,0,0)'))

    fig.update_xaxes({'showgrid': False,  # thin lines in the background
                      'zeroline': False,  # thick line learn x=0
                      'visible': False})
    # 'fixedrange': True})  # numbers below)
    fig.update_yaxes({'showgrid': False,  # thin lines in the background
                      'zeroline': False,  # thick line learn x=0
                      'visible': False})
    # 'fixedrange': True})  # numbers below)
    fig.update_annotations(standoff=marker_size / 2, startstandoff=marker_size / 2)
    fig.update_layout(clickmode='event')
    return fig


def view_graphviz(self, layout="dot", marker_size=20, node_positions=None, show_events=True, show_num_occur=False,
                  show_state_label=True, font_size=10, plot_self_transitions=True, use_previos_node_positions=False,
                  **kwargs):
    """
    Visualizes the internal graph structure using Graphviz and returns a pydot graph object.
    Parameters:
        layout (str): The layout algorithm to use for node positioning (default: "dot").
        marker_size (int): Size of the node markers in the visualization (default: 20).
        node_positions (dict or None): Optional dictionary mapping node names to (x, y) positions. If None, positions are computed.
        show_events (bool): Whether to display event labels on transitions (default: True).
        show_num_occur (bool): Whether to display the number of occurrences for each transition (default: False).
        show_state_label (bool): Whether to display state labels on nodes (default: True).
        font_size (int): Font size for labels and annotations (default: 10).
        plot_self_transitions (bool): Whether to plot self-loop transitions (default: True).
        use_previos_node_positions (bool): Whether to reuse previously computed node positions (default: False).
        **kwargs: Additional keyword arguments for customization.
    Returns:
        pdp.Dot: A pydot graph object representing the visualized graph.
    Notes:
        - Node positions are either computed using Graphviz or taken from the provided/previous positions.
        - Annotations for transitions can include event names and/or occurrence counts.
        - The function prepares the graph for further rendering or export, but does not display it directly.
    """

    graph = None
    if node_positions is None:
        if use_previos_node_positions:
            node_positions = self.previous_node_positions
        else:
            g = self._G
            graph = pdp.graph_from_edges([('"' + tr[0] + '"', '"' + tr[1] + '"') for tr in g.edges], directed=True)
            for nnn in g.nodes:
                graph.add_node(pdp.Node(nnn, shape='point'))
            graph.set_prog('dot')
            graph = graph.create(format="dot")
            graph = pdp.graph_from_dot_data(graph)
            node_positions = {n.get_name().strip('"'): tuple(float(x) for x in n.get_pos()[1:-1].split(','))
                              for n in graph.get_nodes() if
                              n.get_name().strip('"') not in ['\\r\\n', 'node', 'graph']}
        self.previous_node_positions = node_positions
    node_x = []
    node_y = []
    for node in self._G.nodes:
        x, y = node_positions[node]
        node_x.append(x)
        node_y.append(y)
    texts = []
    for v in self._G.nodes:
        try:
            texts.append(self.print_state(v))
        except:
            texts.append('Error printing state: ')

    annotations = [dict(ax=node_positions[tr[0]][0], ay=node_positions[tr[0]][1], axref='x', ayref='y',
                        x=node_positions[tr[1]][0], y=node_positions[tr[1]][1], xref='x', yref='y',
                        showarrow=True, arrowhead=1, arrowsize=2) for tr in self._G.edges]
    def fun(tr):
        if show_events and show_num_occur:
            return '<i>{} ({})</i>'.format(tr[2], self.num_occur(tr[0], tr[2]))
        elif show_events:
            return '<i>{}</i>'.format(tr[2])
        elif show_num_occur:
            return '<i>{}</i>'.format(self.num_occur(tr[0], tr[2]))

    if show_num_occur or show_events:
        annotations_text = [dict(x=(0.4 * node_positions[tr[0]][0] + 0.6 * node_positions[tr[1]][0]),
                                 y=(0.4 * node_positions[tr[0]][1] + 0.6 * node_positions[tr[1]][1]),
                                 xref='x', yref='y', text=fun(tr), font=dict(size=font_size, color='darkblue'),
                                 yshift=0, showarrow=False)
                            for tr in self.get_transitions() if plot_self_transitions or tr[0] != tr[1]]

        annotations += annotations_text

    graph = pdp.Dot(graph_type='digraph')
    for tr in self._G.edges:
        graph.add_edge(pdp.Edge('"' + tr[0] + '"', '"' + tr[1] + '"', label=tr[2]))
    for nnn in self._G.nodes:
        graph.add_node(pdp.Node(nnn, shape='box'))
    return graph


def plot_transition(self, s, d):
    """
    Plots the transition histogram between two states.
    Retrieves the transition data between the source state `s` and destination state `d`,
    and generates a Plotly figure visualizing the timing distribution of the transition.
    The plot includes a title, an annotation indicating the transition, and a histogram
    of the transition timings.
    Args:
        s: The source state identifier.
        d: The destination state identifier.
    Returns:
        plotly.graph_objs._figure.Figure: A Plotly Figure object containing the histogram
        of transition timings.
    """

    trans = self.get_transition(s, d)
    titles = '{0} -> {1} -> {2}'.format(trans[0], trans[2], trans[1])
    fig = go.Figure()
    fig.update_layout(title=trans[2], font=dict(size=6))
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.9,
        text= '{} -> {}'.format(trans[0], trans[1]))
    v = trans[3]['timing']
    fig.add_trace(go.Histogram(x=[o.total_seconds() for o in v],
                               name='Timings'))
    return fig


def plot_state_transitions(ta, state, obs=None):
    """
    Visualizes the outgoing state transitions from a given state in a timed automaton, along with associated observation data.
    Parameters:
        ta: An object representing the timed automaton, expected to have an `out_transitions(state)` method that returns transitions from the given state.
        state: The current state for which outgoing transitions and associated observations are to be visualized.
        obs (optional): A pandas DataFrame containing observation data. Must include at least the columns 'Mode', 'q_next', 'Duration', 'Time', and optionally 'Vergussgruppe', 'HID', 'ChipID', 'Order', and 'ArtNr'. If None, the function raises NotImplemented.
    Returns:
        fig: A Plotly figure object containing subplots for each outgoing transition. For each transition, the function displays:
            - A scatter plot of observation durations over time, grouped by 'Vergussgruppe'.
            - A histogram of durations for each 'Vergussgruppe'.
        The subplots are arranged with shared axes and appropriate titles for each transition.
    Raises:
        NotImplemented: If `obs` is None.
    Notes:
        - The function expects certain columns to exist in the `obs` DataFrame. If missing, default values are assigned.
        - Colors for different 'Vergussgruppe' groups are assigned from `DEFAULT_PLOTLY_COLORS`.
        - The function uses Plotly's `make_subplots`, `go.Scatter`, and `go.Histogram` for visualization.
    """

    trans = ta.out_transitions(state)
    titles = []
    for k in trans:
        titles.append('State: {0} -> {1} -> {2}'.format(k[0], k[3]['event'], k[1]))
        titles.append('')

    fig = subplots.make_subplots(len(trans), 2, shared_xaxes=True, shared_yaxes=True,
                                 subplot_titles=titles, column_widths=[0.8, 0.2],
                                 horizontal_spacing=0.02, vertical_spacing=0.2)
    if obs is None:
        raise NotImplemented()
        # observations = self.get_transition_observations(state)

    obs = obs[obs['Mode'] == state]
    ind = 0
    for k in trans:
        v = obs[obs.q_next == k[1]]
        ind += 1
        ind_color = 0
        if len(v) == 0:
            continue
        # v['VG'] = 'Unknown'
        if 'Vergussgruppe' in v:
            v['Vergussgruppe'] = v['Vergussgruppe'].fillna('Unknown')
        else:
            v['Vergussgruppe'] = 'Unknown'

        v['Order'] = 'Unknown'
        v['ChipID'] = 'Unknown'
        v['Item'] = 'Unknown'
        v['ArtNr'] = 'Unknown'
        for vg, vv in v.groupby('Vergussgruppe'):
            vv = vv.to_dict('records')
            fig.add_trace(go.Histogram(y=[o['Duration'] for o in vv],
                                       name=vg,
                                       marker_color=DEFAULT_PLOTLY_COLORS[ind_color]), row=ind, col=2)
            ind_color += 1

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5, row=ind, col=2)

        ind_color = 0
        # v = pd.DataFrame(v)

        v['Item'] = v['HID']
        for vg, vv in v.groupby('Vergussgruppe'):
            vv = vv.to_dict('records')
            hovertext = [
                'Timing: {}s<br>Zähler: {}<br>ChipID: {}<br>Order: {}<br>VG: {}<br>ArtNr: {}'.format(o['Duration'],
                                                                                                     o['Item'],
                                                                                                     o['ChipID'],
                                                                                                     o['Order'],
                                                                                                     o['Vergussgruppe'],
                                                                                                     o['ArtNr'])
                for o in vv]
            fig.add_trace(go.Scatter(x=[o['Time'] for o in vv], y=[o['Duration'] for o in vv],
                                     marker=dict(size=6, symbol="circle", color=DEFAULT_PLOTLY_COLORS[ind_color]),
                                     name=vg,
                                     mode="markers",
                                     hovertext=hovertext), row=ind, col=1)
            ind_color += 1
        fig.update_xaxes(showticklabels=True, row=ind, col=1)
    fig.update_layout(showlegend=False, margin=dict(b=0, t=30), width=1200)
    return fig


# def plot_bipartite_graph(network):
#     """
#     Plots a bipartite graph of a network graph.
#     :param network: networkx network or a bi-adjacency matrix.
#     :return:
#     """
#     if type(network) is np.array:
#         # Iterate through each column (edge) of the bi-adjacency matrix
#         edges = []
#         for col_name in network.columns:
#             col = network[col_name]
#             inflow = col.index[col == 1]
#             outflow = col.index[col == -1]
#             edges += [(col_name, inf, 1) for inf in inflow] + [(col_name, outf, -1) for outf in outflow]
#
#         SM = nx.DiGraph()
#         SM.add_weighted_edges_from(edges)
#     else:
#         SM = network
#
#     if not nx.bipartite.is_bipartite(SM):
#         raise Exception("Not bipartite graph")
#     top = nx.bipartite.sets(SM)[0]
#     pos = nx.bipartite_layout(SM, top)
#     nx.draw(SM, pos=pos, with_labels=True, node_color='skyblue', edge_color='black', font_color='red', node_size=800,
#             font_size=10)
#     # Draw edge labels (weights)
#     edge_labels = nx.get_edge_attributes(SM, 'weight')
#     nx.draw_networkx_edge_labels(SM, pos, edge_labels=edge_labels,  label_pos=0.7, font_color='blue')
#     plt.show()


def plot_dash_frames(graph_frames, dash_port=8050):
    """
    Launches an interactive Dash web application to visualize a sequence of graph frames with a slider for manual frame selection.
    Args:
        graph_frames (list): A list of Dash components (e.g., Cytoscape graphs) representing different frames to display.
        dash_port (int, optional): The port number on which to run the Dash server. Defaults to 8050.
    Returns:
        dash.Dash: The Dash application instance.
    Side Effects:
        - Starts a Dash server in a separate thread.
        - Opens the default web browser to display the Dash app.
        - Waits for user input before returning.
    Notes:
        - The app displays the first frame by default and allows users to select other frames using a slider.
        - The function blocks until the user presses Enter in the console.
    """

    app = Dash(__name__)
    app.layout = html.Div(children=graph_frames[0], style={'width': '100%',
                                                           'height': '100vh',
                                                           'margin': '0',
                                                           'padding': '0'})

    app.layout = html.Div([
        html.Div(graph_frames[0], id='cytoscape-graph'),

        # Slider for manual frame selection
        dcc.Slider(
            id='graph-slider',
            min=0,
            max=len(graph_frames) - 1,
            step=1,
            marks={i: str(i) for i in range(len(graph_frames))},  # Label frames
            value=0,  # Start at first frame
        ),
    ])


    # Callback to update Cytoscape graph when slider changes
    @app.callback(
        Output('cytoscape-graph', 'children'),
        Input('graph-slider', 'value')
    )
    def update_graph(frame_idx):
        print(frame_idx)
        return graph_frames[frame_idx]  # Update Cytoscape graph

    # Function to start the Dash server
    def run_dash():
        app.run_server(port=dash_port, debug=False, use_reloader=False)  # Start the Dash server

    # Function to open the browser
    def open_browser():
        time.sleep(1)  # Give the server a second to start
        webbrowser.open(f"http://127.0.0.1:{dash_port}/")  # Open the Dash app in the browser

    # Start the Dash server in a separate thread
    server_thread = threading.Thread(target=run_dash)
    server_thread.daemon = True  # Allows the program to exit even if this thread is running
    server_thread.start()

    # Open the Dash app in the default browser
    open_browser()
    # server_thread.join(timeout=1)

    input("Press Enter to continue...")
    return app

def plot_execution_tree(graph, nodes_to_color, color, font_size=30):
    """
    Plots a system execution tree as a graph, where the horizontal position of nodes corresponds to their timestamps and the tree branches vertically.
    Args:
        graph (networkx.DiGraph): A directed graph where each node represents a system state, and edges represent transitions. 
            Each node should have a 'label' (str) and 'weight' (int) attribute. Node names must be timestamp strings in the format "%d/%m/%Y, %H:%M:%S".
        nodes_to_color (list): List of node identifiers (timestamp strings) to be highlighted with a specific color.
        color (str): The color to use for highlighting nodes in `nodes_to_color`.
        font_size (int, optional): Font size for node labels in the visualization. Defaults to 30.
    Returns:
        cyto.Cytoscape: A Dash Cytoscape object representing the execution tree visualization, with nodes positioned by timestamp and colored as specified.
    Notes:
        - The function assumes the first node in `graph.nodes` is the starting node.
        - Node positions are determined by the time difference from the start node (x-axis) and their 'weight' attribute (y-axis).
        - Nodes in `nodes_to_color` are colored with the specified `color`; all others are gray.
        - Requires the `cyto` (Dash Cytoscape) library and `datetime` module.
    """

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

def plot2d(df, x=None, y=None, mode='markers', hovercolumns=None, figure=False, **args):
    """
    Creates a 2D scatter or line plot using Plotly based on the provided DataFrame columns.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        x (str, optional): The column name to use for the x-axis.
        y (str, optional): The column name to use for the y-axis.
        mode (str, optional): The Plotly scatter mode (e.g., 'markers', 'lines'). Defaults to 'markers'.
        hovercolumns (list of str, optional): List of column names to include in the hover tooltip.
        figure (bool, optional): If True, returns a Plotly Figure object; otherwise, returns a Scatter trace. Defaults to False.
        **args: Additional keyword arguments passed to the Plotly Scatter constructor.
    Returns:
        plotly.graph_objs._scatter.Scatter or plotly.graph_objs._figure.Figure:
            The generated Plotly Scatter trace or Figure, depending on the 'figure' parameter.
    Example:
        plot2d(df, x='feature1', y='feature2', hovercolumns=['label'], mode='markers', figure=True)
    """

    hovertemplate = f"{x}: %{{x}}<br>{y}: %{{y}}"
    customdata = None
    if hovercolumns:
        customdata = df[hovercolumns]
        for ind, c in enumerate(hovercolumns):
            hovertemplate += f"<br>{c}: %{{customdata[{ind}]}}"

    trace = go.Scatter(x=df[x], y=df[y], mode=mode, customdata=customdata, hovertemplate=hovertemplate, **args)
    if figure:
        return go.Figure(data=trace)
    return trace


def plot_2d_contour_from_fun(fun, rangex=None, rangey=None, th=50, **kwargs):
    """
    Plots a 2D contour of a function over a specified range.
    Parameters:
        fun (callable): A function that takes a 2D array of shape (n_points, 2) and returns a 1D array of function values.
        rangex (tuple, optional): The range for the x-axis as (min, max). Defaults to (-5, 5) if not provided.
        rangey (tuple, optional): The range for the y-axis as (min, max). Defaults to (-5, 5) if not provided.
        th (int, optional): Unused parameter, kept for compatibility. Defaults to 50.
        **kwargs: Additional keyword arguments passed to the plotly.graph_objs.Contour constructor.
    Returns:
        plotly.graph_objs.Contour: A Plotly contour plot object representing the function values over the specified range.
    """

    if rangex is None:
        rangex = (-5, 5)

    if rangey is None:
        rangey = (-5, 5)

    x = np.linspace(rangex[0], rangex[-1], 100)
    y = np.linspace(rangey[0], rangey[-1], 100)
    [dx, dy] = np.meshgrid(x, y)
    d = np.column_stack([dx.flatten(), dy.flatten()])
    f = fun(d)

    contours = list(f)
    contours.sort()
    contours = contours[0:1000:]
    return go.Contour(x=x, y=y, z=np.reshape(f, dx.shape), contours=dict(coloring='lines'), **kwargs)
    #dict(start=0,
    # end=100,
    # size=2,
    # coloring='lines'), **kwargs)


def plot3d(df, x=None, y=None, z=None, mode='markers', hovercolumns=None, **args):
    """
    Creates a 3D scatter plot using Plotly's Scatter3d, with customizable axes, hover information, and additional plot arguments.
    Parameters:
        df (pandas.DataFrame): The data source containing columns for x, y, z, and optional hover data.
        x (str, optional): The column name in `df` to use for the x-axis.
        y (str, optional): The column name in `df` to use for the y-axis.
        z (str, optional): The column name in `df` to use for the z-axis.
        mode (str, optional): Plotly scatter mode (e.g., 'markers', 'lines'). Defaults to 'markers'.
        hovercolumns (list of str, optional): List of column names in `df` to include in the hover tooltip.
        **args: Additional keyword arguments passed to `go.Scatter3d`.
    Returns:
        plotly.graph_objs._scatter3d.Scatter3d: A Plotly 3D scatter plot object configured with the specified data and options.
    """

    hovertemplate = f"{x}: %{{x}}<br>{y}: %{{y}}<br>{z}: %{{z}}"
    customdata = None
    if hovercolumns:
        customdata = df[hovercolumns]
        for ind, c in enumerate(hovercolumns):
            hovertemplate += f"<br>{c}: %{{customdata[{ind}]}}"
    return go.Scatter3d(x=df[x], y=df[y], z=df[z], mode=mode, customdata=customdata, hovertemplate=hovertemplate, **args)


def export_plotly_frames_animation_cumulative(
    fig: go.Figure,
    output_path: str,
    fps: int = 5,
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    cleanup_frames: bool = True,
    frame_dir: Optional[str] = None,
) -> str:
    """
    Export a Plotly animated figure by cumulatively applying go.Frame updates.
    This better matches Plotly's animation behavior for partial frame updates.
    """

    if not fig.frames:
        raise ValueError("The figure has no frames.")

    output = Path(output_path)
    suffix = output.suffix.lower()

    if suffix not in {".gif", ".mp4"}:
        raise ValueError("output_path must end with '.gif' or '.mp4'.")

    frames_folder = Path(frame_dir) if frame_dir else output.with_name(f"{output.stem}_frames")
    frames_folder.mkdir(parents=True, exist_ok=True)

    state_fig = go.Figure(data=deepcopy(fig.data), layout=deepcopy(fig.layout))
    rendered_paths = []

    for i, frame in enumerate(fig.frames):
        # Apply data updates onto persistent state.
        if frame.data:
            target_traces = frame.traces if frame.traces is not None else list(range(len(frame.data)))
            for trace_update, trace_index in zip(frame.data, target_traces):
                while trace_index >= len(state_fig.data):
                    state_fig.add_trace(go.Scatter())
                state_fig.data[trace_index] = trace_update

        # Apply layout updates onto persistent state.
        if frame.layout:
            state_fig.update_layout(frame.layout)

        png_path = frames_folder / f"frame_{i:04d}.png"
        pio.write_image(state_fig, str(png_path), width=width, height=height, scale=scale)
        rendered_paths.append(png_path)

    images = [imageio.imread(path) for path in rendered_paths]

    if suffix == ".gif":
        imageio.mimsave(str(output), images, fps=fps)
    else:
        with imageio.get_writer(str(output), fps=fps) as writer:
            for img in images:
                writer.append_data(img)

    if cleanup_frames:
        for path in rendered_paths:
            path.unlink(missing_ok=True)
        try:
            frames_folder.rmdir()
        except OSError:
            pass

    return str(output)



def export_plotly_frames_animation(
    fig: go.Figure,
    output_path: str,
    fps: int = 5,
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    cleanup_frames: bool = True,
    frame_dir: Optional[str] = None,
) -> str:
    if not fig.frames:
        raise ValueError("The figure has no frames.")

    output = Path(output_path)
    suffix = output.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("output_path must end with '.gif' or '.mp4'.")

    frames_folder = Path(frame_dir) if frame_dir else output.with_name(f"{output.stem}_frames")
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Persistent state figure
    state_fig = go.Figure(data=deepcopy(fig.data), layout=deepcopy(fig.layout))
    rendered_paths = []

    for i, frame in enumerate(fig.frames):
        # Convert tuple -> list so we can replace traces safely
        current_data = list(deepcopy(state_fig.data))

        if frame.data:
            if frame.traces is not None:
                for trace_update, trace_index in zip(frame.data, frame.traces):
                    current_data[trace_index] = trace_update
            else:
                for j, trace_update in enumerate(frame.data):
                    if j < len(current_data):
                        current_data[j] = trace_update
                    else:
                        current_data.append(trace_update)

        # Rebuild figure from updated trace list
        current_fig = go.Figure(data=current_data, layout=deepcopy(state_fig.layout))

        if frame.layout:
            current_fig.update_layout(frame.layout)

        # Update persistent state so next frame is cumulative
        state_fig = go.Figure(data=deepcopy(current_fig.data), layout=deepcopy(current_fig.layout))

        png_path = frames_folder / f"frame_{i:04d}.png"
        pio.write_image(current_fig, str(png_path), width=width, height=height, scale=scale)
        rendered_paths.append(png_path)

    images = [imageio.imread(path) for path in rendered_paths]

    if suffix == ".gif":
        imageio.mimsave(str(output), images, fps=fps)
    else:
        with imageio.get_writer(str(output), fps=fps) as writer:
            for img in images:
                writer.append_data(img)

    if cleanup_frames:
        for path in rendered_paths:
            path.unlink(missing_ok=True)
        try:
            frames_folder.rmdir()
        except OSError:
            pass

    return str(output)



def add_time_frames_to_subplots(
    fig: go.Figure,
    trace_indices: Optional[Sequence[int]] = None,
    frame_duration_ms: int = 80,
    transition_duration_ms: int = 0,
    redraw: bool = False,
    keep_tail: bool = True,
    sort_each_trace_by_x: bool = True,
    frame_stride: int = 1,
    slider_prefix: str = "Time: ",
    title_prefix: Optional[str] = None,
) -> go.Figure:
    """
    Add animation frames to a subplot figure where each subplot has one scatter trace
    and x represents time.

    Parameters
    ----------
    fig : go.Figure
        A subplot figure.
    trace_indices : sequence[int] | None
        Which traces to animate. If None, animate all traces in fig.data.
    frame_duration_ms : int
        Duration of each frame in milliseconds.
    transition_duration_ms : int
        Duration of frame transition in milliseconds.
    redraw : bool
        Passed to Plotly animation args.
    keep_tail : bool
        If True, each frame shows all points up to time t.
        If False, each frame shows only the current point for each trace.
    sort_each_trace_by_x : bool
        If True, sort each trace by its own x values before animating.
    frame_stride : int
        Use every n-th point to reduce number of frames.
    slider_prefix : str
        Prefix shown before current slider value.
    title_prefix : str | None
        Optional title prefix, e.g. "t = ".

    Returns
    -------
    go.Figure
        The same figure, updated in place with frames and controls.
    """

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    if trace_indices is None:
        trace_indices = list(range(len(fig.data)))
    else:
        trace_indices = list(trace_indices)

    if not trace_indices:
        raise ValueError("No traces selected for animation")

    time_steps = []
    for trace_idx in trace_indices:
        if trace_idx >= len(fig.data):
            raise IndexError(f"trace_index={trace_idx} out of range for figure with {len(fig.data)} traces")

        tr = fig.data[trace_idx]

        if not hasattr(tr, "x") or not hasattr(tr, "y"):
            raise TypeError(f"Trace {trace_idx} does not have x/y data")

        x = list(tr.x) if tr.x is not None else []
        y = list(tr.y) if tr.y is not None else []

        if len(x) == 0 or len(y) == 0:
            raise ValueError(f"Trace {trace_idx} has empty x or y data")
        if len(x) != len(y):
            raise ValueError(f"Trace {trace_idx}: x and y must have same length")

        time_steps += x

    time_steps = np.sort(np.unique(time_steps))
    # Use the shortest trace length so all subplots stay synchronized
    n_frames = len(time_steps) // frame_stride
    if n_frames == 0:
        raise ValueError("No frames can be created")

    frames = []
    slider_steps = []

    for k in range(1, n_frames + 1):
        frame_data = []

        for p in fig.data:
            frame_ind_mask = p['x'] <= time_steps[k-1]
            xk = p["x"][frame_ind_mask]
            yk = p["y"][frame_ind_mask]
            textk = p["text"][frame_ind_mask] if p["text"] is not None else None
            customdatak = p["customdata"][frame_ind_mask] if p["customdata"] is not None else None
            idsk = p["ids"][frame_ind_mask] if p["ids"] is not None else None

            frame_data.append(
                go.Scatter(
                    x=xk,
                    y=yk,
                    text=textk,
                    customdata=customdatak,
                    ids=idsk,
                    mode=p["mode"],
                    marker=p["marker"],
                    line=p["line"],
                    name=p["name"],
                    hovertemplate=p["hovertemplate"],
                )
            )

        frame_name = f"frame_{k - 1}"
        current_time = time_steps[k - 1]

        frame_layout = {}
        if title_prefix is not None:
            frame_layout["title"] = {"text": f"{title_prefix}{current_time}"}

        frames.append(
            go.Frame(
                name=frame_name,
                data=frame_data,
                traces=None,
                layout=frame_layout if frame_layout else None,
            )
        )

        slider_steps.append(
            {
                "method": "animate",
                "label": str(current_time),
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": frame_duration_ms, "redraw": redraw},
                        "transition": {"duration": transition_duration_ms},
                        "mode": "immediate",
                    },
                ],
            }
        )

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.1,
                "y": -0.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": redraw},
                                "transition": {"duration": transition_duration_ms},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": redraw},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "y": -0.08,
                "len": 0.85,
                "pad": {"t": 20},
                "currentvalue": {"prefix": slider_prefix},
                "steps": slider_steps,
            }
        ],
    )

    return fig


def plot_images(images):
    fig = make_subplots(
        rows=4,
        cols=4,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    for i, img in enumerate(images[:16]):
        row = i // 4 + 1
        col = i % 4 + 1

        if hasattr(img, "detach"):  # torch.Tensor
            img = img.detach().cpu().numpy()

        img = np.asarray(img)

        fig.add_trace(
            go.Heatmap(
                z=img,
                colorscale=[[0, "black"], [1, "white"]],
                zmin=0,
                zmax=1,
                showscale=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")

    fig.update_layout(
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="white",
    )
    return fig

