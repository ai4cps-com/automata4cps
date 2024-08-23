"""
    The module provides methods to visualize various kinds of data, such as time series or automata graphs.

    Authors:
    - Nemanja Hranisavljevic, hranisan@hsu-hh.de, nemanja@ai4cps.com
    - Tom Westermann, tom.westermann@hsu-hh.de, tom@ai4cps.com
"""

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
from dash import html, Dash
import dash_bootstrap_components as dbc
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


def plot_timeseries(data, title=None, timestamp=None, use_columns=None, discrete=False, height=None, plotStates=False,
              limit_num_points=None, names=None, xaxis_title=None,
              customdata=None, iterate_colors=True, y_title_font_size=14, opacity=1, vertical_spacing=0.005,
              sharey=False, bounds=None, plot_only_changes=False, yAxisLabelOffset=False, marker_size=4,
              showlegend=False, mode='lines+markers', **kwargs):
    """
    Using plotly library, plots each variable (column) in a collection of dataframe as subplots, one after another.

    Arguments:
    plotStates (bool): if True, changes interpolation to 'hv', therefore keeping the last value
    yAxisLabelOffset (bool): if True, adds an offset to the plots y-axis labels. Improves readability on long subplot names.

    Returns:
        fig (plotly.Figure):
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
                color = DEFAULT_PLOTLY_COLORS[k % len(DEFAULT_PLOTLY_COLORS)]
            else:
                color = DEFAULT_PLOTLY_COLORS[0]

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

    if title is not None:
        fig.update_layout(title={'text': title, 'x': 0.5}, autosize=True, height=height + 180, showlegend=showlegend)
    return fig


def plot_stateflow(stateflow, color_mapping=None, state_col='State', bar_height=12,
                   start_column='Start', finish_column='Finish', return_figure=False, description_col='Description',
                   idle_states=None):
    if idle_states is None:
        idle_states = []
    if type(idle_states) is str:
        idle_states = [idle_states]

    stateflow_df_list = []
    for station, s in stateflow.items():
        if s.size:
            sf = s[(~s.State.isin(idle_states))]
                    # ((start_plot <= s.Timestamp) &
                    #  (s.Timestamp <= finish_plot)) |
                    # ((start_plot <= s.Finish) & (s.Finish <= finish_plot)))
            sf['Task'] = station
            # if sf.size > 0 and pd.isnull(sf['Finish'].iloc[-1]):
            #     sf['Finish'].iloc[-1] = pd.to_datetime(finish_plot)
            # s['Finish'] = pd.to_datetime(s['Finish'])
            stateflow_df_list.append(sf)
        else:
            stateflow_df_list.append(pd.DataFrame([]))
    stateflow_df = pd.concat(stateflow_df_list)

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
        text = []
        for k, row in g.iterrows():  # , g[item_col], g.Source, g.Destination):
            x1, x2, tsk = row[start_column], row[finish_column], row['Task']
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
                if not pd.isnull(val) and k not in [state_col, finish_column, start_column, 'Task', 'Duration']:
                    ht += f'<br>{k}: {val}'
            hovertext.append(ht)
        traces.append(go.Scattergl(x=x, y=y, line=dict(width=bar_height), name=name, line_color=color_mapping.get(name, "b"),
                                   hoverinfo='skip', mode='lines', legendgroup=name, showlegend=True))
        traces.append(go.Scattergl(x=np.asarray(g[start_column] + g.Duration / 2), y=g.Task, mode='text+markers',
                                   marker=dict(size=1), name=name, marker_color=color_mapping.get(name, "b"),
                                   showlegend=False,
                                   hovertext=hovertext, text=text, textfont=dict(size=10, color='olive'),
                                   hovertemplate=f'<extra></extra><b>{name}</b><br>%{{hovertext}}'))

    if return_figure:
        fig = go.Figure(data=traces)
        return fig
    else:
        return traces


def plot_cps_component(cps, id=None, node_labels=False, edge_labels=True, edge_font_size=6, edge_text_max_width=None,
                       output="dash"):
    """

    :param cps:
    :param id:
    :param node_labels:
    :param edge_labels:
    :param edge_font_size:
    :param edge_text_max_width:
    :param output:
    :return:
    """
    if id is None:
        id = "graph"
    nodes = []
    for n in cps.discrete_states:
        nodes.append(dict(data={'id': n, 'label': n}))

    edges = []
    for e in cps.get_transitions():
        if 'timing' in e[3]:
            freq = len(e[3]['timing'])
            timings = [pd.Timedelta(x) for x in e[3]['timing']]
        else:
            freq = 0
            timings = []
        edge = dict(data={'source': e[0],
                          'target': e[1],
                          'label': f'{e[3]["event"]} [{freq}]',
                          'timing': timings})
        edges.append(edge)

    elements = dict(nodes=nodes, edges=edges)

    if output == "elements":
        return elements

    node_style = {'width': 10,
                  'height': 10}
    if node_labels:
        node_style['label'] = 'data(id)'
        node_style['font-size'] = 6
        node_style['text-wrap'] = 'wrap'
        node_style['text-max-width'] = 50

    edge_style = {
                # The default curve style does not work with certain arrows
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'target-arrow-size': 3,
                'width': 1,
                'font-color': 'gray',
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
        id=id,
        layout={'name': 'cose', "fit": True},
        # layout={
        #     'id': 'breadthfirst',
        #     'roots': '[id = "initial"]'
        # },
        maxZoom=2,
        minZoom=0.5,
        style={'width': '100%', 'height': '600px'}, stylesheet=stylesheet,
        elements=elements)

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
        app.run_server(mode='inline')
        return
    return network


def plot_cps(cps, node_labels=False, edge_labels=True, edge_font_size=6, edge_text_max_width=None, output="dash"):
    elements = dict(nodes=[], edges=[])
    for comid, com in cps.com.items():
        els = plot_cps_component(com, output="elements")
        for x in els['nodes']:
            x['data']['group'] = comid
            x['data']['id'] = f"{comid}-{x['data']['id']}"
        for x in els['edges']:
            x['data']['source'] = f"{comid}-{x['data']['source']}"
            x['data']['target'] = f"{comid}-{x['data']['target']}"
        elements['nodes'] += els['nodes']
        elements['edges'] += els['edges']

    node_style = {'width': 10,
                  'height': 10}
    if node_labels:
        node_style['label'] = 'data(id)'
        node_style['font-size'] = 6
        node_style['text-wrap'] = 'wrap'
        node_style['text-max-width'] = 50

    edge_style = {
        # The default curve style does not work with certain arrows
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-size': 3,
        'width': 1,
        'font-color': 'gray',
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
        id=cps.id,
        layout={'name': 'cose', "fit": True},
        # layout={
        #     'id': 'breadthfirst',
        #     'roots': '[id = "initial"]'
        # },
        maxZoom=2,
        minZoom=0.5,
        # style={'width': '100%', 'height': '600px'},
        stylesheet=stylesheet,
        elements=elements)

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
        app.run_server(mode='inline')
        return
    return network

def plot_cps_plotly(cps, layout="dot", marker_size=20, node_positions=None, show_events=True, show_num_occur=False,
                show_state_label=True, font_size=10, plot_self_transitions=True, use_previos_node_positions=False,
                **kwargs):
    """

    :param cps:
    :param layout:
    :param marker_size:
    :param node_positions:
    :param show_events:
    :param show_num_occur:
    :param show_state_label:
    :param font_size:
    :param plot_self_transitions:
    :param use_previos_node_positions:
    :param kwargs:
    :return:
    """
    # layout = 'kamada_kawai'  # TODO
    edge_scatter_lines = None
    annotations = []
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
                # graph.
                # graph.write_dot('temp.dot')
                # graph.write_svg('temp.svg')
                # graph = pdp.graph_from_dot_file('temp.dot')
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
            return '<i>{} ({})</i>'.format(tr[2], cps.num_occur(tr[0], tr[2]))
        elif show_events:
            return '<i>{}</i>'.format(tr[2])
        elif show_num_occur:
            return '<i>{}</i>'.format(cps.num_occur(tr[0], tr[2]))

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


def plot_state_transitions(self, state, obs=None):
    trans = self.out_transitions(state)
    titles = []
    for k in trans:
        titles.append('State: {0} -> {1} -> {2}'.format(k[0], k[2], k[1]))
        titles.append('')

    fig = subplots.make_subplots(len(trans), 2, shared_xaxes=True, shared_yaxes=True,
                                 subplot_titles=titles, column_widths=[0.8, 0.2],
                                 horizontal_spacing=0.02, vertical_spacing=0.2)
    if obs is None:
        raise NotImplemented()
        # observations = self.get_transition_observations(state)

    obs = obs[obs['State'] == state]
    ind = 0
    for k in trans:
        v = obs[obs.q_next == k[1]]
        ind += 1
        ind_color = 0
        if len(v) == 0:
            continue
        # v['VG'] = 'Unknown'
        for vg, vv in v.groupby('Vergussgruppe'):
            vv = vv.to_dict('records')
            fig.add_trace(go.Histogram(y=[o['Timing'].total_seconds() for o in vv],
                                       name=vg,
                                       marker_color=DEFAULT_PLOTLY_COLORS[ind_color]), row=ind, col=2)
            ind_color += 1

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5, row=ind, col=2)

        ind_color = 0
        # v = pd.DataFrame(v)
        v['Vergussgruppe'].fillna('Unknown', inplace=True)
        v['Item'] = v['HID']
        for vg, vv in v.groupby('Vergussgruppe'):
            vv = vv.to_dict('records')
            hovertext = [
                'Timing: {}s<br>Zähler: {}<br>ChipID: {}<br>Order: {}<br>VG: {}<br>ArtNr: {}'.format(o['Timing'],
                                                                                                     o['Item'],
                                                                                                     o['ChipID'],
                                                                                                     o['Order'],
                                                                                                     o['Vergussgruppe'],
                                                                                                     o['ArtNr'])
                for o in vv]
            fig.add_trace(go.Scatter(x=[o['Timestamp'] for o in vv], y=[o['Timing'].total_seconds() for o in vv],
                                     marker=dict(size=6, symbol="circle", color=DEFAULT_PLOTLY_COLORS[ind_color]),
                                     name=vg,
                                     mode="markers",
                                     hovertext=hovertext), row=ind, col=1)
            ind_color += 1
        fig.update_xaxes(showticklabels=True, row=ind, col=1)
    fig.update_layout(showlegend=False, margin=dict(b=0, t=30), width=800)
    return fig


def plot_bipartite_graph(network):
    """
    Plots a bipartite graph of a network graph.
    :param network: networkx network or a bi-adjacency matrix.
    :return:
    """
    if type(network) is np.array:
        # Iterate through each column (edge) of the bi-adjacency matrix
        edges = []
        for col_name in network.columns:
            col = network[col_name]
            inflow = col.index[col == 1]
            outflow = col.index[col == -1]
            edges += [(col_name, inf, 1) for inf in inflow] + [(col_name, outf, -1) for outf in outflow]

        SM = nx.DiGraph()
        SM.add_weighted_edges_from(edges)
    else:
        SM = network

    if not nx.bipartite.is_bipartite(SM):
        raise Exception("Not bipartite graph")
    top = nx.bipartite.sets(SM)[0]
    pos = nx.bipartite_layout(SM, top)
    nx.draw(SM, pos=pos, with_labels=True, node_color='skyblue', edge_color='black', font_color='red', node_size=800,
            font_size=10)
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(SM, 'weight')
    nx.draw_networkx_edge_labels(SM, pos, edge_labels=edge_labels,  label_pos=0.7, font_color='blue')
    plt.show()
