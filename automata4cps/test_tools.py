import dash  # pip install dash
import dash_cytoscape as cyto  # pip install dash-cytoscape==0.2.0 or higher
from dash import html
from dash import dcc
from dash import Output, Input, State, callback
import dash_daq as daq
import pandas as pd  # pip install pandas
import plotly.express as px
from datetime import datetime, timedelta
import networkx as nx
import random
from tools import plot_execution_tree


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

num_of_nodes = 14
# num_of_states = 4 # ???

graph = nx.DiGraph()
dates = [
"01/01/2024, 00:00:00",
"09/01/2024, 08:33:56",
"25/02/2024, 02:09:02",
"28/03/2024, 21:52:42",
"05/06/2024, 12:02:24",
"17/06/2024, 18:32:31",
"11/07/2024, 21:10:53",
"19/07/2024, 08:44:27",
"02/09/2024, 12:17:23",
"24/10/2024, 07:34:56",
"14/11/2024, 18:58:43",
"20/11/2024, 01:42:40",
"19/12/2024, 11:43:14",
"30/12/2024, 21:12:57"
]

graph.add_node(dates[0], label = dates[0], weight = 0)
for i in range(1, num_of_nodes):
    graph.add_node(dates[i], label = dates[i], weight = random.randint(1, num_of_states))

graph.add_edge("01/01/2024, 00:00:00", "09/01/2024, 08:33:56")
graph.add_edge("01/01/2024, 00:00:00", "05/06/2024, 12:02:24")
graph.add_edge("01/01/2024, 00:00:00", "17/06/2024, 18:32:31")
graph.add_edge("01/01/2024, 00:00:00", "11/07/2024, 21:10:53")
graph.add_edge("09/01/2024, 08:33:56", "25/02/2024, 02:09:02")
graph.add_edge("25/02/2024, 02:09:02", "28/03/2024, 21:52:42")
graph.add_edge("11/07/2024, 21:10:53", "19/07/2024, 08:44:27")
graph.add_edge("19/07/2024, 08:44:27", "02/09/2024, 12:17:23")
graph.add_edge("02/09/2024, 12:17:23", "24/10/2024, 07:34:56")
graph.add_edge("28/03/2024, 21:52:42", "14/11/2024, 18:58:43")
graph.add_edge("28/03/2024, 21:52:42", "20/11/2024, 01:42:40")
graph.add_edge("24/10/2024, 07:34:56", "19/12/2024, 11:43:14")
graph.add_edge("24/10/2024, 07:34:56", "30/12/2024, 21:12:57")

ntc = [
"28/03/2024, 21:52:42",
"05/06/2024, 12:02:24",
"17/06/2024, 18:32:31",
"11/07/2024, 21:10:53"
]
ntd = [
"14/12/2024, 18:58:43",
"20/11/2024, 01:42:40"
]

app.layout = html.Div(
        #dcc.Store(id='css-variable-store', data={'--x-translate': get_translation_data(graph, node)[0], '--y-translate': get_translation_data(graph, node)[0]}),
    plot_execution_tree(graph, ntc, 'red', ntd))

# graphs = [graph1, graph2]
# plot_animation(graphs)

app.run()