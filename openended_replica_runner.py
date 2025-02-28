# experiment_runner_with_dashboard.py
import random
import datetime
from threading import Thread

# -------------------------------
# GA and Fitness Function Imports
# -------------------------------

from M_E_GA import M_E_GA_Base
from Openended_replica import PathTrajectoryFitness
# -------------------------------
# Global Settings and Seed
# -------------------------------
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

# -------------------------------
# Best Organism Callback
# -------------------------------
best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# -------------------------------
# (Optional) Global Visit Counts
# -------------------------------
# If your PathTrajectoryFitness expects a global dictionary (so the visit counts
# accumulate over the entire run), define it here:
global_visit_counts = {}  # key: (x,y), value: number of visits so far

# -------------------------------
# Initialize Fitness Function & Genes
# -------------------------------
# 2) Remove the old LeadingOnesFitness with max_length, and switch to the path-based one:
fitness_function = PathTrajectoryFitness(
    visit_counts_dict=global_visit_counts,
    update_best_func=update_best_organism
)

genes = fitness_function.genes  # Should be ['left', 'right', 'forward']

# -------------------------------
# GA Configuration Parameters
# -------------------------------
config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.10,
    'open_mutation_prob': 0.08,
    'metagene_mutation_prob': 0.05,
    'delimiter_insert_prob': 0.03,
    'delimit_delete_prob': 0.06,
    'crossover_prob': 0.70,
    'elitism_ratio': 0.7,
    'base_gene_prob': 0.60,
    'metagene_prob': 0.02,
    'max_individual_length': 10,
    'population_size': 500,
    'num_parents': 300,
    'max_generations': 200,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': True,
    'crossover_logging': True,
    'individual_logging': True,
    'seed': GLOBAL_SEED,
    'lru_cache_size': 100
}

# -------------------------------
# Initialize the GA
# -------------------------------
ga = M_E_GA_Base(
    genes,
    # 3) Wrap the path-based fitness compute in a lambda so it fits the GA's interface:
    lambda ind, ga_instance: fitness_function.compute(ind, ga_instance),
    **config
)

# -------------------------------
# DASH Dashboard Setup
# -------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx

app = dash.Dash(__name__)
app.title = "Metagenome Real-Time Dashboard"

app.layout = html.Div([
    html.H1("Real-Time Metagenome Visualization"),
    html.Div(id='stats-div'),
    dcc.Graph(id='dag-graph'),
    # Show recent logger events (if any)
    html.Div(id='logger-events'),
    # Update every 5 seconds
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

# -------------------------------
# Helper functions for hierarchy computation
# -------------------------------
def compute_metagene_order(mg, encoding_manager, memo):
    if mg in memo:
        return memo[mg]
    encoding = encoding_manager.encodings.get(mg, ())
    orders = []
    for element in encoding:
        if element in encoding_manager.meta_genes:
            child_order = compute_metagene_order(element, encoding_manager, memo)
            orders.append(child_order)
        else:
            orders.append(0)
    order = max(orders, default=0) + 1
    memo[mg] = order
    return order

def get_node_order(node, encoding_manager):
    if node in encoding_manager.meta_genes:
        return compute_metagene_order(node, encoding_manager, memo={})
    else:
        return 0

def build_dag_custom(encoding_manager):
    G = nx.DiGraph()
    # Add base gene nodes
    for gene, hash_key in encoding_manager.reverse_encodings.items():
        if gene in ['Start', 'End']:
            continue
        if hash_key not in encoding_manager.meta_genes:
            G.add_node(hash_key, group="base", label=f"Base: {gene}")
    # Add metagene nodes
    for mg in encoding_manager.meta_genes:
        order = compute_metagene_order(mg, encoding_manager, memo={})
        G.add_node(mg, group="meta", label=f"MG {mg}\n(order {order})", order=order)
    # Add edges
    for mg in encoding_manager.meta_genes:
        encoding = encoding_manager.encodings.get(mg)
        if isinstance(encoding, tuple):
            for element in encoding:
                if not G.has_node(element):
                    gene_label = encoding_manager.encodings.get(element, element)
                    G.add_node(element, group="base", label=f"Base: {gene_label}")
                G.add_edge(mg, element)

    pos = {}
    order_groups = {}
    for node, data in G.nodes(data=True):
        order_val = data.get("order", 0)
        order_groups.setdefault(order_val, []).append(node)

    x_spacing = 0.3
    y_top = 1.0
    y_bottom = 0.0

    for order_val, nodes in order_groups.items():
        x_val = order_val * x_spacing
        nodes.sort(key=lambda n: str(n))
        count = len(nodes)
        for i, node in enumerate(nodes):
            if count > 1:
                y_val = y_top - i * ((y_top - y_bottom) / (count - 1))
            else:
                y_val = 0.5
            pos[node] = (x_val, y_val)

    return G, pos

@app.callback(Output('stats-div', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_stats(n):
    status = ga.encoding_manager.get_metagene_status()
    return html.Div([
        html.P(f"Current Generation: {status.get('generation', 'N/A')}"),
        html.P(f"Total Metagenes: {status.get('total_metagenes', 0)}"),
        html.P(f"Deletion Basket: {status.get('in_basket', 0)}")
    ])

@app.callback(Output('logger-events', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_logger_events(n):
    if ga.logger is not None and hasattr(ga.logger, "events"):
        events = ga.logger.events[-5:]
        items = [html.Li(f"{e['timestamp']} - {e['event_type']}: {e['details']}") for e in events]
        return html.Div([html.H4("Recent Logger Events"), html.Ul(items)])
    return "No logger events."

@app.callback(Output('dag-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_dag(n):
    G, pos = build_dag_custom(ga.encoding_manager)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color = [], [], [], []
    group_color = {"base": "orange", "meta": "blue"}
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get("label", str(node)))
        group = data.get("group", "meta")
        node_color.append(group_color.get(group, "blue"))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2)
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Metagenome Hierarchy at Generation {ga.encoding_manager.current_generation}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

def run_dashboard():
    app.run_server(debug=False, port=8050)

dashboard_thread = Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# -------------------------------
# Run the GA Algorithm
# -------------------------------
ga.run_algorithm()

# -------------------------------
# After the GA completes, print the best solution
# -------------------------------
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution:', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('Length of best genome:', len(best_genome))
print(f"Best Genome (Encoded): {best_genome}")
