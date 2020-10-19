"""Functions to work with the GraKeL library."""
from pathlib import Path
from prov.model import ProvDocument
from prov.graph import prov_to_graph
import networkx as nx
import pandas as pd


def graph_from_prov_networkx_graph(prov_graph: nx.MultiDiGraph):
    graph_object = dict()
    node_labels = dict()
    edge_labels = dict()

    renamed_nodes = {
        node: i for i, node in enumerate(prov_graph.nodes)
    }

    for n, nbrsdict in prov_graph.adjacency():
        u = renamed_nodes[n]
        graph_object[u] = dict()
        node_labels[u] = str(n.get_type())
        for nbr, keydict in nbrsdict.items():
            v = renamed_nodes[nbr]
            for key, eattr in keydict.items():
                if v not in graph_object[u]:
                    # only add the first edge
                    graph_object[u][v] = 1.0
                    edge_labels[(u, v)] = str(eattr["relation"].get_type())

    return (graph_object, node_labels, edge_labels)


def build_grakel_graphs(graphs: pd.DataFrame, dataset_path: Path):
    if "grakel_graphs" in graphs.columns:
        # nothing to do
        return graphs  # unchanged

    # expecting a "graphfile" column in the input DataFrame
    grakel_graphs = []
    for graph_filename in graphs.graph_file:
        filepath = dataset_path / graph_filename
        # load the file
        prov_doc = ProvDocument.deserialize(filepath)
        prov_graph = prov_to_graph(prov_doc)  # type: nx.MultiDiGraph
        grakel_graphs.append(
            graph_from_prov_networkx_graph(prov_graph)
        )
    graphs["grakel_graphs"] = grakel_graphs
    return graphs
