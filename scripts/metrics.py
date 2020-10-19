#!/usr/bin/env python
from __future__ import unicode_literals, print_function, division

"""
Command-line script to generate network metrics in JSON

History:
- 5.1: Using the prov_to_graph method from the prov package instead of graph_to_prov
- 5.0: Added mfd_der and --flat command line argument
- 4.0: Calculating number of entities, agents, and activities
- 3.2: Including lone nodes (without any relation) when generating graphs from PROV.
- 3.1: Flattening documents with bundles (then unifying) before calculating metrics.

@author: Trung Dong Huynh <trungdong@donggiang.com>
"""
from collections import defaultdict, Counter
import itertools
import json
import math
import networkx as nx
import numpy as np
import powerlaw
from prov.graph import prov_to_graph
from prov.model import (
    ProvDocument,
    ProvEntity,
    ProvActivity,
    ProvAgent,
    ProvGeneration,
    ProvUsage,
    ProvDerivation,
    ProvAttribution,
    ProvElement,
)


node_select = lambda g, t: [node for node in g.nodes() if isinstance(node, t)]


def graph_select(graph, start_node_types=None, end_node_types=None, edge_types=None):
    new_graph = nx.MultiDiGraph()
    for edge in graph.edges(data=True):
        start_node, end_node, data = edge
        if start_node_types is not None and not isinstance(
            start_node, start_node_types
        ):
            continue
        if end_node_types is not None and not isinstance(end_node, end_node_types):
            continue
        if edge_types is not None and not isinstance(data["relation"], edge_types):
            continue
        new_graph.add_edge(start_node, end_node, attr_dict=data)
    return new_graph


def paths_select(graph, start_node_types, end_node_types, edge_types=None):
    g = graph_select(graph, edge_types=edge_types) if edge_types is not None else graph
    sp = nx.shortest_path(g)
    paths = defaultdict(dict)
    for i in sp:
        if isinstance(i, start_node_types):
            for j in sp[i]:
                if i != j and isinstance(j, end_node_types):
                    paths[i][j] = sp[i][j]
    return paths


def version1(prov_doc):
    results = dict()

    g = prov_to_graph(prov_doc)
    # Graph size
    results["nodes"] = g.number_of_nodes()
    results["edges"] = g.size()

    ug = nx.Graph(g)
    n_comps = nx.number_connected_components(ug)
    results["components"] = n_comps
    results["diameter"] = nx.diameter(nx.Graph(ug)) if n_comps == 1 else -1

    s_paths = nx.shortest_path(g)
    lengths = lambda g, t1, t2: [
        (len(s_paths[i][j]) - 1)
        for i in node_select(g, t1)
        if i in s_paths
        for j in node_select(g, t2)
        if j in s_paths[i] and i != j
    ]

    def mfd(graph, t1, t2):
        s_distances = lengths(graph, t1, t2)
        return max(s_distances) if s_distances else 0

    results["mfd"] = {
        "entity": {
            "entity": mfd(g, ProvEntity, ProvEntity),
            "activity": mfd(g, ProvEntity, ProvActivity),
            "agent": mfd(g, ProvEntity, ProvAgent),
        },
        "activity": {
            "entity": mfd(g, ProvActivity, ProvEntity),
            "activity": mfd(g, ProvActivity, ProvActivity),
            "agent": mfd(g, ProvActivity, ProvAgent),
        },
        "agent": {
            "entity": mfd(g, ProvAgent, ProvEntity),
            "activity": mfd(g, ProvAgent, ProvActivity),
            "agent": mfd(g, ProvAgent, ProvAgent),
        },
    }

    distributions = dict()
    der_paths = paths_select(g, ProvEntity, ProvEntity, ProvDerivation)
    der_lengths = [(len(der_paths[i][j]) - 1) for i in der_paths for j in der_paths[i]]
    distributions["derivations"] = der_lengths

    aee_paths = paths_select(g, ProvActivity, ProvEntity, (ProvDerivation, ProvUsage))
    aee_lengths = [(len(aee_paths[i][j]) - 1) for i in aee_paths for j in aee_paths[i]]
    distributions["activity_entities"] = aee_lengths

    eeag_paths = paths_select(
        g, ProvEntity, ProvAgent, (ProvDerivation, ProvGeneration)
    )
    eeag_lengths = [
        (len(eeag_paths[i][j]) - 1) for i in eeag_paths for j in eeag_paths[i]
    ]
    distributions["entities_agent"] = eeag_lengths

    results["distributions"] = distributions

    return results


def version2(prov_doc):
    results = dict()

    g = prov_to_graph(prov_doc) if isinstance(prov_doc, ProvDocument) else prov_doc
    # Graph size
    results["nodes"] = g.number_of_nodes()
    results["edges"] = g.size()

    ug = nx.Graph(g)
    n_comps = nx.number_connected_components(ug)
    results["components"] = n_comps
    results["diameter"] = nx.diameter(nx.Graph(ug)) if n_comps == 1 else -1

    s_paths = nx.shortest_path(g)
    lengths = lambda g, t1, t2: [
        (len(s_paths[i][j]) - 1)
        for i in node_select(g, t1)
        if i in s_paths
        for j in node_select(g, t2)
        if j in s_paths[i] and i != j
    ]

    def mfd(graph, t1, t2):
        s_distances = lengths(graph, t1, t2)
        return max(s_distances) if s_distances else 0

    results["mfd"] = {
        "entity": {
            "entity": mfd(g, ProvEntity, ProvEntity),
            "activity": mfd(g, ProvEntity, ProvActivity),
            "agent": mfd(g, ProvEntity, ProvAgent),
        },
        "activity": {
            "entity": mfd(g, ProvActivity, ProvEntity),
            "activity": mfd(g, ProvActivity, ProvActivity),
            "agent": mfd(g, ProvActivity, ProvAgent),
        },
        "agent": {
            "entity": mfd(g, ProvAgent, ProvEntity),
            "activity": mfd(g, ProvAgent, ProvActivity),
            "agent": mfd(g, ProvAgent, ProvAgent),
        },
    }

    distributions = dict()
    # Path length distributions of derivations
    der_paths = paths_select(g, ProvEntity, ProvEntity, ProvDerivation)
    der_lengths = [(len(der_paths[i][j]) - 1) for i in der_paths for j in der_paths[i]]
    distributions["derivations"] = der_lengths

    # Path length distributions of usages
    aee_paths = paths_select(g, ProvActivity, ProvEntity, (ProvDerivation, ProvUsage))
    aee_lengths = [(len(aee_paths[i][j]) - 1) for i in aee_paths for j in aee_paths[i]]
    distributions["activity_entities"] = aee_lengths

    # Path length distributions of attributions
    eeag_paths = paths_select(
        g, ProvEntity, ProvAgent, (ProvDerivation, ProvAttribution)
    )
    eeag_lengths = [
        (len(eeag_paths[i][j]) - 1) for i in eeag_paths for j in eeag_paths[i]
    ]
    distributions["entities_agent"] = eeag_lengths

    # Node degree distribution (undirected)
    distributions["node_degrees"] = dict(ug.degree()).values()

    results["distributions"] = distributions

    # The power law exponent of node degrees
    power_law_fit = powerlaw.Fit(
        distributions["node_degrees"], discrete=True, verbose=False
    )

    if not math.isnan(power_law_fit.alpha):
        # Check if the distribution is likely to be following the power law
        R, p = power_law_fit.distribution_compare("power_law", "exponential")
        if R > 0 and p < 0.05:
            # print power_law_fit.alpha, power_law_fit.sigma, R, p
            results["node_degrees_powerlaw"] = {
                "alpha": power_law_fit.alpha,
                "sigma": power_law_fit.sigma,
            }
    return results


v3_metric_names = [
    "nodes",
    "edges",
    "components",
    "diameter",
    "assortativity",  # standard metrics
    "acc",
    "acc_e",
    "acc_a",
    "acc_ag",  # average clustering coefficients
    "mfd_e_e",
    "mfd_e_a",
    "mfd_e_ag",  # MFDs
    "mfd_a_e",
    "mfd_a_a",
    "mfd_a_ag",
    "mfd_ag_e",
    "mfd_ag_a",
    "mfd_ag_ag",
    "powerlaw_alpha",
    "powerlaw_sigma",  # Power Law
    "node_degrees",
    "derivations",
    "activity_entities",
    "entities_agent",  # distributions
]


def flatten(mv3):
    return (
        mv3["nodes"],  # number of nodes
        mv3["edges"],  # number of edges
        mv3["components"],  # number of connected components
        mv3["diameter"],
        mv3["degree_assortativity_coefficient"],
        mv3["average_clustering_coefficient"]["all"],
        mv3["average_clustering_coefficient"]["entity"],
        mv3["average_clustering_coefficient"]["activity"],
        mv3["average_clustering_coefficient"]["agent"],
        mv3["mfd"]["entity"]["entity"],
        mv3["mfd"]["entity"]["activity"],
        mv3["mfd"]["entity"]["agent"],
        mv3["mfd"]["activity"]["entity"],
        mv3["mfd"]["activity"]["activity"],
        mv3["mfd"]["activity"]["agent"],
        mv3["mfd"]["agent"]["entity"],
        mv3["mfd"]["agent"]["activity"],
        mv3["mfd"]["agent"]["agent"],
        mv3["node_degrees_powerlaw"]["alpha"] if "node_degrees_powerlaw" in mv3 else -1,
        mv3["node_degrees_powerlaw"]["sigma"] if "node_degrees_powerlaw" in mv3 else -1,
        mv3["distributions"]["node_degrees"],  # node degrees distribution
        mv3["distributions"]["derivations"],  # finite shortest distances for all wdf*
        mv3["distributions"][
            "activity_entity"
        ],  # finite shortest distances for all a-(u|wdf)*->e
        mv3["distributions"][
            "entity_agent"
        ],  # finite shortest distances for all e-(wdf|wat)*->ag
    )


def version3(prov_doc, flat=False):
    results = dict()

    if isinstance(prov_doc, ProvDocument):
        g = prov_to_graph(prov_doc)
    else:
        # Assuming we got a NetworkX graph already
        g = prov_doc

    # Graph size
    results["nodes"] = g.number_of_nodes()
    results["edges"] = g.size()

    ug = nx.Graph(g)
    n_comps = nx.number_connected_components(ug)
    results["components"] = n_comps
    results["diameter"] = nx.diameter(nx.Graph(ug)) if n_comps == 1 else -1

    # Clustering coefficients for all nodes
    # cc = nx.clustering(ug)
    cc = dict(
        (n, e) for n, e in nx.clustering(ug).items() if e
    )  # excluding zero values
    cc_by_type = lambda node_type: [cc[n] for n in cc if isinstance(n, node_type)]
    avg_or_0 = lambda l: sum(l) / len(l) if l else 0

    results["average_clustering_coefficient"] = {
        "all": avg_or_0(cc.values()),
        "entity": avg_or_0(cc_by_type(ProvEntity)),
        "activity": avg_or_0(cc_by_type(ProvActivity)),
        "agent": avg_or_0(cc_by_type(ProvAgent)),
    }

    try:
        assortability = nx.degree_assortativity_coefficient(g)
    except ValueError:
        assortability = -1
    results["degree_assortativity_coefficient"] = (
        -1 if math.isnan(assortability) else assortability
    )

    s_paths = nx.shortest_path(g)
    lengths = lambda g, t1, t2: [
        (len(s_paths[i][j]) - 1)
        for i in node_select(g, t1)
        if i in s_paths
        for j in node_select(g, t2)
        if j in s_paths[i] and i != j
    ]

    def mfd(graph, t1, t2):
        s_distances = lengths(graph, t1, t2)
        return max(s_distances) if s_distances else 0

    results["mfd"] = {
        "entity": {
            "entity": mfd(g, ProvEntity, ProvEntity),
            "activity": mfd(g, ProvEntity, ProvActivity),
            "agent": mfd(g, ProvEntity, ProvAgent),
        },
        "activity": {
            "entity": mfd(g, ProvActivity, ProvEntity),
            "activity": mfd(g, ProvActivity, ProvActivity),
            "agent": mfd(g, ProvActivity, ProvAgent),
        },
        "agent": {
            "entity": mfd(g, ProvAgent, ProvEntity),
            "activity": mfd(g, ProvAgent, ProvActivity),
            "agent": mfd(g, ProvAgent, ProvAgent),
        },
    }

    distributions = dict()
    # Path length distributions of derivations
    der_paths = paths_select(g, ProvEntity, ProvEntity, ProvDerivation)
    der_lengths = [(len(der_paths[i][j]) - 1) for i in der_paths for j in der_paths[i]]
    distributions["derivations"] = der_lengths

    # Path length distributions of usages
    aee_paths = paths_select(g, ProvActivity, ProvEntity, (ProvDerivation, ProvUsage))
    aee_lengths = [(len(aee_paths[i][j]) - 1) for i in aee_paths for j in aee_paths[i]]
    distributions["activity_entity"] = aee_lengths

    # Path length distributions of attributions
    eeag_paths = paths_select(
        g, ProvEntity, ProvAgent, (ProvDerivation, ProvAttribution)
    )
    eeag_lengths = [
        (len(eeag_paths[i][j]) - 1) for i in eeag_paths for j in eeag_paths[i]
    ]
    distributions["entity_agent"] = eeag_lengths

    # Node degree distribution (undirected)
    distributions["node_degrees"] = list(dict(ug.degree()).values())

    results["distributions"] = distributions

    # The power law exponent of node degrees
    power_law_fit = powerlaw.Fit(
        distributions["node_degrees"], discrete=True, verbose=False
    )

    if not math.isnan(power_law_fit.alpha):
        # Check if the distribution is likely to be following the power law
        R, p = power_law_fit.distribution_compare("power_law", "exponential")
        if R > 0 and p < 0.05:
            results["node_degrees_powerlaw"] = {
                "alpha": power_law_fit.alpha,
                "sigma": power_law_fit.sigma,
            }

    if not flat:
        return results
    else:
        return flatten(results)


version3.__version__ = "3.2"


def flatten_v4(mv4):
    return [
        mv4["entities"],
        mv4["agents"],
        mv4["activities"],
        mv4["nodes"],  # number of nodes
        mv4["edges"],  # number of edges
        mv4["components"],  # number of connected components
        mv4["diameter"],
        mv4["degree_assortativity_coefficient"],
        mv4["average_clustering_coefficient"]["all"],
        mv4["average_clustering_coefficient"]["entity"],
        mv4["average_clustering_coefficient"]["activity"],
        mv4["average_clustering_coefficient"]["agent"],
        mv4["mfd"]["entity"]["entity"],
        mv4["mfd"]["entity"]["activity"],
        mv4["mfd"]["entity"]["agent"],
        mv4["mfd"]["activity"]["entity"],
        mv4["mfd"]["activity"]["activity"],
        mv4["mfd"]["activity"]["agent"],
        mv4["mfd"]["agent"]["entity"],
        mv4["mfd"]["agent"]["activity"],
        mv4["mfd"]["agent"]["agent"],
        mv4["node_degrees_powerlaw"]["alpha"] if "node_degrees_powerlaw" in mv4 else -1,
        mv4["node_degrees_powerlaw"]["sigma"] if "node_degrees_powerlaw" in mv4 else -1,
        mv4["distributions"]["node_degrees"],  # node degrees distribution
        mv4["distributions"]["derivations"],  # finite shortest distances for all wdf*
        mv4["distributions"][
            "activity_entity"
        ],  # finite shortest distances for all a-(u|wdf)*->e
        mv4["distributions"][
            "entity_agent"
        ],  # finite shortest distances for all e-(wdf|wat)*->ag
    ]


def version4(prov_doc, flat=False):
    results = dict()

    if isinstance(prov_doc, ProvDocument):
        g = prov_to_graph(prov_doc)
    else:
        # Assuming we got a NetworkX graph already
        g = prov_doc

    # PROV types
    type_counter = defaultdict(int, Counter(map(type, g.nodes())))
    results["entities"] = type_counter[ProvEntity]
    results["agents"] = type_counter[ProvAgent]
    results["activities"] = type_counter[ProvActivity]

    # Graph size
    results["nodes"] = g.number_of_nodes()
    results["edges"] = g.size()

    ug = g.to_undirected(as_view=True)
    n_comps = nx.number_connected_components(ug)
    results["components"] = n_comps
    results["diameter"] = nx.diameter(ug) if n_comps == 1 else -1

    # Clustering coefficients for all nodes
    # cc = nx.clustering(ug)
    cc = dict(
        (n, e) for n, e in nx.clustering(nx.Graph(ug)).items() if e
    )  # excluding zero values
    cc_by_type = lambda node_type: [cc[n] for n in cc if isinstance(n, node_type)]
    avg_or_0 = lambda l: sum(l) / len(l) if l else 0

    results["average_clustering_coefficient"] = {
        "all": avg_or_0(cc.values()),
        "entity": avg_or_0(cc_by_type(ProvEntity)),
        "activity": avg_or_0(cc_by_type(ProvActivity)),
        "agent": avg_or_0(cc_by_type(ProvAgent)),
    }

    try:
        assortability = nx.degree_pearson_correlation_coefficient(g)
    except ValueError:
        assortability = -1
    results["degree_assortativity_coefficient"] = (
        assortability if np.isfinite(assortability) else -1
    )

    s_paths = nx.shortest_path(g)
    lengths = lambda g, t1, t2: [
        (len(s_paths[i][j]) - 1)
        for i in node_select(g, t1)
        if i in s_paths
        for j in node_select(g, t2)
        if j in s_paths[i] and i != j
    ]

    def mfd(graph, t1, t2):
        s_distances = lengths(graph, t1, t2)
        return max(s_distances) if s_distances else 0

    results["mfd"] = {
        "entity": {
            "entity": mfd(g, ProvEntity, ProvEntity),
            "activity": mfd(g, ProvEntity, ProvActivity),
            "agent": mfd(g, ProvEntity, ProvAgent),
        },
        "activity": {
            "entity": mfd(g, ProvActivity, ProvEntity),
            "activity": mfd(g, ProvActivity, ProvActivity),
            "agent": mfd(g, ProvActivity, ProvAgent),
        },
        "agent": {
            "entity": mfd(g, ProvAgent, ProvEntity),
            "activity": mfd(g, ProvAgent, ProvActivity),
            "agent": mfd(g, ProvAgent, ProvAgent),
        },
    }

    distributions = dict()
    # Path length distributions of derivations
    der_paths = paths_select(g, ProvEntity, ProvEntity, ProvDerivation)
    der_lengths = [(len(der_paths[i][j]) - 1) for i in der_paths for j in der_paths[i]]
    distributions["derivations"] = der_lengths

    # Path length distributions of usages
    aee_paths = paths_select(g, ProvActivity, ProvEntity, (ProvDerivation, ProvUsage))
    aee_lengths = [(len(aee_paths[i][j]) - 1) for i in aee_paths for j in aee_paths[i]]
    distributions["activity_entity"] = aee_lengths

    # Path length distributions of attributions
    eeag_paths = paths_select(
        g, ProvEntity, ProvAgent, (ProvDerivation, ProvAttribution)
    )
    eeag_lengths = [
        (len(eeag_paths[i][j]) - 1) for i in eeag_paths for j in eeag_paths[i]
    ]
    distributions["entity_agent"] = eeag_lengths

    # Node degree distribution (undirected)
    distributions["node_degrees"] = list(dict(ug.degree()).values())

    results["distributions"] = distributions

    # The power law exponent of node degrees
    power_law_fit = powerlaw.Fit(
        distributions["node_degrees"], discrete=True, verbose=False
    )

    if not math.isnan(power_law_fit.alpha):
        # Check if the distribution is likely to be following the power law
        R, p = power_law_fit.distribution_compare("power_law", "exponential")
        if R > 0 and p < 0.05:
            results["node_degrees_powerlaw"] = {
                "alpha": power_law_fit.alpha,
                "sigma": power_law_fit.sigma,
            }

    if not flat:
        return results
    else:
        return flatten_v4(results)


version4.__version__ = "4.0"
version4.metrics_names = (
    "entities",
    "agents",
    "activities",  # PROV types (for nodes)
    "nodes",
    "edges",
    "components",
    "diameter",
    "assortativity",  # standard metrics
    "acc",
    "acc_e",
    "acc_a",
    "acc_ag",  # average clustering coefficients
    "mfd_e_e",
    "mfd_e_a",
    "mfd_e_ag",  # MFDs
    "mfd_a_e",
    "mfd_a_a",
    "mfd_a_ag",
    "mfd_ag_e",
    "mfd_ag_a",
    "mfd_ag_ag",
    "powerlaw_alpha",
    "powerlaw_sigma",  # Power Law
    "node_degrees",
    "derivations",
    "activity_entities",
    "entities_agent",  # distributions
)


def ensure_prov_networkx_graph(prov_doc):
    if isinstance(prov_doc, ProvDocument):
        g = prov_to_graph(prov_doc)
    else:
        # Assuming we got a NetworkX graph already
        # TODO Raise an exception when this is not the case
        g = prov_doc
    return g


def mfd_derivations(g):
    paths_dict = paths_select(g, ProvElement, ProvElement, ProvDerivation).values()
    if not paths_dict:  # nothing found
        return -1
    shortest_derivation_paths = itertools.chain.from_iterable(
        m.values() for m in paths_dict
    )
    return max(map(len, shortest_derivation_paths)) - 1


def mv4_to_mv5(mv4, prov_doc):
    g = ensure_prov_networkx_graph(prov_doc)
    mfd_der = mfd_derivations(g)
    mv4["mfd"]["derivations"] = mfd_der
    return mv4


def flatten_v5(mv5):
    flat_mv4 = flatten_v4(mv5)
    flat_mv5 = flat_mv4[:-6] + [mv5["mfd"]["derivations"]] + flat_mv4[-6:]
    return flat_mv5


def version5(prov_doc, flat=False):
    mv4 = version4(prov_doc)
    mv5 = mv4_to_mv5(mv4, prov_doc)
    if not flat:
        return mv5
    else:
        flat_mv5 = flatten_v5(mv5)
        return flat_mv5


version5.__version__ = "5.0"
version5.metrics_names = (
    "entities",
    "agents",
    "activities",  # PROV types (for nodes)
    "nodes",
    "edges",
    "components",
    "diameter",
    "assortativity",  # standard metrics
    "acc",
    "acc_e",
    "acc_a",
    "acc_ag",  # average clustering coefficients
    "mfd_e_e",
    "mfd_e_a",
    "mfd_e_ag",  # MFDs
    "mfd_a_e",
    "mfd_a_a",
    "mfd_a_ag",
    "mfd_ag_e",
    "mfd_ag_a",
    "mfd_ag_ag",
    "mfd_der",  # MFD derivations
    "powerlaw_alpha",
    "powerlaw_sigma",  # Power Law
    "node_degrees",
    "derivations",
    "activity_entities",
    "entities_agent",  # distributions
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate provenance network metrics")
    parser.add_argument("input", help="the input PROV-JSON file")
    parser.add_argument(
        "output",
        nargs="?",
        help="the output file to store the calculated metrics; "
        "if missing, the result will be output to the console",
    )
    parser.add_argument(
        "-i",
        "--indent",
        help="pretty-print the JSON output with indentation",
        action="store_true",
    )
    parser.add_argument(
        "-f", "--flat", help="produce a flat list of metrics", action="store_true"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        prov_document = ProvDocument.deserialize(f)

        stats = version5(prov_document, args.flat)
        result = {
            "version": version5.__version__,
            "input": args.input,
            "metrics": stats,
        }
        indent = 2 if args.indent else None
        if args.output:
            with open(args.output, "w") as o:
                json.dump(result, o, indent=indent)
        else:
            print(json.dumps(result, indent=indent))
