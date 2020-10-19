"""Common code for preparing the graph index of CollabMap datasets."""
import logging
from pathlib import Path

import pandas as pd

from .common import calculate_provenance_network_metrics

logger = logging.getLogger(__name__)


def copy_graph_index(dataset_path, output_path):
    logger.debug("Working in folder: %s", dataset_path)
    dataset_path = Path(dataset_path)

    graph_index_filepath = dataset_path / "graphs.csv"
    if not graph_index_filepath.exists():
        logger.error("Graphs index file is not found: %s", graph_index_filepath)
        exit(1)

    logger.debug("Reading graphs index...")
    graphs = pd.read_csv(graph_index_filepath)

    logger.debug("Calculating provenance network metrics for %d graphs...", len(graphs))
    metrics = calculate_provenance_network_metrics(dataset_path, graphs)
    graphs = graphs.join(metrics)

    graphs["trusted"] = graphs.label == "Trusted"
    graphs.drop("label", axis="columns", inplace=True)
    output_filepath = Path(output_path) / "graphs.pickled"
    graphs.to_pickle(output_filepath)
