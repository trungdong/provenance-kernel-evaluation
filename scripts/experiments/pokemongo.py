"""Common code for Pokemon Go experiments.

All Pokemon Go experiments are the same, just on a different dataset.
We're using the common experiment code from this module.
"""
from pathlib import Path

import pandas as pd

from scripts.data.common import NETWORK_METRIC_NAMES
from scripts.graphkernels import build_grakel_graphs
from .common import (
    get_fixed_CV_sets,
    test_prediction_on_ml_classifiers,
    test_prediction_with_provenance_kernels,
    test_prediction_with_generic_graph_kernels,
    test_prediction_with_gnn,
)
from scripts.utils import load_graph_index


def run_experiment(dataset_id: str):
    ROOT_DIR = Path(__file__).parents[2]

    dataset_folder = ROOT_DIR / "datasets" / dataset_id
    outputs_folder = ROOT_DIR / "outputs" / dataset_id
    output_filepath = outputs_folder / "scoring.pickled"
    selected_samples_filepath = outputs_folder / "selected.csv"

    graphs_index = load_graph_index(dataset_id)

    print(f"> Testing predicting the team labels for {dataset_id}...")

    # This dataset is balanced and does not need balancing
    print(
        "  - Current number of players in each team:\n",
        graphs_index.label.value_counts(),
    )
    selected_graphs = graphs_index
    selected_graphs.graph_file.to_csv(selected_samples_filepath)

    cv_sets = get_fixed_CV_sets(
        selected_graphs, selected_graphs.label, output_path=outputs_folder
    )
    print(f"> Got {len(cv_sets)} cross-validation train/test sets.")

    scoring_pna = test_prediction_on_ml_classifiers(
        selected_graphs[NETWORK_METRIC_NAMES],
        outputs_folder,
        selected_graphs.label,
        cv_sets,
        test_prefix="PNA-",
    )
    scoring_pna["time"] = selected_graphs.timings_PNA.sum()

    scorings = [scoring_pna]

    scorings.append(
        test_prediction_with_provenance_kernels(selected_graphs, outputs_folder, "label", cv_sets)
    )

    print(f"> Generating GraKeL graphs for {len(selected_graphs)} files")
    selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)
    scorings.append(
        test_prediction_with_generic_graph_kernels(
            selected_graphs,
            outputs_folder,
            "label",
            cv_sets,
            ignore_kernels={"GK-GSamp"},
        )
    )

    scorings.append(
        test_prediction_with_gnn(selected_graphs, "label", dataset_folder, outputs_folder, cv_sets)
    )

    print("> Saving scoring to:", output_filepath)
    results = pd.concat(scorings, ignore_index=True)
    results.to_pickle(output_filepath)
