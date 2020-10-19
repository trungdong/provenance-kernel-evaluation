"""Common code for CollabMap experiments.

All CollabMap experiments are the same, just on a different dataset.
We're using the common experiment code from this module.
"""
from pathlib import Path

import pandas as pd

from scripts.data.common import NETWORK_METRIC_NAMES
from scripts.graphkernels import build_grakel_graphs
from .common import (
    get_fixed_CV_sets,
    test_prediction_on_classifiers,
    test_prediction_on_kernels,
    test_prediction_on_Grakel_kernels,
)
from scripts.utils import load_graph_index


def run_experiment(dataset_id: str):
    ROOT_DIR = Path(__file__).parents[2]

    dataset_folder = ROOT_DIR / "datasets" / dataset_id
    outputs_folder = ROOT_DIR / "outputs" / dataset_id
    output_filepath = outputs_folder / "scoring.pickled"
    selected_samples_filepath = outputs_folder / "selected.csv"

    graphs_index = load_graph_index(dataset_id)

    print(f"> Testing predicting the data quality labels for {dataset_id}...")

    # Balancing the dataset on the trusted attribute
    print("Current number of trusted values:\n", graphs_index.trusted.value_counts())
    selected_graphs = graphs_index[graphs_index.trusted == False]
    selected_graphs = selected_graphs.append(
        graphs_index[graphs_index.trusted == True].sample(len(selected_graphs))
    )
    print(
        "Number of trusted values in selected graphs:\n",
        selected_graphs.trusted.value_counts(),
    )
    selected_graphs.graph_file.to_csv(selected_samples_filepath)

    print(f"Generating GraKeL graphs for {len(selected_graphs)} files")
    selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)

    # ---------- Ayah's code ----------------------------------------------
    selected_graphs_filepath = outputs_folder / f"{dataset_id}.pickled"
    selected_graphs_filepath_2 = outputs_folder / f"{dataset_id}-Analytics.pickled"
    index = selected_graphs.columns.get_loc(NETWORK_METRIC_NAMES[0])
    applicationdata = selected_graphs.columns[:index]

    X = selected_graphs[applicationdata]
    y = selected_graphs.trusted

    data = pd.concat([X, y], axis=1)
    data = data.rename(columns={"trusted": "Class"})
    data.to_pickle(selected_graphs_filepath)

    X = selected_graphs[NETWORK_METRIC_NAMES]
    y = selected_graphs.trusted

    data = pd.concat([X, y], axis=1)
    data = data.rename(columns={"trusted": "Class"})
    data.to_pickle(selected_graphs_filepath_2)
    # --------------------------------------------------------------------

    cv_sets = get_fixed_CV_sets(selected_graphs, selected_graphs.trusted)
    print(f"Generated {len(cv_sets)} cross-validation train/test sets.")

    results = test_prediction_on_classifiers(
        selected_graphs[NETWORK_METRIC_NAMES],
        selected_graphs.trusted,
        cv_sets,
        test_prefix="PNA-",
    )
    results["time"] = selected_graphs.timings_PNA.sum()

    results = results.append(
        test_prediction_on_Grakel_kernels(selected_graphs, "trusted", cv_sets),
        ignore_index=True,
    )

    results = results.append(
        test_prediction_on_kernels(selected_graphs, outputs_folder, "trusted", cv_sets),
        ignore_index=True,
    )

    print("Saving scoring to:", output_filepath)
    results.to_pickle(output_filepath)
