"""Experiment to test prediction on dead values."""
from pathlib import Path

import pandas as pd

from scripts.data.common import NETWORK_METRIC_NAMES
from scripts.experiments.common import test_prediction_with_gnn
from scripts.graphkernels import build_grakel_graphs
from .common import (
    get_fixed_CV_sets,
    test_prediction_on_ml_classifiers,
    test_prediction_with_provenance_kernels,
    test_prediction_with_generic_graph_kernels,
)
from scripts.utils import load_graph_index

ROOT_DIR = Path(__file__).parents[2]

data_folder = ROOT_DIR / "datasets"
dataset_id = "MIMIC-PXC7"
dataset_folder = ROOT_DIR / "datasets" / dataset_id
outputs_folder = ROOT_DIR / "outputs" / dataset_id
output_filepath = outputs_folder / "scoring.pickled"
selected_samples_filepath = outputs_folder / "selected.csv"

graphs_index = load_graph_index(dataset_id)

print("> Testing predicting a patient is dead at the end of this admission")

if selected_samples_filepath.exists():
    # Loading the previously saved balanced dataset to reproduce the same experiment
    selected_graphfiles = pd.read_csv(selected_samples_filepath, index_col=0)
    selected_graphs = graphs_index.iloc[selected_graphfiles.index].copy()
else:
    # This is the first time we run this experiment
    # Selecting relevant graphs and balancing the dataset
    print(" - Current number of dead values:\n", graphs_index.dead.value_counts())
    graphs_with_true_labels = graphs_index[graphs_index.dead == True]
    selected_graphs = pd.concat(
        [
            graphs_with_true_labels,
            graphs_index[graphs_index.dead == False].sample(len(graphs_with_true_labels))
        ]
    )
    print(
        " - Number of dead values in selected graphs:\n",
        selected_graphs.dead.value_counts(),
    )
    # saving the list of selected graphs for later reproduction of this experiment
    selected_graphs.graph_file.to_csv(selected_samples_filepath)

cv_sets = get_fixed_CV_sets(
    selected_graphs, selected_graphs.dead, output_path=outputs_folder
)
print(f"> Got {len(cv_sets)} cross-validation train/test sets.")

scoring_pna = test_prediction_on_ml_classifiers(
    selected_graphs[NETWORK_METRIC_NAMES],
    outputs_folder,
    selected_graphs.dead,
    cv_sets,
    test_prefix="PNA-",
)
scoring_pna["time"] = selected_graphs.timings_PNA.sum()

scorings = [scoring_pna]

scorings.append(
    test_prediction_with_provenance_kernels(selected_graphs, outputs_folder, "dead", cv_sets),
)

print(f"> Generating GraKeL graphs for {len(selected_graphs)} files")
selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)
scorings.append(
    test_prediction_with_generic_graph_kernels(selected_graphs, outputs_folder, "dead", cv_sets),
)

scorings.append(
    test_prediction_with_gnn(selected_graphs, "dead", dataset_folder, outputs_folder, cv_sets)
)

print("> Saving scoring to:", output_filepath)
results = pd.concat(scorings, ignore_index=True)
results.to_pickle(output_filepath)
