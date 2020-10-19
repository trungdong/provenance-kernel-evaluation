"""Experiment to test prediction on dead values."""
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

ROOT_DIR = Path(__file__).parents[2]

data_folder = ROOT_DIR / "datasets"
dataset_id = "MIMIC-PXC7"
dataset_folder = ROOT_DIR / "datasets" / dataset_id
outputs_folder = ROOT_DIR / "outputs" / dataset_id
output_filepath = outputs_folder / "scoring.pickled"
selected_samples_filepath = outputs_folder / "selected.csv"

graphs_index = load_graph_index(dataset_id)

print("> Testing predicting a patient is dead at the end of this admission")

# Selecting relevant graphs and balancing the dataset
print("Current number of dead values:\n", graphs_index.dead.value_counts())
selected_graphs = graphs_index[graphs_index.dead == True]
selected_graphs = selected_graphs.append(
    graphs_index[graphs_index.dead == False].sample(len(selected_graphs))
)
print(
    "Number of dead values in selected graphs:\n", selected_graphs.dead.value_counts()
)
selected_graphs.graph_file.to_csv(selected_samples_filepath)

print(f"Generating GraKeL graphs for {len(selected_graphs)} files")
selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)

cv_sets = get_fixed_CV_sets(selected_graphs, selected_graphs.dead)
print(f"Generated {len(cv_sets)} cross-validation train/test sets.")

print("--- Testing prediction using classic ML algorithms ---")
results = test_prediction_on_classifiers(
    selected_graphs[all_procedure_codes], selected_graphs.dead, cv_sets
)

print(
    "--- Testing prediction using classic ML algorithms on provenance network metrics ---"
)
pna_results = test_prediction_on_classifiers(
    selected_graphs[NETWORK_METRIC_NAMES],
    selected_graphs.dead,
    cv_sets,
    test_prefix="PNA-",
)
pna_results["time"] = selected_graphs.timings_PNA.sum()
results = results.append(pna_results, ignore_index=True)

print("--- Testing prediction using generic graph kernels ---")
results = results.append(
    test_prediction_on_Grakel_kernels(selected_graphs, "dead", cv_sets),
    ignore_index=True,
)

print("--- Testing prediction using provenance kernels ---")
results = results.append(
    test_prediction_on_kernels(selected_graphs, outputs_folder, "dead", cv_sets),
    ignore_index=True,
)

print("--- All done ---")
print("Saving scoring to:", output_filepath)
results.to_pickle(output_filepath)
