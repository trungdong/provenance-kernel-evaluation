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

if selected_samples_filepath.exists():
    # Loading the previously saved balanced dataset to reproduce the same experiment
    selected_graphfiles = pd.read_csv(selected_samples_filepath, index_col=0)
    selected_graphs = graphs_index.iloc[selected_graphfiles.index].copy()
else:
    # This is the first time we run this experiment
    # Selecting relevant graphs and balancing the dataset
    print("Current number of dead values:\n", graphs_index.dead.value_counts())
    selected_graphs = graphs_index[graphs_index.dead == True]
    selected_graphs = selected_graphs.append(
        graphs_index[graphs_index.dead == False].sample(len(selected_graphs))
    )
    print(
        "Number of dead values in selected graphs:\n",
        selected_graphs.dead.value_counts(),
    )
    # saving the list of selected graphs for later reproduction of this experiment
    selected_graphs.graph_file.to_csv(selected_samples_filepath)

print(f"Generating GraKeL graphs for {len(selected_graphs)} files")
selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)

cv_sets = get_fixed_CV_sets(
    selected_graphs, selected_graphs.dead, output_path=outputs_folder
)
print(f"> Got {len(cv_sets)} cross-validation train/test sets.")

print(
    "--- Testing prediction using classic ML algorithms on provenance network metrics ---"
)
results = test_prediction_on_classifiers(
    selected_graphs[NETWORK_METRIC_NAMES],
    outputs_folder,
    selected_graphs.dead,
    cv_sets,
    test_prefix="PNA-",
)
results["time"] = selected_graphs.timings_PNA.sum()

print("--- Testing prediction using generic graph kernels ---")
results = results.append(
    test_prediction_on_Grakel_kernels(selected_graphs, outputs_folder, "dead", cv_sets),
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
