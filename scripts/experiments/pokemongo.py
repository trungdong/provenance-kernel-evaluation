"""Common code for Pokemon Go experiments.

All Pokemon Go experiments are the same, just on a different dataset.
We're using the common experiment code from this module.
"""
from pathlib import Path

from scripts.data.common import NETWORK_METRIC_NAMES
from scripts.data.pokemongo import POKEMON_GO_DATA_COLUMNS
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

    print(f"> Testing predicting the team labels for {dataset_id}...")

    # Balancing the dataset on the trusted attribute
    print(
        "Current number of players in each team:\n", graphs_index.label.value_counts()
    )
    selected_graphs = graphs_index
    selected_graphs.graph_file.to_csv(selected_samples_filepath)

    print(f"Generating GraKeL graphs for {len(selected_graphs)} files")
    selected_graphs = build_grakel_graphs(selected_graphs, dataset_folder)

    cv_sets = get_fixed_CV_sets(selected_graphs, selected_graphs.label)
    print(f"Generated {len(cv_sets)} cross-validation train/test sets.")

    results = test_prediction_on_classifiers(
        selected_graphs[POKEMON_GO_DATA_COLUMNS], selected_graphs.label, cv_sets
    )

    pna_results = test_prediction_on_classifiers(
        selected_graphs[NETWORK_METRIC_NAMES],
        selected_graphs.label,
        cv_sets,
        test_prefix="PNA-",
    )
    pna_results["time"] = selected_graphs.timings_PNA.sum()
    results = results.append(pna_results, ignore_index=True)

    results = results.append(
        test_prediction_on_Grakel_kernels(
            selected_graphs, "label", cv_sets, ignore_kernels={"GK-GSamp"}
        ),
        ignore_index=True,
    )

    results = results.append(
        test_prediction_on_kernels(selected_graphs, outputs_folder, "label", cv_sets),
        ignore_index=True,
    )

    print("Saving scoring to:", output_filepath)
    results.to_pickle(output_filepath)
