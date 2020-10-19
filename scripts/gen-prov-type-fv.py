"""Generate feature vectors for provenance types found by provenance kernels."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import click


logger = logging.getLogger(__name__)


def load_feature_file(
    dataset_folder: Path, kernel_set, graph_filename, level: int
) -> dict:
    graph_filepath = dataset_folder / graph_filename
    graph_id = graph_filepath.stem
    features_filepath = (
        dataset_folder / kernel_set / f"{graph_id}-{level}.features.json"
    )

    with features_filepath.open() as f:
        dict_structure = json.load(f)
        features = dict_structure[
            "features"
        ]  # expecting a 'features' map in the JSON file
        return features


@click.command()
@click.argument("dataset_folder")
@click.argument("kernel_set", default="summary")
@click.argument("level", default=0)
@click.option("-c", "--check-mappings", default=False, is_flag=True)
def gen_prov_type_features(dataset_folder, kernel_set, level, check_mappings):
    logger.info("Working in folder: %s", dataset_folder)
    dataset_folder = Path(dataset_folder)

    graph_index_filepath = dataset_folder / "graphs.pickled"
    if not graph_index_filepath.exists():
        logger.error("Graphs index file is not found: %s", graph_index_filepath)
        exit(1)

    logger.info("Reading graphs index")
    graphs = pd.read_pickle(graph_index_filepath)

    logger.info("Reading PROV type features from %s level %s", kernel_set, level)
    kernels = []
    for graph_filename in graphs.graph_file:
        kernels.append(
            load_feature_file(dataset_folder, kernel_set, graph_filename, level)
        )

    # Builing the sparse matrix  for the feature vectors
    indptr = [0]
    indices = []
    data = []
    vocabulary = dict()
    for feature in kernels:
        for prov_type, count in feature.items():
            index = vocabulary.setdefault(prov_type, len(vocabulary))
            indices.append(index)
            data.append(count)
        indptr.append(len(indices))
    csr_m = csr_matrix((data, indices, indptr), dtype=int)

    logger.info("Seen %d unique PROV types.", len(vocabulary))

    logger.info("Mapping PROV types to feature vector")
    prov_types_df = pd.DataFrame(vocabulary.items(), columns=["Type", "Number"])
    prov_types_df = prov_types_df[["Number", "Type"]]
    prov_types_df.Number += 1
    # making sure the order of the types are the same (asc. order) as in the column index
    prov_types_df.sort_values("Number", inplace=True)

    filepath = dataset_folder / f"{kernel_set}_{level}_prov_types.csv"
    logger.info("Writing PROV type mappings to: %s", filepath)
    prov_types_df.to_csv(filepath, index=False)

    prov_type_features = pd.DataFrame.sparse.from_spmatrix(
        data=csr_m, index=graphs.graph_file, columns=prov_types_df.Number
    )
    logger.info(prov_type_features.info())

    if check_mappings:
        # Verifying that the mapping is correct
        logger.info("Verifying that the type mappings are correct...")
        type_mappings = {row.Number: row.Type for row in prov_types_df.itertuples()}
        kernel_mapping = dict(zip(graphs.graph_file, kernels))
        for graph_filename, feature in prov_type_features.iterrows():
            # unpack the vector back to a kernel
            repacked_kernel = {
                type_mappings[type_index]: value
                for type_index, value in feature.items()
                if value > 0
            }
            if repacked_kernel != kernel_mapping[graph_filename]:
                logger.error("PROV type mappings for %s incorrect!", graph_filename)
                return -1  # fail here
        logger.info("All correct.")

    filepath = dataset_folder / f"{kernel_set}_{level}_prov_type_features.csv.bz2"
    logger.info("Writing PROV type features to: %s", filepath)
    prov_type_features.to_csv(filepath)
    logger.info("All done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    gen_prov_type_features()
