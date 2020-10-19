"""Generate feature vectors from flat provenance types for a set of graphs from level 0 to 5."""

import logging
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Tuple

import pandas as pd
from scipy.sparse import csr_matrix

from prov.model import ProvDocument

from flatprovenancetypes import calculate_flat_provenance_types, count_fp_types
from utils import Timer

import click


logger = logging.getLogger(__name__)


def count_flatprovenancetypes_for_graphs(
    dataset_path: Path,
    graph_filenames: Collection[str],
    level: int,
    including_primitives_types: bool,
) -> Tuple[List[Dict[int, Dict[str, int]]], List[List[float]]]:
    logger.debug(
        "Calculating flat provenance types up to level %s (with application types: %s) for %d graphs...",
        level,
        including_primitives_types,
        len(graph_filenames),
    )
    results = []  # type: List[Dict[int, Dict[str, int]]]
    timings = []  # type: List[List[float]]
    for graph_filename in graph_filenames:
        filepath = dataset_path / graph_filename
        prov_doc = ProvDocument.deserialize(filepath)
        durations = []  # type: List[float]
        features = dict()  # type: Dict[int, Dict[str, int]]
        for h in range(level + 1):
            timer = Timer(verbose=False)
            with timer:
                fp_types = calculate_flat_provenance_types(
                    prov_doc, h, including_primitives_types
                )
            # counting only the last level
            features[h] = count_fp_types(fp_types[h].values())
            durations.append(timer.interval)
        results.append(features)
        timings.append(durations)
    return results, timings


def save_single_level_table(
    feature_list: List[Dict[str, int]],
    index: Iterable[str],
    output_path: Path,
    kernel_set: str,
    level: int,
):
    logger.debug("Saving %s_%d features...", kernel_set, level)

    # Builing the sparse matrix  for the feature vectors
    indptr = [0]  # type: List[int]
    indices = []  # type: List[int]
    data = []  # type: List[int]
    vocabulary = dict()  # type: Dict[str, int]

    for feature in feature_list:
        for prov_type, count in feature.items():
            idx = vocabulary.setdefault(prov_type, len(vocabulary))
            indices.append(idx)
            data.append(count)
        indptr.append(len(indices))
    csr_m = csr_matrix((data, indices, indptr), dtype=int)

    logger.debug("- Seen %d unique types.", len(vocabulary))

    logger.debug("- Mapping PROV types to feature vector")
    prov_types_df = pd.DataFrame(vocabulary.items(), columns=["Type", "Number"])
    prov_types_df = prov_types_df[["Number", "Type"]]  # swapping the column order
    prov_types_df.Number += 1  # starting the numbering from 1
    # making sure the order of the types are the same (asc. order) as in the column index
    prov_types_df.sort_values("Number", inplace=True)
    # prefixing the feature name with the kernel type and level
    kernel_id = kernel_set + str(level)
    prov_types_df["Number"] = prov_types_df.Number.map(lambda n: f"{kernel_id}_{n}")

    filepath = output_path / "kernels" / f"{kernel_set}_{level}_types.csv"
    logger.debug("- Writing type mappings to: %s", filepath)
    prov_types_df.to_csv(filepath, index=False)

    prov_type_features = pd.DataFrame.sparse.from_spmatrix(
        data=csr_m, index=index, columns=prov_types_df.Number
    )
    logger.debug(prov_type_features.info())
    filepath = output_path / "kernels" / f"{kernel_set}_{level}.pickled"
    logger.debug("- Writing feature table to: %s", filepath)
    prov_type_features.to_pickle(filepath)


def save_feature_tables(
    fpt_counter_list: List[Dict[int, Dict[str, int]]],
    index: Iterable[str],
    output_path: Path,
    kernel_set: str,
    to_level: int,
):
    for level in range(to_level + 1):
        save_single_level_table(
            [
                counters[level] if level in counters else {}
                for counters in fpt_counter_list
            ],
            index,
            output_path,
            kernel_set,
            level,
        )


@click.command()
@click.argument("dataset_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("to_level", default=5)
def main(dataset_folder: str, output_folder: str, to_level: int):
    logger.debug("Working in folder: %s", dataset_folder)
    dataset_path = Path(dataset_folder)
    output_path = Path(output_folder)

    graph_index_filepath = output_path / "graphs.pickled"
    if not graph_index_filepath.exists():
        logger.error("Graphs index file is not found: %s", graph_index_filepath)
        exit(1)

    logger.debug("Reading graphs index")
    graphs = pd.read_pickle(graph_index_filepath)

    # Flat types - Only generic PROV types
    fpt_count_list, timings = count_flatprovenancetypes_for_graphs(
        dataset_path,
        graphs.graph_file,
        level=to_level,
        including_primitives_types=False,
    )
    save_feature_tables(
        fpt_count_list,
        graphs.graph_file,
        output_path,
        kernel_set="FG",
        to_level=to_level,
    )
    # saving timings information
    timings_df = pd.DataFrame(
        timings,
        columns=[f"FG_{h}" for h in range(to_level + 1)],
        index=graphs.graph_file,
    )
    # copy PNA timings over
    # timings_df["PNA"] = graphs.timings_PNA

    # Flat types - Including application types
    fpt_count_list, timings = count_flatprovenancetypes_for_graphs(
        dataset_path, graphs.graph_file, level=to_level, including_primitives_types=True
    )
    save_feature_tables(
        fpt_count_list,
        graphs.graph_file,
        output_path,
        kernel_set="FA",
        to_level=to_level,
    )
    timings_df = timings_df.join(
        pd.DataFrame(
            timings,
            columns=[f"FA_{h}" for h in range(to_level + 1)],
            index=graphs.graph_file,
        )
    )
    # Write timing information back to a pickled DataFrame file
    timings_filepath = output_path / "timings.pickled"
    timings_df.to_pickle(timings_filepath)

    logger.debug("All done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
