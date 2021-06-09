"""Generate feature vectors from flat provenance types for a set of graphs from level 0 to 5."""

from collections import Counter
import json
import logging
from pathlib import Path
import pickle
from typing import Collection, Dict, Iterable, List, Tuple, FrozenSet

import pandas as pd
from scipy.sparse import csr_matrix

from prov.model import ProvDocument

from flatprovenancetypes import (
    FlatProvenanceType,
    calculate_flat_provenance_types,
    print_flat_type,
    ϕ,
)
from utils import Timer

import click


logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).parents[1]


def count_flatprovenancetypes_for_graphs(
    dataset_path: Path,
    graph_filenames: Collection[str],
    level: int,
    including_primitives_types: bool,
    counting_wdf_as_two: bool = False,
    ignored_types: FrozenSet[str] = ϕ,
) -> Tuple[List[Dict[int, Dict[FlatProvenanceType, int]]], List[List[float]]]:
    logger.debug(
        "Producing linear provenance types up to level %s "
        "(with application types: %s, counting derivations as 2-length edges: %s) "
        "for %d graphs...",
        level,
        including_primitives_types,
        counting_wdf_as_two,
        len(graph_filenames),
    )
    results = []  # type: List[Dict[int, Dict[FlatProvenanceType, int]]]
    timings = []  # type: List[List[float]]
    for graph_filename in graph_filenames:
        filepath = dataset_path / graph_filename
        prov_doc = ProvDocument.deserialize(filepath)
        durations = []  # type: List[float]
        features = dict()  # type: Dict[int, Dict[FlatProvenanceType, int]]
        for h in range(level + 1):
            timer = Timer(verbose=False)
            with timer:
                fp_types = calculate_flat_provenance_types(
                    prov_doc,
                    h,
                    including_primitives_types,
                    counting_wdf_as_two,
                    ignored_types=ignored_types,
                )
            # counting only the last level
            features[h] = Counter(fp_types[h].values())
            durations.append(timer.interval)
        results.append(features)
        timings.append(durations)
    return results, timings


def save_single_level_table(
    feature_list: List[Dict[FlatProvenanceType, int]],
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
    vocabulary = dict()  # type: Dict[FlatProvenanceType, int]

    # Constructing a sparse matrix to store the counts of each FlatProvenanceType
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

    # storing a mapping between a type's name and its definition
    types_map = {row.Number: row.Type for row in prov_types_df.itertuples()}
    filepath = output_path / "kernels" / f"{kernel_set}_{level}_types_map.pickled"
    with filepath.open("wb") as f:
        logger.debug("- Writing type mappings to: %s", filepath)
        pickle.dump(types_map, f)

    # converting FlatProvenanceType to a pretty-formatted string
    prov_types_df["Type"] = prov_types_df.Type.map(print_flat_type)

    filepath = output_path / "kernels" / f"{kernel_set}_{level}_types.csv"
    logger.debug("- Writing humand-readable type definitions to: %s", filepath)
    prov_types_df.to_csv(filepath, index=False)

    prov_type_features = pd.DataFrame.sparse.from_spmatrix(
        data=csr_m, index=index, columns=prov_types_df.Number
    )
    logger.debug(prov_type_features.info())
    filepath = output_path / "kernels" / f"{kernel_set}_{level}.pickled"
    logger.debug("- Writing feature table to: %s", filepath)
    prov_type_features.to_pickle(filepath)


def save_feature_tables(
    fpt_counter_list: List[Dict[int, Dict[FlatProvenanceType, int]]],
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


def read_ignored_types(config_path: Path) -> FrozenSet[str]:
    with config_path.open() as f:
        configs: dict = json.load(f)

        if "ignore" in configs:
            ignored_type_uri_list = configs["ignore"]
            return frozenset(ignored_type_uri_list)

    return ϕ


@click.command()
@click.argument("dataset_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("to_level", default=5)
def main(dataset_folder: str, output_folder: str, to_level: int):
    logger.debug("Working in folder: %s", dataset_folder)
    dataset_path = Path(dataset_folder)
    output_path = Path(output_folder)

    dataset = dataset_path.stem
    config_folder = ROOT_DIR / "configs" / "provman"
    config_path = config_folder / f"kernelize-{dataset}.json"
    config_path_str = str(config_path) if config_path.exists() else None

    ignored_types: FrozenSet[str] = ϕ
    if config_path_str is not None:
        logger.debug("Using config file: %s", config_path_str)
        ignored_types = read_ignored_types(config_path)
        logger.debug("- Types to be ignored: %s", ignored_types)

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
        counting_wdf_as_two=False,
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
    # same types but counting derivations as 2-length edges
    fpt_count_list, timings = count_flatprovenancetypes_for_graphs(
        dataset_path,
        graphs.graph_file,
        level=to_level,
        including_primitives_types=False,
        counting_wdf_as_two=True,
    )
    save_feature_tables(
        fpt_count_list,
        graphs.graph_file,
        output_path,
        kernel_set="DG",
        to_level=to_level,
    )
    timings_df = timings_df.join(
        pd.DataFrame(
            timings,
            columns=[f"DG_{h}" for h in range(to_level + 1)],
            index=graphs.graph_file,
        )
    )

    # Flat types - Including application types
    fpt_count_list, timings = count_flatprovenancetypes_for_graphs(
        dataset_path,
        graphs.graph_file,
        level=to_level,
        including_primitives_types=True,
        counting_wdf_as_two=False,
        ignored_types=ignored_types,
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
    # same types but counting derivations as 2-length edges
    fpt_count_list, timings = count_flatprovenancetypes_for_graphs(
        dataset_path,
        graphs.graph_file,
        level=to_level,
        including_primitives_types=True,
        counting_wdf_as_two=True,
        ignored_types=ignored_types,
    )
    save_feature_tables(
        fpt_count_list,
        graphs.graph_file,
        output_path,
        kernel_set="DA",
        to_level=to_level,
    )
    timings_df = timings_df.join(
        pd.DataFrame(
            timings,
            columns=[f"DA_{h}" for h in range(to_level + 1)],
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
