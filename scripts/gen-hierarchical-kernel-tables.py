"""Generate feature vectors from hierarchical provenance types for a set of graphs from level 0 to 5."""

import json
import logging
from pathlib import Path
import tarfile
import tempfile
from typing import Dict, Iterable, List

import click
import pandas as pd
from scipy.sparse import csr_matrix

from provtools import provconvert_file, provman_kernelize, provman_batch

ROOT_DIR = Path(__file__).parents[1]

logger = logging.getLogger(__name__)


def generate_kenerlize_commands(
    dataset_path: Path, graph_files: pd.Series, output_path: Path
) -> str:
    dataset = dataset_path.stem
    config_folder = ROOT_DIR / "configs" / "provman"
    config_path = config_folder / f"kernelize-{dataset}.json"
    config_path_str = str(config_path) if config_path.exists() else None

    if config_path_str is not None:
        logger.debug("Using config file: %s", config_path_str)

    # paths for variations of kernels and create them
    HA_kernels_path = output_path / "HA"
    HA_kernels_path.mkdir(parents=True, exist_ok=True)
    HG_kernels_path = output_path / "HG"
    HG_kernels_path.mkdir(parents=True, exist_ok=True)
    TA_kernels_path = output_path / "TA"
    TA_kernels_path.mkdir(parents=True, exist_ok=True)
    TG_kernels_path = output_path / "TG"
    TG_kernels_path.mkdir(parents=True, exist_ok=True)

    command_lines = []  # type: List[str]

    for graph_filename in graph_files:
        graph_filepath = dataset_path / graph_filename

        # converting the PROV document to PROV-N if needed
        if graph_filepath.suffix != ".provn":
            provn_graph_filepath = graph_filepath.with_suffix(".provn")
            if not provn_graph_filepath.exists():
                logger.debug(
                    f"Converting {graph_filepath} into {provn_graph_filepath}..."
                )
                provconvert_file(graph_filepath, provn_graph_filepath)
        else:
            provn_graph_filepath = graph_filepath

        graph_id = graph_filepath.stem
        kernelize_common_args = {
            "filepath": provn_graph_filepath,
            "file_id": graph_id,
            "level0": config_path_str,
            "level_from": -5,
            "level_to": 5,
            "return_cmd_only": True,
        }

        # generating the hierarchical types WITHOUT WdfT transformations
        cmd_line = "provmanagement " + provman_kernelize(
            outpath=HA_kernels_path,
            triangle=False,
            **kernelize_common_args,
        )
        command_lines.append(cmd_line)
        # generating the hierarchical types WITHOUT WdfT transformations using no application types
        cmd_line = "provmanagement " + provman_kernelize(
            outpath=HG_kernels_path,
            no_primitives=True,
            triangle=False,
            **kernelize_common_args,
        )
        command_lines.append(cmd_line)

        # generating the hierarchical types with WdfT transformations
        cmd_line = "provmanagement " + provman_kernelize(
            outpath=TA_kernels_path,
            triangle=True,
            **kernelize_common_args,
        )
        command_lines.append(cmd_line)
        # generating the hierarchical types with WdfT transformations using no application types
        cmd_line = "provmanagement " + provman_kernelize(
            outpath=TG_kernels_path,
            no_primitives=True,
            triangle=True,
            **kernelize_common_args,
        )
        command_lines.append(cmd_line)

    return "\n".join(command_lines)


def load_feature_file(
    kernels_path: Path, kernel_set, graph_filename, level: int
) -> dict:
    graph_id = Path(graph_filename).stem  # stripped off the file extension
    features_filepath = kernels_path / kernel_set / f"{graph_id}-{level}.features.json"

    with features_filepath.open() as f:
        dict_structure = json.load(f)
        features = dict_structure[
            "features"
        ]  # expecting a 'features' map in the JSON file
        # remove the empty set if any
        try:
            del features["[]"]
        except KeyError:
            pass
        return features


def read_provenance_types_for_graphs(
    kernels_path: Path,
    graph_filenames: Iterable[str],
    kernel_set: str,
    level: int,
) -> List[Dict[str, int]]:
    results = []  # type: List[Dict[str, int]]

    logger.debug("Reading PROV type features from %s level %s", kernel_set, level)
    for graph_filename in graph_filenames:
        results.append(
            load_feature_file(kernels_path, kernel_set, graph_filename, level)
        )
    return results


def generate_save_kernel_table(
    output_path: Path,
    kernels_path: Path,
    graph_filenames: pd.Series,
    kernel_set: str,
    level: int,
):
    logger.debug("Generating feature table for %s level %s", kernel_set, level)
    # read feature files
    kernels = read_provenance_types_for_graphs(
        kernels_path, graph_filenames, kernel_set, level
    )

    # Builing the sparse matrix  for the feature vectors
    indptr = [0]
    indices = []
    data = []
    vocabulary = dict()  # type: Dict[str, int]
    for feature in kernels:
        for prov_type, count in feature.items():
            index = vocabulary.setdefault(prov_type, len(vocabulary))
            indices.append(index)
            data.append(count)
        indptr.append(len(indices))
    csr_m = csr_matrix((data, indices, indptr), dtype=int)
    logger.debug("Seen %d unique PROV types.", len(vocabulary))

    logger.debug("Mapping PROV types to feature vector")
    prov_types_df = pd.DataFrame(vocabulary.items(), columns=["Type", "Number"])
    prov_types_df = prov_types_df[["Number", "Type"]]
    prov_types_df.Number += 1
    # making sure the order of the types are the same (asc. order) as in the column index
    prov_types_df.sort_values("Number", inplace=True)
    # prefixing the feature name with the kernel type and level
    kernel_id = kernel_set + str(level)
    prov_types_df["Number"] = prov_types_df.Number.map(lambda n: f"{kernel_id}_{n}")
    types_filepath = output_path / f"{kernel_set}_{level}_prov_types.csv"
    logger.debug("Writing PROV type mappings to: %s", types_filepath)
    prov_types_df.to_csv(types_filepath, index=False)

    # save kernel tables
    prov_type_features = pd.DataFrame.sparse.from_spmatrix(
        data=csr_m, index=graph_filenames, columns=prov_types_df.Number
    )
    table_filepath = output_path / f"{kernel_set}_{level}.pickled"
    logger.debug("Writing PROV type features to: %s", table_filepath)
    prov_type_features.to_pickle(table_filepath)


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

    # create a temp dir for feature files, automatically removed when all done
    with tempfile.TemporaryDirectory() as tmpdirname:
        kernels_path = Path(tmpdirname)
        # generate commands
        batch_commands = generate_kenerlize_commands(
            dataset_path, graphs.graph_file, kernels_path
        )
        # call batch mode
        provman_batch(batch_commands)

        # store all the output files in the temp folder into a tar archive
        tarfile_path = output_path / "provman-features-files.tar.gz"
        logger.debug("Saving provman's output files to: %s", tarfile_path)
        with tarfile.open(tarfile_path, "w:gz") as tar:
            tar.add(kernels_path / "TG", "TG")
            tar.add(kernels_path / "TA", "TA")

        # generate kernel tables from feature files and save to the output path
        tables_output_path = output_path / "kernels"
        tables_output_path.mkdir(parents=True, exist_ok=True)
        for kernel_set in ["TA", "TG"]:
            for level in range(-5, 6):
                generate_save_kernel_table(
                    tables_output_path,
                    kernels_path,
                    graphs.graph_file,
                    kernel_set,
                    level,
                )

    logger.debug("All done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
