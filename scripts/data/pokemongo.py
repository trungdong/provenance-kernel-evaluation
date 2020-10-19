"""Common code for preparing the graph index of Pokemon Go datasets."""
import logging
from pathlib import Path

import pandas as pd
from prov.model import (
    ProvDocument,
    ProvElement,
    ProvEntity,
    ProvActivity,
    ProvInvalidation,
    Namespace,
)
import numpy as np

from .common import calculate_provenance_network_metrics

POKEMON_GO_DATA_COLUMNS = [
    "n_balls_collected",
    "n_pokemons_captured",
    "n_pokemons_disposed",
    "strength_captured_avg",
    "strength_disposed_avg",
]

logger = logging.getLogger(__name__)
NS_PGO = Namespace("pgo", "http://sociam.org/pokemongo#")
PGO_strength = NS_PGO["strength"]


def create_graph_index(dataset_path, output_path):
    logger.debug("Working in folder: %s", dataset_path)
    dataset_path = Path(dataset_path)

    graph_index_filepath = dataset_path / "graphs.csv"
    if not graph_index_filepath.exists():
        logger.error("Graphs index file is not found: %s", graph_index_filepath)
        exit(1)

    logger.debug("Reading graphs index...")
    graphs = pd.read_csv(graph_index_filepath)

    logger.debug("Extracting Pokemon Go data from the provenance graphs")
    pg_data = [
        extract_pg_data(dataset_path / graph_filename)
        for graph_filename in graphs.graph_file
    ]
    pg_df = pd.DataFrame(pg_data, index=graphs.index, columns=POKEMON_GO_DATA_COLUMNS)
    graphs = graphs.join(pg_df)
    print(graphs.head())

    logger.debug("Calculating provenance network metrics for %d graphs...", len(graphs))
    metrics = calculate_provenance_network_metrics(dataset_path, graphs)
    graphs = graphs.join(metrics)

    output_filepath = Path(output_path) / "graphs.pickled"
    graphs.to_pickle(output_filepath)


def extract_pg_data(filepath: Path):
    prov_doc = ProvDocument.deserialize(filepath)

    n_balls_collected = 0

    pokemons_strength = dict()
    pokemons_captured = []
    pokemons_disposed = []
    strength_captured_avg = -1
    strength_disposed_avg = -1

    for record in prov_doc.get_records(ProvElement):
        if isinstance(record, ProvEntity):
            ent_id = str(record.identifier)
            if "pokemons" in ent_id:
                strength_values = record.get_attribute(PGO_strength)  # type: set
                strength = (
                    next(iter(strength_values)) if strength_values else 0
                )  # type: int
                pokemon_id = ent_id[:-2]
                if ent_id.endswith(".0"):
                    pokemons_captured.append(pokemon_id)
                    if strength and (pokemon_id not in pokemons_strength):
                        pokemons_strength[pokemon_id] = strength
        elif isinstance(record, ProvActivity):
            act_id = str(record.identifier)
            if "collectballs" in act_id:
                n_balls_collected += 1

    for record in prov_doc.get_records(ProvInvalidation):
        ent_id = str(record.args[0])
        pokemon_id = ent_id[:-2]
        pokemons_disposed.append(pokemon_id)

    n_pokemons_captured = len(pokemons_captured)
    n_pokemons_disposed = len(pokemons_disposed)
    if pokemons_captured:
        strength_captured_avg = np.mean(
            [
                pokemons_strength[pokemon_id]
                for pokemon_id in pokemons_captured
                if pokemon_id in pokemons_strength
            ]
        )
    if pokemons_disposed:
        strength_disposed_avg = np.mean(
            [
                pokemons_strength[pokemon_id]
                for pokemon_id in pokemons_disposed
                if pokemon_id in pokemons_strength
            ]
        )

    return (
        n_balls_collected,
        n_pokemons_captured,
        n_pokemons_disposed,
        strength_captured_avg,
        strength_disposed_avg,
    )
