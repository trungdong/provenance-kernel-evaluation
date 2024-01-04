from collections import defaultdict
import logging
from pathlib import Path
from typing import List, Tuple

import click
import pandas as pd
import torch
from torch_geometric.data import Data

from prov.model import ProvDocument, ProvElement, ProvRelation, QualifiedName
from prov.constants import *

logger = logging.getLogger(__name__)


class PROV2PyGConverter:
    PROV_NODE_TYPES_MAP = {
        PROV_ENTITY: 1,
        PROV_AGENT: 2,
        PROV_ACTIVITY: 3,
    }
    PROV_EDGE_TYPES_MAP = {
        PROV_INFLUENCE: 0,
        PROV_GENERATION: 1,
        PROV_USAGE: 2,
        PROV_DERIVATION: 3,
        PROV_MEMBERSHIP: 5,
        PROV_COMMUNICATION: 11,
        PROV_START: 12,
        PROV_END: 13,
        PROV_INVALIDATION: 21,
        PROV_ATTRIBUTION: 31,
        PROV_ASSOCIATION: 32,
        PROV_DELEGATION: 41,
        PROV_ALTERNATE: 81,
        PROV_SPECIALIZATION: 82,
        PROV_MENTION: 91,
    }

    def __init__(self, device: str = "cpu"):
        # the device to be used for storing the PyG Data object, default to CPU
        self.device: str = device
        # the type index map for each PROV record type: {record_type: {type: index}}
        # this is used as an ordered set of types (for each record type) as 3.6+ dict maintains the order of insertion
        self._record_types_map: dict[QualifiedName, dict[QualifiedName, int]] = defaultdict(dict)
        # the attribute index map for each PROV record type: {record_type: {attribute: index}}
        self._record_attrs_map: dict[QualifiedName, dict[QualifiedName, int]] = defaultdict(dict)
        # the flattened index maps for the above two maps: {type/attribute: index}
        self._all_node_types_map: dict[QualifiedName, int] = dict()
        self._all_edge_types_map: dict[QualifiedName, int] = dict()
        # the classification (graph) label index map: {label: index}
        self._labels_map: dict[str, int] = dict()

    def index(self, prov_doc: ProvDocument, label: str = None):
        """
        Index the types and attributes of the PROV records in the given document.
        Args:
            prov_doc: the PROV document to be indexed
            label: the classification label of the document (optional)
        """
        flattened = prov_doc.flattened()  # remove bundles, if any
        for record in flattened.get_records():
            flattened_types_map = self._all_node_types_map if record.is_element() else self._all_edge_types_map
            record_type = record.get_type()
            # indexing the types of each record
            types_map = self._record_types_map[record_type]
            for asserted_type in record.get_asserted_types():
                if asserted_type not in types_map:
                    types_map[asserted_type] = len(types_map)
                if asserted_type not in flattened_types_map:
                    flattened_types_map[asserted_type] = len(flattened_types_map)
            attrs_map = self._record_attrs_map[record_type]
            # TODO: process the remaining formal PROV attributes, if any (apart from the first two)
            # indexing the other attributes
            for attr_name, _ in record.extra_attributes:
                if attr_name not in attrs_map:
                    attrs_map[attr_name] = len(attrs_map)

        # indexing the label if previously unseen
        if label is not None and label not in self._labels_map:
            self._labels_map[label] = len(self._labels_map)

    def label_index(self, label: str):
        return self._labels_map[label]

    def to_PyG_data(
        self,
        prov_doc: ProvDocument,
        label: str = None,
        converting_node_types: bool = True,
        converting_edge_types: bool = True,
        converting_node_attrs: bool = True,
        converting_edge_attrs: bool = True,
    ) -> Data:
        # Creating a PyG `Data` object (a homogeneous graph)
        data = {
            "y": torch.tensor(
                [self._labels_map[label]], dtype=torch.long, device=self.device
            )  # classification class for the graph
        }  # the structure of the Data object to be created

        node_map: dict[QualifiedName, int] = dict()

        unified = (
            prov_doc.flattened().unified()
        )  # remove bundles, if any, and merge the records by their identifiers
        node_features = []  # the node feature matrix of the PyG Data object
        for element in unified.get_records(ProvElement):
            # indexing the nodes in the PROV graph
            node_map[element.identifier] = len(node_map)
            node_type = element.get_type()
            features = [
                self.PROV_NODE_TYPES_MAP[node_type]
            ]  # the PROV node type always as the first feature
            if converting_node_types and self._all_node_types_map:
                asserted_types = element.get_asserted_types()
                features.extend(
                    [(1 if a_type in asserted_types else 0) for a_type in self._all_node_types_map]
                )
            node_features.append(features)

            if converting_node_attrs:
                # TODO: process the node's other attributes
                # For continuous attributes, we can directly cast the value to the `torch.float` or `torch.short` type
                # For categorical attributes, we need to map the value to an integer index
                pass

        # int16 should be good for 32k node types
        data["x"] = torch.tensor(node_features, dtype=torch.float, device=self.device)

        edges = []  # the edge index of the PyG Data object
        edge_attr = []  # the edge feature matrix of the PyG Data object
        for relation in unified.get_records(ProvRelation):
            # taking the first two elements of a relation
            attr_pair_1, attr_pair_2 = relation.formal_attributes[:2]
            # only need the QualifiedName (i.e. the value of the attribute)
            qn1, qn2 = attr_pair_1[1], attr_pair_2[1]
            if qn1 and qn2:  # only proceed if both ends of the relation exist
                try:
                    # TODO: infer the type of the unseen node and add it to the node map
                    edges.append((node_map[qn1], node_map[qn2]))
                except KeyError:
                    continue  # skipping this relation
            rel_type = relation.get_type()
            attrs = [
                self.PROV_EDGE_TYPES_MAP[rel_type]
            ]  # the PROV relation type as the first feature
            if converting_edge_types and self._all_edge_types_map:
                asserted_types = relation.get_asserted_types()
                attrs.extend(
                    [(1 if a_type in asserted_types else 0) for a_type in self._all_edge_types_map]
                )
            edge_attr.append(attrs)
            if converting_edge_attrs:
                # TODO: process the edge's attributes
                # Similar to the `converting_node_attrs` section above
                pass

        data["edge_index"] = torch.tensor(
            list(zip(*edges)), dtype=torch.int64, device=self.device
        )  # int64 is required for node indices
        data["edge_attr"] = torch.tensor(
            edge_attr, dtype=torch.long, device=self.device
        )  # int16 should be good for 32k edge types
        data["num_nodes"] = len(node_map)

        return Data.from_dict(data)

    @property
    def num_classes(self):
        return len(self._labels_map)

    @property
    def index_stats(self):
        return {
            "classes": set(self._labels_map.keys()),
            "types": {record_type: set(types_map.keys()) for record_type, types_map in self._record_types_map.items()},
            "attrs": {record_type: set(attrs_map.keys()) for record_type, attrs_map in self._record_attrs_map.items()},
        }


def load_PROV_graph(filepath: Path) -> ProvDocument:
    try:
        # load the file
        prov_doc = ProvDocument.deserialize(filepath)
    except Exception as e:
        logger.error("Cannot deserialize %s", filepath)
        raise e

    return prov_doc


def convert_PROV_graphs_to_PyG_data(
    graphs: pd.DataFrame,  # expecting a `graph_file` column in the input DataFrame
    label_column: str,
    dataset_folder: Path,
    converting_node_types: bool = True,
    converting_edge_types: bool = True,
    converting_node_attrs: bool = True,
    converting_edge_attrs: bool = True,
) -> Tuple[List[Data], int]:
    converter = PROV2PyGConverter()
    # Indexing the types and attributes of the PROV records found in all the graphs
    logger.info("Indexing %d PROV graphs...", len(graphs))
    for row in graphs.itertuples():
        prov_doc = load_PROV_graph(dataset_folder / row.graph_file)
        converter.index(prov_doc, getattr(row, label_column))

    # Showing the summary of the indexing for debugging information
    logger.debug("Indexing summary: %s", converter.index_stats)

    # Converting the graphs to PyG Data objects
    logger.info("Converting %d PROV graphs...", len(graphs))
    data_list: list[Data] = []
    for row in graphs.itertuples():
        prov_doc = load_PROV_graph(dataset_folder / row.graph_file)
        data = converter.to_PyG_data(
            prov_doc,
            getattr(row, label_column),
            converting_node_types=converting_node_types,
            converting_edge_types=converting_edge_types,
            converting_node_attrs=converting_node_attrs,
            converting_edge_attrs=converting_edge_attrs,
        )
        data_list.append(data)

    return data_list, converter.num_classes


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(writable=True, dir_okay=False))
def main(dataset_path: Path, output: str):
    graph_index_filepath = dataset_path / "graphs.csv"
    if not graph_index_filepath.exists():
        logger.error("Graphs index file is not found: %s", graph_index_filepath)
        exit(1)
    logger.info("Reading graph database from: %s", graph_index_filepath)
    graphs_df = pd.read_csv(graph_index_filepath)

    data_list = convert_PROV_graphs_to_PyG_data(graphs_df, "label", dataset_path)
    logger.info("Saving the converted data_list to: %s", output)
    torch.save(data_list, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
