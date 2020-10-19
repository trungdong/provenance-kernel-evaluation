"""Common data preparation code."""
from collections import Counter
from itertools import chain
import logging

from pathlib import Path
from prov.model import ProvDocument, ProvBundle, ProvRecord, PROV_N_MAP
import pandas as pd

from scripts.metrics import version5
from scripts.utils import Timer

logger = logging.getLogger(__name__)

NETWORK_METRIC_NAMES = list(version5.metrics_names[:-4])
PROV_RELATION_NAMES = [
    # "entity",
    # "activity",
    "wasGeneratedBy",
    "used",
    "wasInformedBy",
    "wasStartedBy",
    "wasEndedBy",
    "wasInvalidatedBy",
    "wasDerivedFrom",
    # "agent",
    "wasAttributedTo",
    "wasAssociatedWith",
    "actedOnBehalfOf",
    "wasInfluencedBy",
    "alternateOf",
    "specializationOf",
    "mentionOf",
    "hadMember",
    # "bundle",
]
PROVENANCE_FEATURE_NAMES = NETWORK_METRIC_NAMES + PROV_RELATION_NAMES


def count_record_types(prov_doc: ProvDocument) -> dict:
    counter = Counter(map(ProvRecord.get_type, prov_doc.get_records()))
    counter.update(
        map(
            ProvRecord.get_type,
            chain.from_iterable(map(ProvBundle.get_records, prov_doc.bundles)),
        )
    )
    result = dict((PROV_N_MAP[rec_type], count) for rec_type, count in counter.items())
    return result


def calculate_provenance_features_for_file(filepath: Path) -> list:
    # Calculate Provenance Network Metrics (22) and number of edge types
    try:
        # load the file
        prov_doc = ProvDocument.deserialize(filepath)
    except Exception as e:
        logger.error("Cannot deserialize %s", filepath)
        raise e
    try:
        timer = Timer(verbose=False)
        with timer:
            # counting the record types
            rec_type_counts = count_record_types(prov_doc)
            prov_rel_cols = [
                rec_type_counts[rec_type] if rec_type in rec_type_counts else 0
                for rec_type in PROV_RELATION_NAMES
            ]
            mv5 = version5(prov_doc, flat=True)  # calculate

        return mv5[:-4] + prov_rel_cols + [timer.interval]
    except Exception as e:
        logger.error("Cannot calculate metrics for %s", filepath)
        raise e


def calculate_provenance_network_metrics(
    dataset_path: Path, graph_index: pd.DataFrame
) -> pd.DataFrame:
    filepaths = [
        dataset_path / graph_filename for graph_filename in graph_index.graph_file
    ]

    pn_data = list(map(calculate_provenance_features_for_file, filepaths))

    return pd.DataFrame(
        pn_data, index=graph_index.index, columns=(PROVENANCE_FEATURE_NAMES + ["timings_PNA"])
    )
