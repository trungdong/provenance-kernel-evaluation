"""Calculating the flat provenance types from a provenance document."""
from collections import Counter, defaultdict
from typing import Dict, Iterable, FrozenSet, Set, Tuple

from prov.model import ProvDocument, ProvElement, ProvRecord, QualifiedName
from prov.constants import (
    PROV_ENTITY,
    PROV_ACTIVITY,
    PROV_GENERATION,
    PROV_USAGE,
    PROV_COMMUNICATION,
    PROV_START,
    PROV_END,
    PROV_INVALIDATION,
    PROV_DERIVATION,
    PROV_AGENT,
    PROV_ATTRIBUTION,
    PROV_ASSOCIATION,
    PROV_DELEGATION,
    PROV_INFLUENCE,
    PROV_ALTERNATE,
    PROV_SPECIALIZATION,
    PROV_MENTION,
    PROV_MEMBERSHIP,
)

SHORT_NAMES = {
    PROV_ENTITY: "ent",
    PROV_ACTIVITY: "act",
    PROV_GENERATION: "gen",
    PROV_USAGE: "usd",
    PROV_COMMUNICATION: "wib",
    PROV_START: "wsb",
    PROV_END: "web",
    PROV_INVALIDATION: "inv",
    PROV_DERIVATION: "der",
    PROV_AGENT: "agt",
    PROV_ATTRIBUTION: "att",
    PROV_ASSOCIATION: "waw",
    PROV_DELEGATION: "del",
    PROV_INFLUENCE: "inf",
    PROV_ALTERNATE: "alt",
    PROV_SPECIALIZATION: "spe",
    PROV_MENTION: "men",
    PROV_MEMBERSHIP: "mem",
}

Fingerprint = FrozenSet[QualifiedName]
FlatProvenanceType = Tuple[Fingerprint, ...]
MultiLevelTypeDict = Dict[int, Dict[QualifiedName, FlatProvenanceType]]


def get_element_types(
    element: ProvElement, including_additional_types: bool = True
) -> Set[QualifiedName]:
    types = {element.get_type()}
    if including_additional_types:
        types.update(
            {t for t in element.get_asserted_types() if isinstance(t, QualifiedName)}
        )
    return types


def join_flat_types(
    t1: FlatProvenanceType, t2: FlatProvenanceType
) -> FlatProvenanceType:
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    assert len(t1) == len(t2)
    return tuple(f1 | f2 for f1, f2 in zip(t1, t2))


def format_fingerprint(f: Fingerprint) -> str:
    try:
        types = sorted(SHORT_NAMES[qn] for qn in f)
    except KeyError:
        # only happen with fingerprint_0 containing additional types
        types = sorted([SHORT_NAMES[qn] for qn in f if qn in SHORT_NAMES])
        additional_types = sorted([str(qn) for qn in f if qn not in SHORT_NAMES])
        types.extend(additional_types)
    return "[" + "|".join(types) + "]"


def print_flat_type(t: FlatProvenanceType) -> str:
    return "→".join(map(format_fingerprint, reversed(t)))


def count_fp_types(types: Iterable[FlatProvenanceType]) -> Dict[str, int]:
    counter = Counter(types)
    return {print_flat_type(t): count for t, count in counter.items()}


def calculate_flat_provenance_types(
    prov_doc: ProvDocument,
    to_level: int = 0,
    including_primitives_types: bool = True,
    counting_wdf_as_two: bool = False,
) -> MultiLevelTypeDict:
    # flatten all the bundles, if any
    prov_doc = prov_doc.flattened()

    # initialise index structures
    level0_types = defaultdict(set)  # type: Dict[QualifiedName, Set[QualifiedName]]
    predecessors = defaultdict(
        set
    )  # type: Dict[QualifiedName, Set[Tuple[QualifiedName, QualifiedName]]]

    # indexing node types and relations
    for rec in prov_doc.get_records():  # type: ProvRecord
        if rec.is_element():
            level0_types[rec.identifier] |= get_element_types(
                rec, including_primitives_types
            )
        elif rec.is_relation():
            rel_type = rec.get_type()
            attrs, values = zip(*rec.formal_attributes)
            # expecting a QualifiedName from the first argument of a relation
            predecessor, successor = values[:2]
            if predecessor is not None and successor is not None:
                predecessors[successor].add((rel_type, predecessor))

    # the type map for this graph
    fp_types = defaultdict(dict)  # type: MultiLevelTypeDict
    # converting type sets to FlatProvenanceType level 0
    fp_types[0] = {node: (frozenset(level0_types[node]),) for node in level0_types}
    # propagating level-0 types to the specified level
    for k in range(1, to_level + 1):
        # only propagating (k-1) types from nodes that have them
        for node, types in fp_types[k - 1].items():
            # propagating the types to the predecessors
            for rel_type, predecessor in predecessors[node]:
                k_type = types + (frozenset({rel_type}),)  # type: FlatProvenanceType
                if counting_wdf_as_two and (rel_type == PROV_DERIVATION):
                    k_p1_type = k_type + (
                        frozenset({rel_type}),
                    )  # type: FlatProvenanceType
                    fp_types[k + 1][predecessor] = (
                        join_flat_types(fp_types[k + 1][predecessor], k_p1_type)
                        if predecessor in fp_types[k + 1]
                        else k_p1_type
                    )
                else:
                    fp_types[k][predecessor] = (
                        join_flat_types(fp_types[k][predecessor], k_type)
                        if predecessor in fp_types[k]
                        else k_type
                    )

    return fp_types
