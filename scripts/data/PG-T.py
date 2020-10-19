"""Preparing the graph index for experiment."""
from .pokemongo import create_graph_index


dataset_id = "PG-T"
create_graph_index(f"datasets/{dataset_id}", f"outputs/{dataset_id}")
