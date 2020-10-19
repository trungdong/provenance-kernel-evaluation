"""Preparing the graph index for experiment."""
from .pokemongo import create_graph_index


dataset_id = "PG-D"
create_graph_index(f"datasets/{dataset_id}", f"outputs/{dataset_id}")
