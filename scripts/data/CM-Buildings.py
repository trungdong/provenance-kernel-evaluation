"""Preparing the graph index for experiment."""
from .collabmap import copy_graph_index


dataset_id = "CM-Buildings"
copy_graph_index(f"datasets/{dataset_id}", f"outputs/{dataset_id}")
