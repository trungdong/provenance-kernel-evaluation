"""Plotting type counts."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

import click

from scripts.experiments.common import read_kernel_dataframes

logger = logging.getLogger(__name__)


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.argument("kernel_set")
def plot_type_stats(data_path, output_path, kernel_set):
    all_types = read_kernel_dataframes(Path(data_path), kernel_set, to_level=5)
    logger.debug("Seen %d types.", all_types.shape[1])

    sorted_type_occurences = all_types.sum().sort_values(ascending=False)[:20]
    logger.info(
        "Plotting number of occurrences for: %s",
        ", ".join(sorted_type_occurences.index),
    )
    plot = sorted_type_occurences.plot(kind="barh", figsize=(8, 8))
    plt.tight_layout()
    plot.figure.savefig(Path(output_path) / f"top_{kernel_set}_types.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_type_stats()
