"""Plotting type counts."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import click

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
sns.set_context("talk")


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def plot_type_counts(input, output):
    df = pd.read_csv(input)
    plot = sns.barplot(x="level", y="type_count", hue="kernel", data=df)
    plot.figure.set_size_inches(12, 6)
    plt.tight_layout()
    for p in plot.patches:
        plot.annotate(
            "{:.0f}".format(p.get_height()),
            (p.get_x() + 0.133, p.get_height()),
            ha="center",
            va="bottom",
            color="black",
        )
    plot.figure.savefig(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_type_counts()
