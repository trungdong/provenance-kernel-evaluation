"""Generate provenance network metrics for a provenance dataset."""

import csv
import logging
from pathlib import Path

import pandas as pd

import click


logger = logging.getLogger(__name__)


@click.command()
@click.argument("output", type=click.File("w"))
@click.argument("kernel_file", nargs=-1, type=click.Path(exists=True))
def count_kernel_types(output, kernel_file):
    logger.debug("Counting types from %d kernel files", len(kernel_file))
    rows = []
    for each in kernel_file:
        filepath = Path(each)
        kernel_set, level = filepath.stem.split("_")
        df = pd.read_pickle(filepath)
        rows.append((kernel_set, level, len(df.columns)))

    # writing the table
    csvwriter = csv.writer(output)
    csvwriter.writerow(("kernel", "level", "type_count"))
    csvwriter.writerows(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    count_kernel_types()
