"""Generate flat provenance types for generated graphs in outputs/generated and collect timing stats"""
import timeit
import logging
from pathlib import Path

import click
import pandas as pd

from prov.model import ProvDocument

from flatprovenancetypes import calculate_flat_provenance_types
from utils import Timer


@click.command()
@click.argument("generated_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output", type=click.File("wb"))
@click.argument("level", default=5)
def main(generated_folder: str, output: click.utils.LazyFile, level: int):
    timings = []
    filepaths = sorted(Path(generated_folder).glob("*.json"), reverse=True)
    logging.info("Generate flat provenance types up to level %d for %d generated graphs from %s", level, len(filepaths),
                 generated_folder)
    last_update_timer = timeit.default_timer()
    n_rows = len(filepaths) * (level + 1)  # the number of expected rows in the timings table
    for filepath in filepaths:
        n_nodes, n_branches, _, seed, _ = filepath.stem.split("_")
        prov_doc = ProvDocument.deserialize(filepath)
        durations = []
        for h in range(level + 1):
            timer = Timer(verbose=False)
            with timer:
                _ = calculate_flat_provenance_types(
                    prov_doc,
                    h,
                    including_primitives_types=False,  # the generated graphs do not have primitive (application) types
                    counting_wdf_as_two=False,
                )
            durations.append((int(n_nodes), int(n_branches), int(seed), h, timer.interval))
        timings.extend(durations)
        logging.debug("- %s <-- %s.", filepath.stem, ", ".join(f"{h}: {t:.2f}s" for _, _, _, h, t in durations))
        now_timer = timeit.default_timer()
        elapsed_time_since_last_update = now_timer - last_update_timer
        if elapsed_time_since_last_update > 300:  # over 5 minutes has passed since the last progress update
            logging.info("... measured %.2f%% graphs ...", len(timings) * 100 / n_rows)
            last_update_timer = now_timer
    # saving timings information
    timings_df = pd.DataFrame(
        timings,
        columns=["n_nodes", "n_branches", "seed", "level", "duration"],
    )
    timings_df.to_pickle(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
