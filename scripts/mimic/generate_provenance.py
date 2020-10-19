from collections import defaultdict
import logging
from pathlib import Path

import pandas as pd
from prov.model import ProvDocument

from admission import Admission, get_blank_prov_document
from common import SKIPPED_ADMISSION_IDS
import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parents[2]


def save_provenance(prov_doc: ProvDocument, filepath: Path):
    logging.debug("Saving provenance files:")
    logging.debug(" - %s", filepath)
    with filepath.open("w") as f:
        prov_doc.serialize(f)
    provn_content = prov_doc.get_provn()
    filepath = filepath.with_suffix(".provn")
    logging.debug(" - %s", filepath)
    with filepath.open("w") as f:
        f.write(provn_content)


if __name__ == "__main__":
    admissions = pd.read_sql_table(
        "admissions", db.db, schema="mimiciii", index_col="hadm_id"
    )
    output_path = ROOT_DIR / "datasets" / "MIMIC-PXC7"
    output_path.mkdir(parents=True, exist_ok=True)

    count_processed = 0
    count_skipped = 0
    graph_filenames = []
    for admission_id in admissions.index:
        if admission_id in SKIPPED_ADMISSION_IDS:
            continue

        json_filename = f"{admission_id}.json"
        graph_filenames.append((json_filename, admission_id))
        json_filepath = output_path / json_filename
        if json_filepath.exists():
            logger.debug("Already exists, skipping admission #%d", admission_id)
            count_skipped += 1
            continue

        logger.info("Processing admission #%d", admission_id)
        prov_doc = get_blank_prov_document()
        adm = Admission(prov_doc, admission_id)
        adm.process()
        save_provenance(prov_doc, json_filepath)
        count_processed += 1

    logger.info("Processed: %d; Skipped: %d", count_processed, count_skipped)

    graphs_index = pd.DataFrame(graph_filenames, columns=["graph_file", "hadm_id"])
    graphs_index.to_csv(output_path / "graphs.csv", index=False)

    db.close_session()
