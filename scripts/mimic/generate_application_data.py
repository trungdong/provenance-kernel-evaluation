"""This module extracts relevant MIMIC application data to a table.

It restricts to admissions where we have data on the procedures carried therein.
"""

from collections import defaultdict
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common import SKIPPED_ADMISSION_IDS
from db import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parents[2]
output_filepath = ROOT_DIR / "outputs" / "MIMIC-PXC7" / "graphs.pickled"

print("Finding admissions with procedures...")
procedures = pd.read_sql_table(
    "procedureevents_mv", db, schema="mimiciii", index_col="row_id"
)
adm_ids_with_procedures = set(procedures.hadm_id.unique())
print(f"Found {len(adm_ids_with_procedures)} admissions with procedure data.")

# Looking up definitions of procedures
items = pd.read_sql_table("d_items", db, schema="mimiciii", index_col="row_id")
procedure_types = items[items.linksto == "procedureevents_mv"]

print("Filtering out procedures in Category 7 - Communication")
communication_procedure_ids = set(
    procedure_types[procedure_types.category == "7-Communication"].itemid
)
procedures = procedures[~procedures.itemid.isin(communication_procedure_ids)]
print(f"Total {len(procedures)} procedures remained.")

# Counting procedures in each admissions
procedures_data = defaultdict(lambda: defaultdict(int))
for row in procedures.itertuples():
    procedures_data[row.hadm_id][row.itemid] += 1

# Reading diagnotic codes for admissions
drgcodes = pd.read_sql_table("drgcodes", db, schema="mimiciii", index_col="row_id")
labels = defaultdict(dict)
for drg_code in drgcodes.itertuples():
    if drg_code.hadm_id not in SKIPPED_ADMISSION_IDS:
        labels[drg_code.hadm_id][drg_code.drg_type] = int(drg_code.drg_code)


print("Checking if a patient is returning to the hospital")
admissions = pd.read_sql_table("admissions", db, schema="mimiciii", index_col="hadm_id")

# Filtering out admissions without procedure data
admissions = admissions[admissions.index.isin(adm_ids_with_procedures)]

patient_admissions = (
    admissions[["subject_id", "admittime"]].groupby("subject_id").agg([np.min, np.max])
)

patient_admissions.columns = ["admittime_first", "admittime_last"]

returning_patients = {
    patient.Index: (patient.admittime_first, patient.admittime_last)
    for patient in patient_admissions.itertuples()
    if patient.admittime_first < patient.admittime_last
}
print(
    f"There are {len(returning_patients)} returning patients (out of {len(patient_admissions)})."
)

rows = [
    (
        f"{admission.Index}.json",
        admission.Index,
        admission.subject_id,
        labels[admission.Index]["HCFA"] if "HCFA" in labels[admission.Index] else -1,
        labels[admission.Index]["APR "] if "APR " in labels[admission.Index] else -1,
        labels[admission.Index]["MS"] if "MS" in labels[admission.Index] else -1,
        (  # is this the first admission?
            admission.subject_id not in returning_patients
            or admission.admittime == returning_patients[admission.subject_id][0]
        ),
        (  # is this the last admission?
            admission.subject_id not in returning_patients
            or admission.admittime == returning_patients[admission.subject_id][1]
        ),
        (  # will the patient return?
            admission.subject_id in returning_patients
            and admission.admittime < returning_patients[admission.subject_id][1]
        ),
        (  # has the patient been admitted before?
            admission.subject_id in returning_patients
            and admission.admittime > returning_patients[admission.subject_id][0]
        ),
        admission.hospital_expire_flag,
        dict(procedures_data[admission.Index]),
    )
    for admission in admissions.itertuples()
    if admission.Index not in SKIPPED_ADMISSION_IDS
]

graphs_index = pd.DataFrame(
    rows,
    columns=[
        "graph_file",
        "hadm_id",
        "subject_id",
        "drgcode_hcfa",
        "drgcode_apr",
        "drgcode_ms",
        "first",
        "last",
        "will_return",
        "returning",
        "dead",
        "procedures_data",
    ],
)

print("Generating features from procedure data...")
all_procedure_codes = set()
for keys in graphs_index.procedures_data.map(lambda d: d.keys()):
    all_procedure_codes.update(keys)
all_procedure_codes = sorted(all_procedure_codes)
print(f"Seen {len(all_procedure_codes)} different procedures.")

procedure_code_pos_lookup = dict()
for pos, val in enumerate(all_procedure_codes):
    procedure_code_pos_lookup[val] = pos

n_features = len(all_procedure_codes)
rows = []
for procedure_codes in graphs_index.procedures_data:
    row = np.zeros(n_features, dtype=int)
    for code, count in procedure_codes.items():
        row[procedure_code_pos_lookup[code]] = count
    rows.append(row)

procedures_features = pd.DataFrame(
    data=rows, index=graphs_index.index, columns=all_procedure_codes
)

graphs_index = graphs_index.join(procedures_features)
graphs_index.drop("procedures_data", axis="columns", inplace=True)

print("Write the graphs index table to file: ", output_filepath)
graphs_index.to_pickle(output_filepath)
graphs_index.to_csv(output_filepath.with_suffix(".csv"))
