import datetime
import logging
from collections import namedtuple
from pathlib import Path
from typing import List, MutableMapping, Sequence

from prov.model import (
    Namespace,
    ProvDocument,
    ProvBundle,
    ProvEntity,
    ProvAgent,
    PROV_TYPE,
    PROV,
)
from prov.dot import prov_to_dot

import db

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

output_path = Path("outputs")

# Provenance initialisation
# Namespaces
ns_type_uri = "https://openprovenance.org/ns/mimic#"
ns_data_uri = "https://mimic.physionet.org/mimiciii/1.4"

# Types
ns_type = Namespace("mimic", ns_type_uri)
ns_attrs = Namespace("attrs", ns_type_uri + "attr_")

# Data
gen_data_ns = lambda name: Namespace(name, f"{ns_data_uri}/{name}/")
ns_patients = gen_data_ns("patients")
ns_unit = gen_data_ns("units")
ns_staff = gen_data_ns("staff")
ns_process = gen_data_ns("process")
ns_admissions = gen_data_ns("admissions")
ns_stay = gen_data_ns("stay")
ns_procedures = gen_data_ns("procedures")

all_namespaces = [
    ns_type,
    ns_attrs,
    ns_patients,
    ns_unit,
    ns_staff,
    ns_admissions,
    ns_process,
    ns_stay,
    ns_procedures,
]
procedure_entities = {}  # type: MutableMapping[int, ProvEntity]

EVENT_TYPES = {
    "admit": ns_type["Admitting"],
    "transfer": ns_type["Transfering"],
    "discharge": ns_type["Discharging"],
}


class Patient:
    def __init__(self, prov_bundle: ProvBundle, db_patient):
        self.prov_bundle = prov_bundle  # type: ProvBundle
        self.subject_id = db_patient.subject_id  # type: int
        patient_id = ns_patients[str(self.subject_id)]
        attributes = [
            (PROV_TYPE, PROV["Person"]),
            (ns_type["gender"], db_patient.gender),
            (ns_type["dob"], db_patient.dob),
        ]
        if db_patient.dod is not None:
            attributes.append((ns_type["dod"], db_patient.dod))
        self.prov_entity = prov_bundle.entity(patient_id, attributes)
        self.specializations = dict()
        self.versions = dict()  # type: MutableMapping[datetime.datetime, ProvEntity]

    def get_entity(self):
        return self.prov_entity

    def at_admission(self, admission_id: int) -> ProvEntity:
        patient_adm_id = ns_patients[f"{self.subject_id}/{admission_id}"]
        if patient_adm_id in self.specializations:
            return self.specializations[patient_adm_id]
        else:
            patient_adm_entity = self.prov_bundle.entity(
                patient_adm_id, [(PROV_TYPE, ns_type["Patient"])]
            )
            patient_adm_entity.specializationOf(self.prov_entity)
            self.specializations[patient_adm_id] = patient_adm_entity
            return patient_adm_entity

    def at_admission_time(
        self, admission_id: int, time: datetime.datetime
    ) -> ProvEntity:
        try:
            return self.versions[time]
        except KeyError:
            patient_adm_ts_id = ns_patients[
                f"{self.subject_id}/{admission_id}.{time.timestamp()}"
            ]
            patient_adm_ts_entity = self.prov_bundle.entity(
                patient_adm_ts_id, [(PROV_TYPE, ns_type["Patient"])]
            )
            self.versions[time] = patient_adm_ts_entity
            return patient_adm_ts_entity


def get_unit_agent(
    prov_bundle: ProvBundle, unit_id: int, unit_specific_type: str
) -> ProvAgent:
    unit_agent_id = ns_unit[str(unit_id)]
    unit_records = prov_bundle.get_record(unit_agent_id)
    attributes = [
        (PROV_TYPE, ns_type["Ward"]),
    ]
    if unit_specific_type is not None:
        attributes.append((PROV_TYPE, ns_type[unit_specific_type]))
    return (
        unit_records[0]
        if unit_records
        else prov_bundle.agent(unit_agent_id, attributes)
    )


def get_staff_agent(prov_bundle: ProvBundle, care_giver: db.CareGiver) -> ProvAgent:
    staff_id = ns_staff[str(care_giver.cgid)]
    staff_records = prov_bundle.get_record(staff_id)
    return (
        staff_records[0]
        if staff_records
        else prov_bundle.agent(
            staff_id,
            {
                "prov:type": PROV["Person"],
                "prov:label": care_giver.label,
                ns_attrs["description"]: care_giver.description,
            },
        )
    )


def get_process_entity(prov_bundle: ProvBundle, item: db.Item) -> ProvEntity:
    try:
        entity_process = procedure_entities[item.itemid]
        if not prov_bundle.get_record(entity_process.identifier):
            # the entity was created in another admission
            prov_bundle.add_record(entity_process)
        return entity_process
    except KeyError:
        entity_id = ns_process[str(item.itemid)]
        entity_process = prov_bundle.entity(
            entity_id,
            {
                "prov:type": ns_type["Process"],
                "prov:label": item.label,
                ns_attrs["abbreviation"]: item.abbreviation,
                ns_attrs["category"]: item.category,
                ns_attrs["dbsource"]: item.dbsource,
            },
        )
        procedure_entities[item.itemid] = entity_process
        return entity_process


Procedure = namedtuple(
    "Procedure",
    [
        "id",
        "category",
        "cat_description",
        "care_giver",
        "process",
        "start_time",
        "end_time",
        "location",
        "location_cat",
        "value",
        "uom",
    ],
)


class Admission:
    def __init__(self, prov_bundle: ProvBundle, admission_id: int):
        self.admission_id = admission_id
        self.prov_bundle = prov_bundle  # type: ProvBundle
        self.db_session = None  # type: db.Session
        self.patient = None  # type: Patient
        self.db_admission = None
        self.procedures = []  # type: List[Procedure]
        self.current_ward = None

    def prepare_procedures(self):
        for procedure, item, care_giver in db.get_procedures_mv(
            self.db_session, self.admission_id
        ):

            if item.category == "7-Communication":
                # Excluding "communication" procedures
                continue

            self.procedures.append(
                Procedure(
                    procedure.row_id,
                    procedure.ordercategoryname,
                    procedure.ordercategorydescription,
                    get_staff_agent(self.prov_bundle, care_giver),
                    get_process_entity(self.prov_bundle, item),
                    procedure.starttime,
                    procedure.endtime,
                    procedure.location,
                    procedure.locationcategory,
                    procedure.value,
                    procedure.valueuom if procedure.valueuom != "None" else None,
                )
            )

    def get_patient_at_admission(self):
        return self.patient.at_admission(self.admission_id)

    def get_patient_at_time(self, time):
        return self.patient.at_admission_time(self.admission_id, time)

    def process(self):
        self.db_session = db.get_session()
        self.db_admission = db.get_admission(self.db_session, self.admission_id)
        self.prepare_procedures()

        db_patient = db.get_patient(self.db_session, self.db_admission.subject_id)
        self.patient = Patient(self.prov_bundle, db_patient)

        for transfer in db.get_transfers(self.db_session, self.admission_id):
            if transfer.eventtype == "discharge":
                # using the patient from the last transfer
                # as they are sorted according to intime, the discharge should be the last transfer
                patient_out.add_asserted_type(ns_type['DischargedPatient'])
                generation.add_asserted_type(EVENT_TYPES[transfer.eventtype])
                break  # no further activity from here

            patient_in = self.get_patient_at_time(transfer.intime)
            patient_out = self.get_patient_at_time(transfer.outtime)
            # patient_out.wasDerivedFrom(patient_in)

            activity_type = ns_type['Treating']
            unit_agent = get_unit_agent(
                self.prov_bundle, transfer.curr_wardid, transfer.curr_careunit
            )
            stay_activity = self.prov_bundle.activity(
                ns_stay[str(transfer.row_id)],
                transfer.intime,
                transfer.outtime,
                [(PROV_TYPE, activity_type), (ns_type["los"], transfer.los)],
            )
            stay_activity.wasAssociatedWith(unit_agent)
            if transfer.icustay_id is not None:
                stay_activity.add_asserted_type(ns_type["IntensiveCare"])

            stay_activity.used(
                patient_in, transfer.intime, [(PROV_TYPE, EVENT_TYPES[transfer.eventtype])]
            )
            generation = self.prov_bundle.wasGeneratedBy(
                patient_out, stay_activity, transfer.outtime
            )

        for procedure in self.procedures:
            proc_attrs = [
                ("prov:type", ns_type["Performing"]),
                ("prov:type", procedure.process.identifier),
                (ns_attrs["category"], procedure.category),
                (ns_attrs["cat_desc"], procedure.cat_description),
            ]
            if procedure.location is not None:
                proc_attrs.append((ns_attrs["location"], procedure.location))
                proc_attrs.append((ns_attrs["location_cat"], procedure.location_cat))
            if procedure.uom is not None:
                proc_attrs.append((ns_attrs["value"], procedure.value))
                proc_attrs.append((ns_attrs["uom"], procedure.uom))
            proc_activity = self.prov_bundle.activity(
                ns_procedures[str(procedure.id)],
                procedure.start_time,
                procedure.end_time,
                proc_attrs,
            )
            patient_start = self.get_patient_at_time(procedure.start_time)
            patient_end = self.get_patient_at_time(procedure.end_time)

            proc_activity.used(patient_start)
            patient_end.wasGeneratedBy(proc_activity)
            proc_activity.wasAssociatedWith(procedure.care_giver, procedure.process)

        # Linking up versions of the patients
        sorted_versions = list(
            version for _, version in sorted(self.patient.versions.items())
        )
        if sorted_versions:  # having at least one
            sorted_versions[0].wasDerivedFrom(self.get_patient_at_admission())
            if len(sorted_versions) > 1:
                for prev, curr in zip(sorted_versions[:-1], sorted_versions[1:]):
                    curr.wasDerivedFrom(prev)


def get_blank_prov_document():
    return ProvDocument(namespaces=all_namespaces)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and generate provenance for a MIMIC patient admission"
    )
    parser.add_argument("admission_id", type=int, help="The ID of admission to process")
    args = parser.parse_args()

    prov_doc1 = ProvDocument(namespaces=all_namespaces)
    admission = Admission(prov_doc1, args.admission_id)
    admission.process()

    filepath = output_path / f"{args.admission_id}.json"
    with filepath.open("w") as f:
        prov_doc1.serialize(f)
    provn_content = prov_doc1.get_provn()
    print(provn_content)
    with filepath.with_suffix(".provn").open("w") as f:
        f.write(provn_content)

    dot = prov_to_dot(prov_doc1)
    dot.write_pdf(filepath.with_suffix(".pdf"))
    db.close_session()
