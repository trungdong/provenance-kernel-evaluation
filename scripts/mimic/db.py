from typing import Tuple, Sequence
import environs
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text

# Reading local environment variables
ROOT_DIR = environs.Path(__file__).parents[2]
env = environs.Env()
env.read_env(str(ROOT_DIR / ".env"))

# Database initialisation
db_uri = env("MIMIC_DB_URI")
db = create_engine(db_uri)
Base = automap_base()
Base.prepare(autoload_with=db, schema="mimiciii")

CareGiver = Base.classes.caregivers
Admission = Base.classes.admissions
Patient = Base.classes.patients
Item = Base.classes.d_items
DTEvent = Base.classes.datetimeevents
ProcedureEvent = Base.classes.procedureevents_mv
ProcedureICD = Base.classes.procedures_icd
Transfer = Base.classes.transfers
AdmDRGCode = Base.classes.drgcodes
AdmICDCode = Base.classes.diagnoses_icd
ICDDiagnose = Base.classes.d_icd_diagnoses
ICDProcedure = Base.classes.d_icd_procedures


SHARED_DB_SESSION = None  # type: Session


def get_session() -> Session:
    global SHARED_DB_SESSION
    if SHARED_DB_SESSION is None:
        SHARED_DB_SESSION = Session(db)
        SHARED_DB_SESSION.execute(text("SET search_path TO mimiciii"))

    return SHARED_DB_SESSION


def close_session():
    if SHARED_DB_SESSION is not None:
        SHARED_DB_SESSION.close()


def get_patient(session: Session, subject_id: int) -> Patient:
    return session.query(Patient).filter(Patient.subject_id == subject_id).first()


def get_admission(session: Session, admission_id: int) -> Admission:
    return session.query(Admission).filter(Admission.hadm_id == admission_id).first()


def get_transfers(session: Session, admission_id: int) -> Sequence[Transfer]:
    return (
        session.query(Transfer)
        .filter(Transfer.hadm_id == admission_id)
        .order_by(Transfer.intime)
        .all()
    )


def get_procedures_mv(
    session: Session, admission_id: int
) -> Sequence[Tuple[ProcedureEvent, Item, CareGiver]]:
    # Returning the list of procedures done in an admission along with its definition ordered by start time
    return (
        session.query(ProcedureEvent, Item, CareGiver)
        .join(Item, Item.itemid == ProcedureEvent.itemid)
        .join(CareGiver, CareGiver.cgid == ProcedureEvent.cgid)
        .filter(ProcedureEvent.hadm_id == admission_id)
        .order_by(ProcedureEvent.starttime)
        .all()
    )


def get_procedures_icd(
    session: Session, admission_id: int
) -> Sequence[Tuple[ProcedureICD, ICDProcedure]]:
    return (
        session.query(ProcedureICD, ICDProcedure)
        .join(ICDProcedure, ProcedureICD.icd9_code == ICDProcedure.icd9_code)
        .filter(ProcedureICD.hadm_id == admission_id)
        .all()
    )
