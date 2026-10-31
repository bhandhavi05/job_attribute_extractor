from pathlib import Path


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# DATA PATHS
# ---------------------------------------------------------

RAW_DATA = BASE_DIR / "data" / "raw" / "large_jobs.csv"

ANNOTATIONS_FILE = (
    BASE_DIR
    / "data"
    / "annotations"
    / "large_annotations.json"
)

BIO_DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "large_bio_dataset.json"
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "job_ner_model"
)

BASE_MODEL = "distilbert-base-uncased"


# ---------------------------------------------------------
# NER LABELS
# ---------------------------------------------------------

LABELS = [
    "O",

    "B-COMPANY",
    "I-COMPANY",

    "B-ROLE",
    "I-ROLE",

    "B-LOCATION",
    "I-LOCATION",

    "B-EDUCATION",
    "I-EDUCATION",

    "B-EXPERIENCE",
    "I-EXPERIENCE",

    "B-SKILL",
    "I-SKILL",

    "B-WORK_MODE",
    "I-WORK_MODE",

    "B-SALARY",
    "I-SALARY",
]


LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(LABELS)
}