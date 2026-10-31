from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    JobDescriptionRequest,
    ExtractionResponse
)

from src.extractor import JobAttributeExtractor


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Job Description Attribute Extractor",
    description=(
        "Extract structured job attributes "
        "using Transformer-based NER."
    ),
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================================================
# LOAD MODEL ONCE
# =========================================================

extractor = JobAttributeExtractor()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message":
            "Job Description Attribute Extractor API",

        "status":
            "running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "model":
            "job_ner_model"
    }


# =========================================================
# EXTRACTION API
# =========================================================

@app.post(
    "/extract",
    response_model=ExtractionResponse
)
def extract(
    request: JobDescriptionRequest
):

    text = request.text.strip()


    if not text:

        raise HTTPException(
            status_code=400,

            detail=(
                "Job description "
                "cannot be empty."
            )
        )


    try:

        attributes = extractor.extract(
            text
        )


        return {

            "success": True,

            "attributes": attributes

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,

            detail=str(e)
        )
def validate_attributes(attributes):

    required_attributes = [
        "COMPANY",
        "ROLE",
        "LOCATION",
        "EDUCATION",
        "EXPERIENCE"
    ]

    missing = []

    for attribute in required_attributes:

        if (
            attribute not in attributes
            or not attributes[attribute]
        ):

            missing.append(
                attribute
            )


    total = len(
        required_attributes
    )

    present = total - len(
        missing
    )


    completeness = (
        present / total
    ) * 100


    return {
        "missing_attributes": missing,

        "completeness_score":
            round(
                completeness,
                2
            )
    }