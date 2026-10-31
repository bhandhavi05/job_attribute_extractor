import json
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/extract"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title=
        "Job Description Attribute Extractor",

    page_icon="💼",

    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "💼 Job Description Attribute Extractor"
)

st.write(
    """
    Extract structured information from an
    unstructured job description using a
    Transformer-based NER model.
    """
)


# =========================================================
# INPUT
# =========================================================

job_description = st.text_area(

    "📄 Paste Job Description",

    height=300,

    placeholder=(
        "Example:\n\n"
        "Google is hiring a Software Engineer "
        "in Hyderabad. Candidates should have "
        "a B.Tech degree and 2 years of experience "
        "with Python, Java and AWS. "
        "Salary is 12-18 LPA."
    )
)


# =========================================================
# EXTRACTION
# =========================================================

if st.button(
    "🔍 Extract Attributes",
    type="primary"
):

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

    else:

        with st.spinner(
            "Analyzing job description..."
        ):

            try:

                response = requests.post(

                    API_URL,

                    json={
                        "text":
                            job_description
                    },

                    timeout=120
                )


                if response.status_code != 200:

                    st.error(
                        "Backend error."
                    )

                    st.code(
                        response.text
                    )

                    st.stop()


                result = response.json()

                attributes = result[
                    "attributes"
                ]


                # =========================================
                # SUCCESS
                # =========================================

                st.success(
                    "Extraction completed successfully!"
                )


                # =========================================
                # VALIDATION
                # =========================================

                validation = result.get(
                    "validation",
                    {}
                )


                if validation:

                    score = validation.get(
                        "completeness_score",
                        0
                    )


                    st.metric(
                        "Completeness Score",
                        f"{score}%"
                    )


                    missing = validation.get(
                        "missing_attributes",
                        []
                    )


                    if missing:

                        st.warning(
                            "Missing attributes: "
                            + ", ".join(missing)
                        )

                    else:

                        st.success(
                            "All important attributes detected."
                        )


                # =========================================
                # BASIC INFORMATION
                # =========================================

                st.subheader(
                    "📋 Basic Information"
                )


                col1, col2 = st.columns(2)


                def display_entity(
                    entity_name
                ):

                    entities = attributes.get(
                        entity_name,
                        []
                    )

                    if not entities:

                        return "Not detected"

                    return entities[0]["value"]


                with col1:

                    st.write(
                        "**Company:**",
                        display_entity(
                            "COMPANY"
                        )
                    )

                    st.write(
                        "**Job Role:**",
                        display_entity(
                            "ROLE"
                        )
                    )

                    st.write(
                        "**Education:**",
                        display_entity(
                            "EDUCATION"
                        )
                    )

                    st.write(
                        "**Experience:**",
                        display_entity(
                            "EXPERIENCE"
                        )
                    )


                with col2:

                    st.write(
                        "**Location:**",
                        display_entity(
                            "LOCATION"
                        )
                    )

                    st.write(
                        "**Salary:**",
                        display_entity(
                            "SALARY"
                        )
                    )

                    st.write(
                        "**Work Mode:**",
                        display_entity(
                            "WORK_MODE"
                        )
                    )


                # =========================================
                # SKILLS
                # =========================================

                st.subheader(
                    "🛠️ Skills"
                )


                skills = attributes.get(
                    "SKILL",
                    []
                )


                if skills:

                    for skill in skills:

                        confidence = (
                            skill["confidence"]
                            * 100
                        )

                        st.write(
                            f"• **{skill['value']}** "
                            f"— {confidence:.2f}%"
                        )

                else:

                    st.write(
                        "No skills detected."
                    )


                # =========================================
                # DETAILED ENTITY TABLE
                # =========================================

                st.subheader(
                    "🔎 Detailed Extraction"
                )


                for entity_type, entities in (
                    attributes.items()
                ):

                    st.markdown(
                        f"### {entity_type}"
                    )


                    for entity in entities:

                        confidence = (
                            entity["confidence"]
                            * 100
                        )


                        st.write(
                            f"**{entity['value']}** "
                            f"— Confidence: "
                            f"{confidence:.2f}%"
                        )


                # =========================================
                # JSON OUTPUT
                # =========================================

                st.subheader(
                    "📄 JSON Output"
                )


                st.json(
                    result
                )


                # =========================================
                # DOWNLOAD
                # =========================================

                json_data = json.dumps(

                    result,

                    indent=4,

                    ensure_ascii=False

                )


                st.download_button(

                    label=
                        "⬇️ Download JSON",

                    data=json_data,

                    file_name=
                        "job_attributes.json",

                    mime=
                        "application/json"

                )


            except requests.exceptions.ConnectionError:

                st.error(
                    """
                    ❌ Cannot connect to the FastAPI backend.

                    Make sure the backend is running on:

                    http://127.0.0.1:8000
                    """
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )