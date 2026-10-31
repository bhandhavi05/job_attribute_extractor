import csv
import json
import re


INPUT_FILE = "data/raw/large_jobs.csv"
OUTPUT_FILE = "data/annotations/large_annotations.json"


SKILLS = [
    "Machine Learning",
    "Spring Boot",
    "Scikit-learn",
    "JavaScript",
    "PostgreSQL",
    "TensorFlow",
    "Power BI",
    "PyTorch",
    "Node.js",
    "MongoDB",
    "NumPy",
    "Pandas",
    "Python",
    "Django",
    "Java",
    "MySQL",
    "React",
    "Flask",
    "FastAPI",
    "SQL",
    "Spring",
    "Oracle"
]


EDUCATION_VALUES = [
    "B.Tech",
    "M.Tech",
    "B.E",
    "MCA",
    "B.Sc"
]


WORK_MODES = [
    "remote",
    "hybrid",
    "on-site"
]


def overlaps(start, end, entities):
    """
    Check whether this entity overlaps
    with an already existing entity.
    """

    for entity in entities:

        existing_start = entity["start"]
        existing_end = entity["end"]

        if start < existing_end and end > existing_start:
            return True

    return False


def add_entity(text, value, label, entities):

    pattern = re.escape(value)

    for match in re.finditer(
        pattern,
        text,
        flags=re.IGNORECASE
    ):

        start = match.start()
        end = match.end()

        if not overlaps(start, end, entities):

            entities.append({
                "start": start,
                "end": end,
                "label": label
            })

            return


def main():

    annotations = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            text = row["job_description"]

            entities = []

            # --------------------------------
            # COMPANY
            # --------------------------------

            match = re.search(
                r"^(.*?) is hiring|^(.*?) is looking",
                text
            )

            if match:

                company = (
                    match.group(1)
                    if match.group(1)
                    else match.group(2)
                )

                add_entity(
                    text,
                    company,
                    "COMPANY",
                    entities
                )

            # --------------------------------
            # ROLE
            # --------------------------------

            match = re.search(
                r"(?:hiring a|looking for a) (.*?) in ",
                text
            )

            if match:

                role = match.group(1)

                add_entity(
                    text,
                    role,
                    "ROLE",
                    entities
                )

            # --------------------------------
            # LOCATION
            # --------------------------------

            match = re.search(
                r" in ([A-Za-z ]+)\. Candidates",
                text
            )

            if match:

                location = match.group(1)

                add_entity(
                    text,
                    location,
                    "LOCATION",
                    entities
                )

            # --------------------------------
            # EDUCATION
            # --------------------------------

            for education in EDUCATION_VALUES:

                add_entity(
                    text,
                    education,
                    "EDUCATION",
                    entities
                )

            # --------------------------------
            # EXPERIENCE
            # --------------------------------

            match = re.search(
                r"\d+(?:-\d+|\+)? years",
                text
            )

            if match:

                experience = match.group()

                add_entity(
                    text,
                    experience,
                    "EXPERIENCE",
                    entities
                )

            # --------------------------------
            # SKILLS
            # --------------------------------

            # Longer skills first.
            # This prevents Java from matching
            # inside JavaScript.

            sorted_skills = sorted(
                SKILLS,
                key=len,
                reverse=True
            )

            for skill in sorted_skills:

                # Word-boundary matching
                pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

                for match in re.finditer(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                ):

                    start = match.start()
                    end = match.end()

                    if not overlaps(
                        start,
                        end,
                        entities
                    ):

                        entities.append({
                            "start": start,
                            "end": end,
                            "label": "SKILL"
                        })

            # --------------------------------
            # WORK MODE
            # --------------------------------

            for mode in WORK_MODES:

                add_entity(
                    text,
                    mode,
                    "WORK_MODE",
                    entities
                )

            # --------------------------------
            # SALARY
            # --------------------------------

            match = re.search(
                r"\d+-\d+ LPA",
                text
            )

            if match:

                salary = match.group()

                add_entity(
                    text,
                    salary,
                    "SALARY",
                    entities
                )

            # Sort entities by start position

            entities.sort(
                key=lambda x: x["start"]
            )

            annotations.append({
                "text": text,
                "entities": entities
            })

    # --------------------------------
    # SAVE
    # --------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            annotations,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("Annotations generated successfully.")
    print("Total records:", len(annotations))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()