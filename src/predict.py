from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)


MODEL_PATH = "models/job_ner_model"


print("Loading trained model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH
)


ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)


def clean_entity(value, label):

    value = value.strip()

    # Remove punctuation from beginning/end
    value = value.strip(".,;:!?")

    # Clean common connector words accidentally included
    if label == "SKILL":

        value = value.rstrip(".,;:")

        if value.lower().endswith(" and"):
            value = value[:-4].strip()

        if value.lower().endswith(" or"):
            value = value[:-3].strip()

    # Normalize work mode
    if label == "WORK_MODE":

        value = value.lower()

    # Normalize salary
    if label == "SALARY":

        value = value.lower()

        value = value.replace(
            " - ",
            "-"
        )

        value = value.replace(
            " ",
            ""
        )

        value = value.replace(
            "lpa.",
            "LPA"
        )

        value = value.replace(
            "lpa",
            "LPA"
        )

    return value


def extract_attributes(text):

    results = ner_pipeline(text)

    extracted = {}

    for entity in results:

        label = entity["entity_group"]

        word = entity["word"]

        score = entity["score"]

        word = clean_entity(
            word,
            label
        )

        if not word:
            continue

        if label not in extracted:

            extracted[label] = []

        extracted[label].append({

            "value": word,

            "confidence": round(
                float(score),
                4
            )
        })

    return extracted


if __name__ == "__main__":
    text = """
TechVision Pvt Ltd is looking for a Machine
Learning Engineer in Bangalore. Applicants should
have an M.Tech degree and 3 years of experience
with Python, TensorFlow and PyTorch. The position
is remote with a salary of 10-15 LPA.
"""

    print("\nJob Description:")
    print(text)

    print("\nExtracted Attributes:")
    print("----------------------")

    results = extract_attributes(text)

    for label, values in results.items():

        print(f"\n{label}:")

        for item in values:

            print(
                f"  {item['value']} "
                f"(confidence={item['confidence']})"
            )