import json
import re


INPUT_FILE = "data/annotations/large_annotations.json"
OUTPUT_FILE = "data/processed/large_bio_dataset.json"


def tokenize_with_offsets(text):
    """
    Tokenize text while keeping the character
    start and end positions of every token.
    """

    tokens = []

    for match in re.finditer(r"\S+", text):

        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


def convert_record(record):

    text = record["text"]
    entities = record["entities"]

    tokens = tokenize_with_offsets(text)

    words = []
    labels = []

    for token in tokens:

        token_start = token["start"]
        token_end = token["end"]

        words.append(token["token"])

        label = "O"

        for entity in entities:

            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_label = entity["label"]

            # Check whether token overlaps entity
            if (
                token_start < entity_end
                and token_end > entity_start
            ):

                # First token of entity
                if token_start <= entity_start:

                    label = "B-" + entity_label

                else:

                    label = "I-" + entity_label

                break

        labels.append(label)

    return {
        "tokens": words,
        "ner_tags": labels
    }


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        annotations = json.load(file)

    dataset = []

    for record in annotations:

        converted = convert_record(record)

        dataset.append(converted)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("BIO dataset created successfully.")
    print("Total records:", len(dataset))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()