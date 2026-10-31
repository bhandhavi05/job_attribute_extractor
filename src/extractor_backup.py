import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from src.config import MODEL_DIR


class JobAttributeExtractor:

    def __init__(self):

        print("Loading trained model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR)
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            str(MODEL_DIR)
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

        print("Model loaded successfully.")


    # -----------------------------------------------------
    # PREDICT TOKENS
    # -----------------------------------------------------

    def predict(self, text):

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )

        offset_mapping = encoding.pop(
            "offset_mapping"
        )

        with torch.no_grad():

            outputs = self.model(
                **encoding
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )[0]

        confidences = torch.max(
            probabilities,
            dim=-1
        ).values[0]


        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0]
        )


        entities = []

        for token, prediction, confidence, offset in zip(
            tokens,
            predictions,
            confidences,
            offset_mapping[0]
        ):

            start = int(offset[0])
            end = int(offset[1])

            # Ignore special tokens
            if start == end:
                continue

            label = self.id2label[
                int(prediction)
            ]

            entities.append({
                "token": token,
                "label": label,
                "confidence": float(confidence),
                "start": start,
                "end": end
            })

        return entities


    # -----------------------------------------------------
    # MERGE BIO TOKENS
    # -----------------------------------------------------

    def merge_entities(
        self,
        text,
        predictions
    ):

        attributes = {}

        current_entity = None


        for item in predictions:

            label = item["label"]

            if label == "O":

                if current_entity:
                    self._save_entity(
                        attributes,
                        current_entity
                    )

                    current_entity = None

                continue


            prefix, entity_type = label.split(
                "-",
                1
            )


            # ---------------------------------------------
            # Start new entity
            # ---------------------------------------------

            if prefix == "B":

                if current_entity:

                    self._save_entity(
                        attributes,
                        current_entity
                    )


                current_entity = {
                    "type": entity_type,
                    "start": item["start"],
                    "end": item["end"],
                    "confidence": [
                        item["confidence"]
                    ]
                }


            # ---------------------------------------------
            # Continue entity
            # ---------------------------------------------

            elif prefix == "I":

                if (
                    current_entity
                    and current_entity["type"]
                    == entity_type
                ):

                    current_entity["end"] = item["end"]

                    current_entity[
                        "confidence"
                    ].append(
                        item["confidence"]
                    )

                else:

                    # Handle an unexpected I- tag
                    current_entity = {
                        "type": entity_type,
                        "start": item["start"],
                        "end": item["end"],
                        "confidence": [
                            item["confidence"]
                        ]
                    }


        # Save final entity

        if current_entity:

            self._save_entity(
                attributes,
                current_entity
            )


        return attributes


    # -----------------------------------------------------
    # SAVE ENTITY
    # -----------------------------------------------------

    def _save_entity(
        self,
        attributes,
        entity
    ):

        entity_type = entity["type"]

        value = entity["text"] if "text" in entity else None

        confidence = sum(
            entity["confidence"]
        ) / len(
            entity["confidence"]
        )


        # Text will be filled separately
        if value is None:

            attributes.setdefault(
                entity_type,
                []
            )

            attributes[
                entity_type
            ].append({
                "start": entity["start"],
                "end": entity["end"],
                "confidence": round(
                    confidence,
                    4
                )
            })


    # -----------------------------------------------------
    # COMPLETE EXTRACTION
    # -----------------------------------------------------

    def extract(self, text):

        predictions = self.predict(
            text
        )


        attributes = {}

        current = None


        for item in predictions:

            label = item["label"]


            if label == "O":

                if current:

                    self._finalize(
                        text,
                        attributes,
                        current
                    )

                    current = None

                continue


            prefix, entity_type = label.split(
                "-",
                1
            )


            if prefix == "B":

                if current:

                    self._finalize(
                        text,
                        attributes,
                        current
                    )


                current = {
                    "type": entity_type,
                    "start": item["start"],
                    "end": item["end"],
                    "confidences": [
                        item["confidence"]
                    ]
                }


            elif prefix == "I":

                if (
                    current
                    and current["type"]
                    == entity_type
                ):

                    current["end"] = item["end"]

                    current[
                        "confidences"
                    ].append(
                        item["confidence"]
                    )

                else:

                    current = {
                        "type": entity_type,
                        "start": item["start"],
                        "end": item["end"],
                        "confidences": [
                            item["confidence"]
                        ]
                    }


        if current:

            self._finalize(
                text,
                attributes,
                current
            )


        return attributes


    # -----------------------------------------------------
    # FINALIZE ENTITY
    # -----------------------------------------------------

    def _finalize(
        self,
        text,
        attributes,
        entity
    ):

        value = text[
            entity["start"]:
            entity["end"]
        ].strip()


        if not value:

            return


        confidence = (
            sum(entity["confidences"])
            /
            len(entity["confidences"])
        )


        entity_type = entity["type"]


        attributes.setdefault(
            entity_type,
            []
        )


        attributes[
            entity_type
        ].append({

            "value": value,

            "confidence": round(
                confidence,
                4
            )

        })
        