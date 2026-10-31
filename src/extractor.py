
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from src.config import MODEL_DIR


class JobAttributeExtractor:

    MIN_CONFIDENCE = 0.70

    STOP_WORDS = {
        "and", "or", "the", "a", "an", "of", "in", "for",
        "with", "is", "are", "to", "by", "on", "er"
    }

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

        offset_mapping = encoding.pop("offset_mapping")

        with torch.no_grad():
            outputs = self.model(**encoding)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        predictions = torch.argmax(
            probabilities, dim=-1
        )[0]

        confidences = torch.max(
            probabilities, dim=-1
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

            # Ignore special tokens.
            if start == end:
                continue

            label = self.id2label[int(prediction)]

            entities.append({
                "token": token,
                "label": label,
                "confidence": float(confidence),
                "start": start,
                "end": end
            })

        return entities

    # -----------------------------------------------------
    # CLEAN ENTITY TEXT
    # -----------------------------------------------------

    def clean_value(self, value):
        value = value.strip()

        # Remove punctuation at the edges only.
        value = value.strip(" \t\r\n.,;:!?\"'")

        # Normalize whitespace.
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    # -----------------------------------------------------
    # CHECK ENTITY QUALITY
    # -----------------------------------------------------

    def is_valid_entity(self, entity_type, value, confidence):
        if not value:
            return False

        if confidence < self.MIN_CONFIDENCE:
            return False

        if not any(char.isalnum() for char in value):
            return False

        if entity_type == "SKILL":
            if value.lower() in self.STOP_WORDS:
                return False

            if len(value) < 3:
                return False

        return True

    # -----------------------------------------------------
    # FINALIZE AND SAVE ENTITY
    # -----------------------------------------------------

    def _finalize(self, text, attributes, entity):
        value = text[
            entity["start"]:entity["end"]
        ]

        value = self.clean_value(value)

        if not entity["confidences"]:
            return

        confidence = (
            sum(entity["confidences"])
            / len(entity["confidences"])
        )

        entity_type = entity["type"]

        if not self.is_valid_entity(
            entity_type, value, confidence
        ):
            return

        if entity_type == "SKILL":
            # Split lists of skills into individual values.
            parts = re.split(
                r",|/|;|\band\b|\.",
                value,
                flags=re.IGNORECASE
            )

            cleaned_parts = []

            for part in parts:
                part = self.clean_value(part)

                part = re.sub(
                    r"^(knowledge of|experience with|"
                    r"skills include|required skills include)\s+",
                    "",
                    part,
                    flags=re.IGNORECASE
                )

                part = self.clean_value(part)

                if self.is_valid_entity(
                    "SKILL", part, confidence
                ):
                    cleaned_parts.append(part)

            values_to_save = cleaned_parts

        else:
            values_to_save = [value]

        attributes.setdefault(entity_type, [])

        for final_value in values_to_save:
            already_exists = any(
                item["value"].lower() == final_value.lower()
                for item in attributes[entity_type]
            )

            if not already_exists:
                attributes[entity_type].append({
                    "value": final_value,
                    "confidence": round(confidence, 4)
                })

    # -----------------------------------------------------
    # ADD RULE-BASED WORK-MODE CORRECTIONS
    # -----------------------------------------------------

    def _add_work_mode(self, text, attributes):
        patterns = [
            r"\bhybrid\s+work\s+mode\b",
            r"\bremote\s+work\s+mode\b",
            r"\bon[- ]site\s+work\s+mode\b",
            r"\bwork\s+from\s+home\b",
            r"\bwork\s+from\s+office\b",
            r"\bhybrid\b",
            r"\bremote\b",
            r"\bon[- ]site\b"
        ]

        matches = []

        # Prefer longer phrases, such as "hybrid work mode",
        # over their shorter component words.
        for pattern in patterns:
            for match in re.finditer(
                pattern, text, flags=re.IGNORECASE
            ):
                matches.append({
                    "start": match.start(),
                    "end": match.end(),
                    "value": self.clean_value(match.group(0))
                })

        matches.sort(
            key=lambda item: (
                item["start"],
                -(item["end"] - item["start"])
            )
        )

        selected = []

        for match in matches:
            # Avoid overlapping matches.
            overlaps = any(
                match["start"] < item["end"]
                and match["end"] > item["start"]
                for item in selected
            )

            if not overlaps:
                selected.append(match)

        for match in selected:
            value = match["value"]

            existing = attributes.get("WORK_MODE", [])

            if any(
                item["value"].lower() == value.lower()
                for item in existing
            ):
                continue

            attributes.setdefault("WORK_MODE", []).append({
                "value": value,
                # This is a rule match, not model confidence.
                "confidence": 1.0
            })

    # -----------------------------------------------------
    # COMPLETE EXTRACTION
    # -----------------------------------------------------

    def extract(self, text):
        predictions = self.predict(text)

        attributes = {}
        current = None

        for item in predictions:
            label = item["label"]

            if label == "O" or "-" not in label:
                if current:
                    self._finalize(
                        text, attributes, current
                    )
                    current = None

                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "B":
                if current:
                    self._finalize(
                        text, attributes, current
                    )

                current = {
                    "type": entity_type,
                    "start": item["start"],
                    "end": item["end"],
                    "confidences": [item["confidence"]]
                }

            elif prefix == "I":
                if (
                    current
                    and current["type"] == entity_type
                ):
                    current["end"] = item["end"]
                    current["confidences"].append(
                        item["confidence"]
                    )

                else:
                    # Handle an unexpected I-tag as a new entity.
                    if current:
                        self._finalize(
                            text, attributes, current
                        )

                    current = {
                        "type": entity_type,
                        "start": item["start"],
                        "end": item["end"],
                        "confidences": [item["confidence"]]
                    }

            else:
                if current:
                    self._finalize(
                        text, attributes, current
                    )
                    current = None

        if current:
            self._finalize(text, attributes, current)

        # Add work-mode phrases even when the model assigns
        # them an incorrect entity label.
        self._add_work_mode(text, attributes)

        return attributes
