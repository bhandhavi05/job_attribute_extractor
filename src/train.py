import json
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import classification_report


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FILE = "data/processed/large_bio_dataset.json"

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "models/job_ner_model"

MAX_LENGTH = 256


# ==========================================
# LABELS
# ==========================================

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


label2id = {
    label: i
    for i, label in enumerate(LABELS)
}


id2label = {
    i: label
    for i, label in enumerate(LABELS)
}


print("Number of labels:", len(LABELS))

print("Labels:")
for label, index in label2id.items():
    print(index, "->", label)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_FILE,
    split="train"
)

print("Total records:", len(dataset))


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

validation_test = split["test"].train_test_split(
    test_size=0.5,
    seed=42
)

train_dataset = split["train"]

validation_dataset = validation_test["train"]

test_dataset = validation_test["test"]


print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))


# ==========================================
# TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# TOKENIZE + ALIGN LABELS
# ==========================================

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True
    )

    all_labels = []

    for batch_index, labels in enumerate(
        examples["ner_tags"]
    ):

        word_ids = tokenized_inputs.word_ids(
            batch_index=batch_index
        )

        previous_word_id = None

        label_ids = []

        for word_id in word_ids:

            # Special token
            if word_id is None:

                label_ids.append(-100)

            # First subword of a word
            elif word_id != previous_word_id:

                label_ids.append(
                    label2id[labels[word_id]]
                )

            # Additional subword
            else:

                label = labels[word_id]

                if label.startswith("B-"):

                    label = "I-" + label[2:]

                label_ids.append(
                    label2id[label]
                )

            previous_word_id = word_id

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


print("\nTokenizing dataset...")

tokenized_train = train_dataset.map(
    tokenize_and_align_labels,
    batched=True
)

tokenized_validation = validation_dataset.map(
    tokenize_and_align_labels,
    batched=True
)

tokenized_test = test_dataset.map(
    tokenize_and_align_labels,
    batched=True
)


print("Tokenization completed.")


# ==========================================
# MODEL
# ==========================================

print("\nLoading model...")

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)


# ==========================================
# DATA COLLATOR
# ==========================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ==========================================
# METRICS
# ==========================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []
        current_labels = []

        for pred, lab in zip(
            prediction,
            label
        ):

            if lab == -100:
                continue

            current_predictions.append(
                id2label[pred]
            )

            current_labels.append(
                id2label[lab]
            )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    report = classification_report(
        true_labels,
        true_predictions,
        output_dict=True
    )

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }


# ==========================================
# TRAINING ARGUMENTS
# ==========================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    save_total_limit=2
)


# ==========================================
# TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ==========================================
# TRAIN
# ==========================================

print("\nStarting model training...")

trainer.train()


# ==========================================
# SAVE MODEL
# ==========================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ==========================================
# FINAL TEST
# ==========================================

print("\nEvaluating on test dataset...")

results = trainer.evaluate(
    tokenized_test
)

print("\nTest Results:")

for key, value in results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


print("\nTraining completed successfully.")
print("Model saved to:", OUTPUT_DIR)