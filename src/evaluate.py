from datasets import load_dataset
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer
)

from src.config import (
    BIO_DATASET,
    MODEL_DIR
)


print("Loading dataset...")

dataset = load_dataset(
    "json",

    data_files=str(
        BIO_DATASET
    ),

    split="train"
)


print(
    "Total records:",
    len(dataset)
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

model = AutoModelForTokenClassification.from_pretrained(
    str(MODEL_DIR)
)


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

def tokenize_and_align_labels(
    examples
):

    tokenized = tokenizer(

        examples["tokens"],

        is_split_into_words=True,

        truncation=True,

        max_length=512

    )


    all_labels = []


    for i, labels in enumerate(
        examples["ner_tags"]
    ):

        word_ids = tokenized.word_ids(
            batch_index=i
        )

        previous_word = None

        label_ids = []


        for word_id in word_ids:

            if word_id is None:

                label_ids.append(-100)


            elif word_id != previous_word:

                label_ids.append(
                    model.config.label2id[
                        labels[word_id]
                    ]
                )


            else:

                label_ids.append(-100)


            previous_word = word_id


        all_labels.append(
            label_ids
        )


    tokenized[
        "labels"
    ] = all_labels


    return tokenized


print(
    "Tokenizing dataset..."
)

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True
)


# ---------------------------------------------------------
# TEST SET
# ---------------------------------------------------------

split = tokenized_dataset.train_test_split(
    test_size=0.1,
    seed=42
)

test_dataset = split["test"]


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)


predictions = trainer.predict(
    test_dataset
)


predicted_ids = predictions.predictions.argmax(
    axis=-1
)


# ---------------------------------------------------------
# CONVERT LABELS
# ---------------------------------------------------------

true_labels = []
pred_labels = []


for i in range(
    len(test_dataset)
):

    true_sequence = []
    pred_sequence = []


    for true_id, pred_id in zip(

        test_dataset[i]["labels"],

        predicted_ids[i]

    ):

        if true_id == -100:

            continue


        true_sequence.append(
            model.config.id2label[
                true_id
            ]
        )

        pred_sequence.append(
            model.config.id2label[
                int(pred_id)
            ]
        )


    true_labels.append(
        true_sequence
    )

    pred_labels.append(
        pred_sequence
    )


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

precision = precision_score(
    true_labels,
    pred_labels
)

recall = recall_score(
    true_labels,
    pred_labels
)

f1 = f1_score(
    true_labels,
    pred_labels
)


print("\n==========================")
print("MODEL EVALUATION")
print("==========================")

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        true_labels,
        pred_labels
    )
)