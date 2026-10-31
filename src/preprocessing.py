from datasets import load_dataset


DATASET_FILE = "data/processed/large_bio_dataset.json"


def load_data():

    dataset = load_dataset(
        "json",
        data_files=DATASET_FILE,
        split="train"
    )

    return dataset


def split_dataset(dataset):

    # 80% training, 20% temporary
    split = dataset.train_test_split(
        test_size=0.2,
        seed=42
    )

    # Split the 20% temporary data
    # into validation and test
    validation_test = split["test"].train_test_split(
        test_size=0.5,
        seed=42
    )

    return {
        "train": split["train"],
        "validation": validation_test["train"],
        "test": validation_test["test"]
    }


if __name__ == "__main__":

    print("Loading dataset...")

    dataset = load_data()

    print(
        "Total records:",
        len(dataset)
    )

    datasets = split_dataset(dataset)

    print()
    print("Dataset split completed.")
    print("------------------------")

    print(
        "Training:",
        len(datasets["train"])
    )

    print(
        "Validation:",
        len(datasets["validation"])
    )

    print(
        "Testing:",
        len(datasets["test"])
    )