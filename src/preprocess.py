from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


RAW_DIR = Path("data/raw")
GRAY_DIR = Path("outputs/processed_gray")
META_DIR = Path("outputs/metadata")

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1,
}

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")
DEFAULT_TRAIN_MAX_PER_CLASS = 100
DEFAULT_TEST_MAX_PER_CLASS = 50


def collect_image_paths(split: str, max_per_class: int | None = None):
    """
    Collect image paths and labels from the train or test split.
    """
    records = []

    for class_name, label in LABEL_MAP.items():
        class_dir = RAW_DIR / split / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {class_dir}")

        image_paths = []
        for pattern in IMAGE_PATTERNS:
            image_paths.extend(class_dir.glob(pattern))
        image_paths = sorted(image_paths)

        if max_per_class is not None:
            image_paths = image_paths[:max_per_class]

        for img_path in image_paths:
            records.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "label": label,
                    "image_path": img_path.as_posix(),
                }
            )

    return records


def convert_to_gray_and_save(record):
    """
    Convert one RGB image to 32x32 grayscale and save it.
    """
    img_path = Path(record["image_path"])
    split = record["split"]
    class_name = record["class_name"]

    save_dir = GRAY_DIR / split / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / img_path.name

    with Image.open(img_path) as image:
        gray_image = image.convert("L").resize((32, 32))
        gray_image.save(save_path)

    new_record = record.copy()
    new_record["gray_path"] = save_path.as_posix()
    new_record["width"] = 32
    new_record["height"] = 32

    return new_record


def preprocess_split(split: str, max_per_class: int | None = None):
    """
    Process the train or test split and save its metadata CSV.
    """
    print(f"Processing {split} dataset...")

    records = collect_image_paths(split, max_per_class=max_per_class)
    processed_records = []

    for record in tqdm(records, desc=split):
        processed_record = convert_to_gray_and_save(record)
        processed_records.append(processed_record)

    df = pd.DataFrame(processed_records)

    META_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = META_DIR / f"{split}_metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"{split} preprocessing finished")
    print(f"Sample count: {len(df)}")
    print(f"Metadata saved to: {csv_path}")

    return df


def parse_args():
    parser = ArgumentParser(description="Preprocess CIFAKE images to grayscale.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process the full dataset instead of the default debug subset.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit each class to this many images for both train and test.",
    )
    parser.add_argument(
        "--train-max-per-class",
        type=int,
        default=DEFAULT_TRAIN_MAX_PER_CLASS,
        help="Debug limit per class for train. Ignored when --full is used.",
    )
    parser.add_argument(
        "--test-max-per-class",
        type=int,
        default=DEFAULT_TEST_MAX_PER_CLASS,
        help="Debug limit per class for test. Ignored when --full is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.full:
        train_limit = None
        test_limit = None
    elif args.max_per_class is not None:
        train_limit = args.max_per_class
        test_limit = args.max_per_class
    else:
        train_limit = args.train_max_per_class
        test_limit = args.test_max_per_class

    print("Label mapping:")
    for class_name, label in LABEL_MAP.items():
        print(f"{class_name} -> {label}")

    if train_limit is not None or test_limit is not None:
        print(
            "Debug subset limits: "
            f"train={train_limit or 'all'} per class, "
            f"test={test_limit or 'all'} per class"
        )
        print("Use --full after debugging to process the complete dataset.")

    train_df = preprocess_split("train", max_per_class=train_limit)
    test_df = preprocess_split("test", max_per_class=test_limit)

    print("\nTrain class counts:")
    print(train_df["class_name"].value_counts())

    print("\nTest class counts:")
    print(test_df["class_name"].value_counts())


if __name__ == "__main__":
    main()
