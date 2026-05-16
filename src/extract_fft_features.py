from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fft_features import extract_fft_radial_feature_from_path


META_DIR = Path("outputs/metadata")
FEATURE_DIR = Path("outputs/features")


def limit_metadata(df, max_rows=None, max_per_class=None):
    if max_per_class is not None:
        return (
            df.groupby("class_name", group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True)
        )

    if max_rows is not None:
        return df.head(max_rows).reset_index(drop=True)

    return df


def extract_features_from_metadata(
    csv_path,
    split,
    image_size=(32, 32),
    remove_mean=False,
    max_rows=None,
    max_per_class=None,
):
    """
    Extract FFT radial average log-power features from a metadata CSV.

    Returns:
        X: feature matrix with shape (n_samples, n_features)
        y: label vector with shape (n_samples,)
    """
    df = pd.read_csv(csv_path)
    df = limit_metadata(df, max_rows=max_rows, max_per_class=max_per_class)

    features = []
    labels = []

    rows = df.itertuples(index=False)
    for row in tqdm(rows, total=len(df), desc=f"Extracting FFT features for {split}"):
        feature = extract_fft_radial_feature_from_path(
            row.gray_path,
            image_size=image_size,
            remove_mean=remove_mean,
        )

        features.append(feature)
        labels.append(int(row.label))

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def parse_args():
    parser = ArgumentParser(description="Batch extract CIFAKE FFT radial features.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the first N rows from each metadata CSV.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Use only the first N images per class from each split.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Square image size used before FFT.",
    )
    parser.add_argument(
        "--remove-mean",
        action="store_true",
        help="Subtract each image mean before FFT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_size = (args.image_size, args.image_size)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = META_DIR / "train_metadata.csv"
    test_csv = META_DIR / "test_metadata.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train metadata not found: {train_csv}")

    if not test_csv.exists():
        raise FileNotFoundError(f"Test metadata not found: {test_csv}")

    X_train_fft, y_train = extract_features_from_metadata(
        train_csv,
        split="train",
        image_size=image_size,
        remove_mean=args.remove_mean,
        max_rows=args.max_rows,
        max_per_class=args.max_per_class,
    )
    X_test_fft, y_test = extract_features_from_metadata(
        test_csv,
        split="test",
        image_size=image_size,
        remove_mean=args.remove_mean,
        max_rows=args.max_rows,
        max_per_class=args.max_per_class,
    )

    np.save(FEATURE_DIR / "X_train_fft.npy", X_train_fft)
    np.save(FEATURE_DIR / "y_train.npy", y_train)
    np.save(FEATURE_DIR / "X_test_fft.npy", X_test_fft)
    np.save(FEATURE_DIR / "y_test.npy", y_test)

    print("FFT feature extraction finished.")
    print("X_train_fft shape:", X_train_fft.shape)
    print("y_train shape:", y_train.shape)
    print("X_test_fft shape:", X_test_fft.shape)
    print("y_test shape:", y_test.shape)

    print("\nFeature files saved to:")
    print(FEATURE_DIR / "X_train_fft.npy")
    print(FEATURE_DIR / "y_train.npy")
    print(FEATURE_DIR / "X_test_fft.npy")
    print(FEATURE_DIR / "y_test.npy")


if __name__ == "__main__":
    main()
