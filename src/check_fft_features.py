from argparse import ArgumentParser
from pathlib import Path

import numpy as np


FEATURE_DIR = Path("outputs/features")


def parse_args():
    parser = ArgumentParser(description="Check generated FFT feature arrays.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_DIR,
        help="Directory containing FFT .npy feature files.",
    )
    return parser.parse_args()


def label_counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def main():
    args = parse_args()

    X_train = np.load(args.feature_dir / "X_train_fft.npy")
    y_train = np.load(args.feature_dir / "y_train.npy")
    X_test = np.load(args.feature_dir / "X_test_fft.npy")
    y_test = np.load(args.feature_dir / "y_test.npy")

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nTrain label counts:", label_counts(y_train))
    print("Test label counts:", label_counts(y_test))

    print("\nFirst 5 train labels:")
    print(y_train[:5])

    print("\nFirst image FFT radial feature:")
    print(X_train[0])

    print("\nX_train has NaN:", np.isnan(X_train).any())
    print("X_train has Inf:", np.isinf(X_train).any())
    print("X_test has NaN:", np.isnan(X_test).any())
    print("X_test has Inf:", np.isinf(X_test).any())


if __name__ == "__main__":
    main()
