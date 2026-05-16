from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fft_features import (
    compute_log_power_spectrum,
    extract_fft_radial_feature,
    load_gray_image,
)


DEFAULT_IMAGE_PATH = Path("outputs/processed_gray/train/REAL/0000.jpg")
DEFAULT_METADATA_PATH = Path("outputs/metadata/train_metadata.csv")
DEFAULT_SAVE_PATH = Path("outputs/features/fft_single_check.png")


def parse_args():
    parser = ArgumentParser(description="Test FFT radial feature extraction on one image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to one processed grayscale image.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Fallback metadata CSV used when --image does not exist.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=f"Save the visualization, for example {DEFAULT_SAVE_PATH}.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window.",
    )
    parser.add_argument(
        "--remove-mean",
        action="store_true",
        help="Subtract the image mean before FFT.",
    )
    return parser.parse_args()


def resolve_image_path(image_path, metadata_path):
    if image_path.exists():
        return image_path

    if metadata_path.exists():
        df = pd.read_csv(metadata_path, usecols=["gray_path"], nrows=1)
        if not df.empty:
            fallback_path = Path(df.loc[0, "gray_path"])
            if fallback_path.exists():
                return fallback_path

    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        f"Pass --image with a valid path under outputs/processed_gray."
    )


def main():
    args = parse_args()
    image_path = resolve_image_path(args.image, args.metadata)

    gray = load_gray_image(image_path)
    log_power, _ = compute_log_power_spectrum(gray, remove_mean=args.remove_mean)
    radial_feature = extract_fft_radial_feature(gray, remove_mean=args.remove_mean)

    print("Image path:", image_path)
    print("Gray image shape:", gray.shape)
    print("Log power spectrum shape:", log_power.shape)
    print("Radial average feature shape:", radial_feature.shape)
    print("Radial average feature:")
    print(radial_feature)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(log_power, cmap="gray")
    plt.title("Log Power Spectrum")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(radial_feature)
    plt.title("Radial Average Power Spectrum")
    plt.xlabel("Radius / Frequency")
    plt.ylabel("Average Log Power")

    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Preview saved to: {args.save}")

    if not args.no_show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
