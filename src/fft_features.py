from pathlib import Path

import numpy as np
from PIL import Image


def load_gray_image(image_path, image_size=(32, 32)):
    """
    Load an image as a normalized grayscale numpy array.

    Returns an array with shape (H, W) and values in [0, 1].
    """
    with Image.open(Path(image_path)) as image:
        image = image.convert("L").resize(image_size)
        gray = np.asarray(image, dtype=np.float32)

    return gray / 255.0


def compute_log_power_spectrum(gray, remove_mean=False):
    """
    Compute the centered log power spectrum.

    Pipeline:
        grayscale image -> 2D FFT -> fftshift -> power spectrum -> log power
    """
    gray = np.asarray(gray, dtype=np.float32)

    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {gray.shape}")

    if remove_mean:
        gray = gray - np.mean(gray)

    fft_result = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_result)
    power_spectrum = np.abs(fft_shift) ** 2
    log_power = np.log1p(power_spectrum)

    return log_power, fft_shift


def radial_average_power_spectrum(log_power):
    """
    Convert a 2D log power spectrum into a 1D radial average profile.
    """
    log_power = np.asarray(log_power)

    if log_power.ndim != 2:
        raise ValueError(f"Expected a 2D log power spectrum, got shape {log_power.shape}")

    h, w = log_power.shape
    center_y = h // 2
    center_x = w // 2

    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(np.int32)

    radial_sum = np.bincount(r.ravel(), weights=log_power.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / np.maximum(radial_count, 1)

    return radial_profile.astype(np.float32)


def extract_fft_radial_feature(gray, remove_mean=False):
    """
    Extract a 1D FFT radial average log-power feature from a grayscale image.
    """
    log_power, _ = compute_log_power_spectrum(gray, remove_mean=remove_mean)
    return radial_average_power_spectrum(log_power)


def extract_fft_radial_feature_from_path(
    image_path,
    image_size=(32, 32),
    remove_mean=False,
):
    """
    Load an image path and extract its FFT radial average log-power feature.
    """
    gray = load_gray_image(image_path, image_size=image_size)
    return extract_fft_radial_feature(gray, remove_mean=remove_mean)
