# CIFAKE Frequency Feature Pipeline

This repository contains member 1's deliverables for the digital image
processing project: data preprocessing and FFT radial average power spectrum
feature extraction.

## Completed Work

- Converted CIFAKE images to 32x32 grayscale images.
- Fixed labels for the whole project:
  - `REAL = 0`
  - `FAKE = 1`
- Generated metadata CSV files:
  - `outputs/metadata/train_metadata.csv`
  - `outputs/metadata/test_metadata.csv`
- Extracted FFT radial average log-power spectrum features:
  - `outputs/features/X_train_fft.npy`
  - `outputs/features/y_train.npy`
  - `outputs/features/X_test_fft.npy`
  - `outputs/features/y_test.npy`

## Final Feature Shapes

```text
X_train_fft shape: (100000, 23)
y_train shape: (100000,)
X_test_fft shape: (20000, 23)
y_test shape: (20000,)
```

Label counts:

```text
Train: REAL 50000, FAKE 50000
Test: REAL 10000, FAKE 10000
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Reproduce Preprocessing

Place the CIFAKE dataset under:

```text
data/raw/train/REAL/
data/raw/train/FAKE/
data/raw/test/REAL/
data/raw/test/FAKE/
```

Run a small debug preprocessing pass:

```powershell
python src/preprocess.py
```

Run full preprocessing:

```powershell
python src/preprocess.py --full
```

Check grayscale examples:

```powershell
python src/check_gray.py --save outputs/metadata/gray_check.png --no-show
```

## Reproduce FFT Feature Extraction

Test one image:

```powershell
python src/test_fft_single.py --save outputs/features/fft_single_check.png --no-show
```

Extract a small debug subset:

```powershell
python src/extract_fft_features.py --max-per-class 100
```

Extract full FFT features:

```powershell
python src/extract_fft_features.py
```

Check generated features:

```powershell
python src/check_fft_features.py
```

## Handoff To Member 2

Member 2 can directly load member 1's FFT features:

```python
import numpy as np

X_train_fft = np.load("outputs/features/X_train_fft.npy")
y_train = np.load("outputs/features/y_train.npy")
X_test_fft = np.load("outputs/features/X_test_fft.npy")
y_test = np.load("outputs/features/y_test.npy")
```

For LBP extraction, member 2 should either use the same local CIFAKE data and
run `python src/preprocess.py --full`, or receive the large
`outputs/processed_gray/` folder through a shared drive. The raw images and
processed grayscale images are intentionally excluded from Git because they are
large generated data.

## Git Notes

The GitHub repository should contain source code, metadata CSV files, and FFT
feature `.npy` files. It should not contain the raw CIFAKE image folders or the
downloaded zip archive.
