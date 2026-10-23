"""Classical baseline: hand-crafted image features and logistic regression.

Each tray image is reduced to a short feature vector:

- an HSV colour histogram (which colours are present, and how much of each),
  which responds to exposed meat, packaging film and tray colours; and
- the mean edge strength in each cell of a coarse grid (where strong edges
  are), which responds to wrinkles, folds and lifted film edges.

A logistic regression maps the features to a defect probability per tray.
The features describe the whole image and say nothing about where a defect
is, so the baseline can only answer the tray-level pass/fail question.

Cost per image is dominated by resizing, colour conversion and the Sobel
filters on a 320x240 image: of the order of 10 million arithmetic operations,
several hundred times fewer than the YOLO detector.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from packinspect.data import IMAGE_EXTENSIONS, read_yolo_labels
from packinspect.evaluate import roc_auc

WORK_SIZE = (320, 240)   # (width, height) images are shrunk to before computing features
HSV_BINS = (8, 4, 4)     # histogram bins for hue, saturation and value
GRID = (4, 4)            # (rows, columns) of cells for the edge features
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)


def image_features(image_bgr, hsv_bins=HSV_BINS, grid=GRID, work_size=WORK_SIZE):
    """Compute the colour and edge feature vector for one image.

    Parameters
    ----------
    image_bgr : numpy.ndarray
        Image as returned by ``cv2.imread``: shape (height, width, 3),
        dtype uint8, channels in BGR order.
    hsv_bins : tuple of int, optional
        Number of histogram bins for hue, saturation and value.
    grid : tuple of int, optional
        Rows and columns of the grid used for the edge features.
    work_size : tuple of int, optional
        ``(width, height)`` the image is resized to first, which makes the
        features independent of the original resolution and cheaper to compute.

    Returns
    -------
    numpy.ndarray
        1-D float array of length ``prod(hsv_bins) + rows * columns``
        (144 with the defaults). The histogram part sums to 1.

    Raises
    ------
    ValueError
        If ``image_bgr`` is not a 3-channel uint8 image.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8):
        raise ValueError("expected a 3-channel uint8 BGR image")
    small = cv2.resize(image_bgr, work_size, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # OpenCV stores 8-bit hue in [0, 180), saturation and value in [0, 256)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(hsv_bins), [0, 180, 0, 256, 0, 256])
    hist = hist.ravel() / hist.sum()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    rows, cols = grid
    h, w = magnitude.shape
    # Crop to a whole number of cells, then average within each cell
    cells = magnitude[: h - h % rows, : w - w % cols].reshape(rows, h // rows, cols, w // cols)
    edges = cells.mean(axis=(1, 3)).ravel()

    return np.concatenate([hist, edges]).astype(np.float64)


def load_split(split_dir, class_names):
    """Compute features and tray-level labels for every image in a split.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Split folder containing ``images/`` and ``labels/``.
    class_names : list of str
        Class names, indexed by class id.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix, one row per image.
    meta : pandas.DataFrame
        One row per image with ``image``, ``gt_defective`` (1 if the tray
        has any labelled defect) and ``gt_classes`` (space-separated names).

    Raises
    ------
    FileNotFoundError
        If the split has no images.
    ValueError
        If an image cannot be read.
    """
    split_dir = Path(split_dir)
    features, rows = [], []
    for image_path in sorted((split_dir / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        features.append(image_features(image))
        boxes = read_yolo_labels(split_dir / "labels" / f"{image_path.stem}.txt", len(class_names))
        gt = sorted({b[0] for b in boxes})
        rows.append({"image": image_path.name, "gt_defective": int(bool(gt)),
                     "gt_classes": " ".join(class_names[c] for c in gt)})
    if not rows:
        raise FileNotFoundError(f"No images found in {split_dir / 'images'}")
    return np.vstack(features), pd.DataFrame(rows)


def fit_baseline(X_train, y_train, X_valid, y_valid, C_values=C_VALUES):
    """Fit the logistic regression and choose its regularisation on validation data.

    Features are standardised to zero mean and unit variance (using the
    training set only), so one regularisation strength suits all of them.
    ``class_weight="balanced"`` up-weights the minority class (good trays) so
    the model cannot do well by calling every tray defective. ``C`` is the
    inverse regularisation strength: small values give a simpler model.
    The value with the highest validation ROC AUC is kept.

    Parameters
    ----------
    X_train, y_train : array_like
        Training features and tray labels (1 = defective).
    X_valid, y_valid : array_like
        Validation features and tray labels, used only to choose ``C``.
    C_values : sequence of float, optional
        Regularisation strengths to try.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The fitted model with the best validation AUC. Use
        ``model.predict_proba(X)[:, 1]`` for defect probabilities.
    search : pandas.DataFrame
        ``C`` and ``valid_auc`` for every value tried.
    """
    best_model, best_auc, rows = None, -np.inf, []
    for C in C_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
        )
        model.fit(X_train, y_train)
        auc = roc_auc(y_valid, model.predict_proba(X_valid)[:, 1])
        rows.append({"C": C, "valid_auc": auc})
        if auc > best_auc:
            best_model, best_auc = model, auc
    return best_model, pd.DataFrame(rows)