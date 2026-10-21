"""Loading and checking the pork rasher packaging dataset.

The dataset is a Roboflow export in YOLO format::

    data/
        data.yaml
        train/images/*.jpg    train/labels/*.txt
        valid/images/*.jpg    valid/labels/*.txt
        test/images/*.jpg     test/labels/*.txt

Each label file has one line per bounding box::

    class_id x_centre y_centre width height

with the four coordinates normalised to [0, 1] by the image size.
An empty label file means the image has no labelled defects.
"""

import re
from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")

# Roboflow appends ".rf.<hash>" to every exported filename.
_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9A-Za-z]+$")

# Allow for float rounding when checking normalised coordinates.
_COORD_TOL = 1e-6


def count_images(folder):
    """Count the image files in a folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing image files.

    Returns
    -------
    int
        Number of files with a .jpg, .jpeg or .png extension
        (case-insensitive).

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist or is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_class_names(yaml_path):
    """Read the class names from a YOLO ``data.yaml`` file.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to the dataset's ``data.yaml``.

    Returns
    -------
    list of str
        Class names, indexed by class id.

    Raises
    ------
    FileNotFoundError
        If ``yaml_path`` does not exist.
    KeyError
        If the file has no ``names`` entry.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No such file: {yaml_path}")
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    if "names" not in config:
        raise KeyError(f"{yaml_path} has no 'names' entry")
    names = config["names"]
    # Some exporters write names as {id: name} instead of a list.
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return list(names)


def read_yolo_labels(label_path, num_classes):
    """Read and validate a YOLO-format label file.

    Parameters
    ----------
    label_path : str or pathlib.Path
        Path to a ``.txt`` label file.
    num_classes : int
        Number of classes; valid class ids are ``0`` to ``num_classes - 1``.

    Returns
    -------
    list of tuple
        One ``(class_id, x_centre, y_centre, width, height)`` tuple per box,
        with ``class_id`` an int and the coordinates floats in [0, 1].
        An empty list means the image has no labelled defects.

    Raises
    ------
    ValueError
        If a line does not have exactly five values, the class id is out
        of range, or a coordinate is outside [0, 1].
    """
    boxes = []
    lines = Path(label_path).read_text().splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_no}: class id {class_id} out of range")
        if not all(-_COORD_TOL <= v <= 1 + _COORD_TOL for v in coords):
            raise ValueError(f"{label_path}:{line_no}: coordinate outside [0, 1]")
        boxes.append((class_id, *coords))
    return boxes


def original_stem(image_name):
    """Recover the original image name from a Roboflow export filename.

    Roboflow renames each exported image as ``<original>_<ext>.rf.<hash>.jpg``.
    Augmented copies of one original share the ``<original>_<ext>`` part but
    have different hashes.

    Parameters
    ----------
    image_name : str
        Filename (not a full path) of an exported image or label file.

    Returns
    -------
    str
        The name before ``.rf.<hash>``, or the name without its extension
        if there is no Roboflow suffix.

    Examples
    --------
    >>> original_stem("tray_012_jpg.rf.3f9a1c.jpg")
    'tray_012_jpg'
    """
    return _ROBOFLOW_SUFFIX.sub("", Path(image_name).stem)


def summarise_split(split_dir, class_names):
    """Count images, labels and boxes in one dataset split.

    Parameters
    ----------
    split_dir : str or pathlib.Path
        Split folder containing ``images/`` and ``labels/`` subfolders.
    class_names : list of str
        Class names, indexed by class id.

    Returns
    -------
    dict
        ``images``: number of image files;
        ``originals``: number of distinct original images (before augmentation);
        ``missing_labels``: images with no matching label file;
        ``no_defect_images``: images whose label file is empty;
        ``boxes``: Counter of boxes per class name;
        ``images_with``: Counter of images containing each class name.

    Raises
    ------
    FileNotFoundError
        If ``images/`` or ``labels/`` is missing.
    """
    split_dir = Path(split_dir)
    image_dir, label_dir = split_dir / "images", split_dir / "labels"
    for d in (image_dir, label_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"No such folder: {d}")

    summary = {"images": 0, "missing_labels": 0, "no_defect_images": 0,
               "boxes": Counter(), "images_with": Counter()}
    originals = set()

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        summary["images"] += 1
        originals.add(original_stem(image_path.name))

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            summary["missing_labels"] += 1
            continue

        boxes = read_yolo_labels(label_path, len(class_names))
        if not boxes:
            summary["no_defect_images"] += 1
        classes = [class_names[b[0]] for b in boxes]
        summary["boxes"].update(classes)
        summary["images_with"].update(set(classes))

    summary["originals"] = len(originals)
    return summary


def find_split_leakage(data_dir, splits=SPLITS):
    """Find original images whose copies appear in more than one split.

    Different photos can share a filename in this dataset, so results
    are candidates for manual inspection, not confirmed leaks.

    Only catches copies that share a Roboflow original name; near-duplicate
    photos saved under different names are not detected.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Dataset root containing one folder per split.
    splits : sequence of str, optional
        Split folder names to compare.

    Returns
    -------
    dict
        Maps each leaked original name to the sorted list of splits it
        appears in. An empty dict means no leakage was found.
    """
    found = {}
    for split in splits:
        for image_path in (Path(data_dir) / split / "images").iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                found.setdefault(original_stem(image_path.name), set()).add(split)
    return {stem: sorted(s) for stem, s in found.items() if len(s) > 1}