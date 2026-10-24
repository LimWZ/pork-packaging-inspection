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
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: {folder}")
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_class_names(yaml_path):
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
    return _ROBOFLOW_SUFFIX.sub("", Path(image_name).stem)


def summarise_split(split_dir, class_names):
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
    found = {}
    for split in splits:
        for image_path in (Path(data_dir) / split / "images").iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                found.setdefault(original_stem(image_path.name), set()).add(split)
    return {stem: sorted(s) for stem, s in found.items() if len(s) > 1}

def write_data_yaml(data_dir, out_name="data_local.yaml"):
    data_dir = Path(data_dir).resolve()
    source = data_dir / "data.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"No such file: {source}")
    with open(source) as f:
        config = yaml.safe_load(f)
    config.update(path=str(data_dir), train="train/images",
                  val="valid/images", test="test/images")
    out_path = data_dir / out_name
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f)
    return out_path