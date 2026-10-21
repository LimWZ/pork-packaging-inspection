"""Tests for packinspect.data."""

import pytest

from packinspect.data import (count_images, find_split_leakage, original_stem,
                              read_yolo_labels, summarise_split)


def test_count_images_counts_only_images(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.PNG").touch()
    (tmp_path / "labels.txt").touch()
    assert count_images(tmp_path) == 2


def test_count_images_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_images(tmp_path / "does_not_exist")


def test_read_yolo_labels_valid(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.1\n3 0.1 0.9 0.05 0.05\n")
    assert read_yolo_labels(p, num_classes=5) == [
        (0, 0.5, 0.5, 0.2, 0.1),
        (3, 0.1, 0.9, 0.05, 0.05),
    ]


def test_read_yolo_labels_empty_file_means_no_defects(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("")
    assert read_yolo_labels(p, num_classes=5) == []


@pytest.mark.parametrize("line", [
    "0 0.5 0.5 0.2",          # too few values
    "7 0.5 0.5 0.2 0.1",      # class id out of range
    "0 1.5 0.5 0.2 0.1",      # coordinate outside [0, 1]
])
def test_read_yolo_labels_rejects_bad_lines(tmp_path, line):
    p = tmp_path / "a.txt"
    p.write_text(line)
    with pytest.raises(ValueError):
        read_yolo_labels(p, num_classes=5)


def test_original_stem_strips_roboflow_suffix():
    assert original_stem("tray_012_jpg.rf.3f9a1c.jpg") == "tray_012_jpg"


def test_original_stem_without_suffix():
    assert original_stem("tray_012.jpg") == "tray_012"


def test_summarise_split_counts(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    examples = [
        ("a", "0 0.5 0.5 0.2 0.1\n0 0.2 0.2 0.1 0.1\n"),  # two loose-meat boxes
        ("b", ""),                                       # no defects
        ("c", None),                                     # label file missing
    ]
    for name, labels in examples:
        (tmp_path / "images" / f"{name}.jpg").touch()
        if labels is not None:
            (tmp_path / "labels" / f"{name}.txt").write_text(labels)

    s = summarise_split(tmp_path, ["loose-meat", "wrinkle"])
    assert s["images"] == 3
    assert s["no_defect_images"] == 1
    assert s["missing_labels"] == 1
    assert s["boxes"]["loose-meat"] == 2
    assert s["images_with"]["loose-meat"] == 1


def test_find_split_leakage_detects_shared_original(tmp_path):
    files = [
        ("train", "a_jpg.rf.111.jpg"),
        ("train", "a_jpg.rf.222.jpg"),   # augmented copy, same split: fine
        ("valid", "a_jpg.rf.333.jpg"),   # same original in valid: leak
        ("test", "b_jpg.rf.444.jpg"),
    ]
    for split, name in files:
        d = tmp_path / split / "images"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).touch()
    assert find_split_leakage(tmp_path) == {"a_jpg": ["train", "valid"]}