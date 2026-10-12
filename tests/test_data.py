"""Tests for packinspect.data."""

import pytest

from packinspect.data import count_images


def test_count_images_counts_only_images(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.PNG").touch()
    (tmp_path / "labels.txt").touch()
    assert count_images(tmp_path) == 2


def test_count_images_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_images(tmp_path / "does_not_exist")