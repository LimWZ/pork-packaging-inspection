import cv2
import numpy as np
import pytest

from packinspect.baseline import fit_baseline, image_features, load_split

N_FEATURES = 8 * 4 * 4 + 4 * 4


def test_image_features_shape_and_histogram_sums_to_one():
    image = np.random.default_rng(0).integers(0, 256, (540, 720, 3), dtype=np.uint8)
    f = image_features(image)
    assert f.shape == (N_FEATURES,)
    assert f[:128].sum() == pytest.approx(1.0)


def test_image_features_plain_image_has_no_edges_and_one_colour():
    image = np.full((540, 720, 3), (40, 80, 200), dtype=np.uint8)
    f = image_features(image)
    assert f[:128].max() == pytest.approx(1.0)       # every pixel in one colour bin
    assert np.allclose(f[128:], 0.0)                 # no edges anywhere


def test_image_features_finds_edge_in_the_right_cell():
    image = np.zeros((540, 720, 3), dtype=np.uint8)
    image[:, 90:] = 255     # vertical edge in the first column of cells (x < 180)
    edges = image_features(image)[128:].reshape(4, 4)
    assert (edges[:, 0] > 0).all()
    assert np.allclose(edges[:, 1:], 0.0)


@pytest.mark.parametrize("bad", [
    np.zeros((540, 720), dtype=np.uint8),           # greyscale
    np.zeros((540, 720, 3), dtype=np.float32),      # wrong dtype
    None,                                           # failed image read
])
def test_image_features_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        image_features(bad)


def test_load_split(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for name, labels in [("a", "1 0.5 0.5 0.2 0.2\n"), ("b", "")]:
        cv2.imwrite(str(tmp_path / "images" / f"{name}.jpg"),
                    np.full((54, 72, 3), 128, dtype=np.uint8))
        (tmp_path / "labels" / f"{name}.txt").write_text(labels)
    X, meta = load_split(tmp_path, ["loose-meat", "wrinkle"])
    assert X.shape == (2, N_FEATURES)
    assert meta["gt_defective"].tolist() == [1, 0]
    assert meta["gt_classes"].tolist() == ["wrinkle", ""]


def test_load_split_empty_folder_raises(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, ["loose-meat"])


def test_fit_baseline_learns_separable_data():
    rng = np.random.default_rng(0)
    def make(n):
        y = rng.integers(0, 2, n)
        X = rng.normal(size=(n, 5)) + 3 * y[:, None]   # classes 3 standard deviations apart
        return X, y
    X_train, y_train = make(200)
    X_valid, y_valid = make(100)
    model, search = fit_baseline(X_train, y_train, X_valid, y_valid, C_values=(0.1, 1.0))
    assert len(search) == 2
    assert search["valid_auc"].max() > 0.95
    assert model.predict_proba(X_valid).shape == (100, 2)