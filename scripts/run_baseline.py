"""Train and evaluate the classical baseline (colour and edge features + logistic regression).

Run from the repository root::

    python scripts/run_baseline.py
    python scripts/analyse_threshold.py --model baseline

Writes to ``results/baseline/``:

- ``predictions_valid.csv``, ``predictions_test.csv``: one row per tray, in the
  same format as the YOLO predictions. The ``max_conf`` column holds the
  baseline's defect probability, so ``analyse_threshold.py`` works unchanged
  and both methods are judged by exactly the same code.
- ``c_search.csv``: validation ROC AUC for each regularisation strength tried

and saves the fitted model to ``runs/baseline/model.joblib`` for timing.
"""

import time
from pathlib import Path

import joblib

from packinspect.baseline import fit_baseline, load_split
from packinspect.data import SPLITS, load_class_names

DATA_DIR = Path("data")


def main():
    """Extract features, fit the model and save predictions."""
    names = load_class_names(DATA_DIR / "data.yaml")
    data = {}
    for split in SPLITS:
        start = time.perf_counter()
        X, meta = load_split(DATA_DIR / split, names)
        data[split] = (X, meta)
        print(f"{split}: {len(meta)} images, {X.shape[1]} features "
              f"({time.perf_counter() - start:.1f} s)")

    (X_train, m_train), (X_valid, m_valid) = data["train"], data["valid"]
    model, search = fit_baseline(X_train, m_train["gt_defective"],
                                 X_valid, m_valid["gt_defective"])
    print("\nRegularisation search (chosen by validation ROC AUC):")
    print(search.round(3).to_string(index=False))

    out_dir = Path("results") / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(out_dir / "c_search.csv", index=False)
    for split in ("valid", "test"):
        X, meta = data[split]
        preds = meta.assign(max_conf=model.predict_proba(X)[:, 1])
        preds.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    model_dir = Path("runs") / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    print(f"\nPredictions saved to {out_dir}. Next: python scripts/analyse_threshold.py --model baseline")


if __name__ == "__main__":
    main()