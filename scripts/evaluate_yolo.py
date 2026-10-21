"""Evaluate trained YOLO weights on the validation and test sets.

Run from the repository root after training::

    python scripts/evaluate_yolo.py                  # evaluates runs/yolo26n
    python scripts/evaluate_yolo.py --device cpu     # CPU timing

Writes to ``results/<model>/``:

- ``detection_metrics.csv``: precision, recall, mAP50 and mAP50-95 per class
- ``speed.csv``: milliseconds per image for each processing stage
- ``predictions_valid.csv`` and ``predictions_test.csv``: ground truth and
  highest predicted confidence per image, for the pass/fail analysis
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from packinspect.data import load_class_names, write_data_yaml
from packinspect.detector import default_device, detection_metrics, export_predictions

DATA_DIR = Path("data")


def main():
    """Run the evaluation and save the result tables."""
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO detector.")
    parser.add_argument("--model", default="yolo26n", help="run name under runs/")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    run_dir = Path("runs").resolve() / args.model
    weights = run_dir / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"No trained weights at {weights}. Run train_yolo.py first.")

    model = YOLO(str(weights))
    names = load_class_names(DATA_DIR / "data.yaml")
    out_dir = Path("results") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, speed = detection_metrics(model, write_data_yaml(DATA_DIR), names,
                                       imgsz=args.imgsz, device=args.device, out_dir=run_dir)
    metrics.to_csv(out_dir / "detection_metrics.csv", index=False)
    speed.to_csv(out_dir / "speed.csv", index=False)
    print(metrics.round(3).to_string(index=False))

    for split in ("valid", "test"):
        preds = export_predictions(model, DATA_DIR / split, names,
                                   imgsz=args.imgsz, device=args.device)
        preds.to_csv(out_dir / f"predictions_{split}.csv", index=False)
        fail = preds["max_conf"] >= 0.25
        defective = preds["gt_defective"] == 1
        print(f"{split} at threshold 0.25: caught {(fail & defective).sum()}/{defective.sum()} "
              f"defective trays, passed {(~fail & ~defective).sum()}/{(~defective).sum()} good trays")


if __name__ == "__main__":
    main()