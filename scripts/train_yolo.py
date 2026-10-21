"""Train a COCO-pretrained YOLO detector on the pork rasher dataset.

Run from the repository root::

    python scripts/train_yolo.py --epochs 1     # quick timing test
    python scripts/train_yolo.py                # full run

Training outputs (curves, plots, weights) are written to
``runs/<model>/train/``, which Git ignores. The per-epoch training
curves are copied to ``results/<model>/training_curves.csv``.
"""

import argparse
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from packinspect.data import write_data_yaml
from packinspect.detector import default_device


def main():
    """Parse settings, train the model and save the training curves."""
    parser = argparse.ArgumentParser(description="Train a YOLO defect detector.")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained weights to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="training image size; the 720x540 images are letterboxed to this")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15,
                        help="stop early if validation mAP has not improved for this many epochs")
    parser.add_argument("--device", default=default_device(), help="mps, cpu, or a CUDA GPU index")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path("runs").resolve() / Path(args.model).stem
    data_yaml = write_data_yaml("data")
    print(f"Training {args.model} on '{args.device}' for up to {args.epochs} epochs")

    start = time.perf_counter()
    YOLO(args.model).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, seed=args.seed, deterministic=True,
        project=str(run_dir), name="train", exist_ok=True,
    )
    print(f"Training took {(time.perf_counter() - start) / 60:.1f} min")

    results_dir = Path("results") / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "train" / "results.csv", results_dir / "training_curves.csv")
    print("Best weights:", run_dir / "train" / "weights" / "best.pt")


if __name__ == "__main__":
    main()