import argparse
from pathlib import Path

import cv2
import joblib
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from packinspect.baseline import image_features
from packinspect.data import IMAGE_EXTENSIONS
from packinspect.profiling import summarise_times, time_per_item


def main():
    parser = argparse.ArgumentParser(description="Time inference for YOLO and the baseline.")
    parser.add_argument("--model", default="yolo26n", help="YOLO run name under runs/")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 480, 640],
                        help="YOLO input sizes to time")
    args = parser.parse_args()

    image_dir = Path("data") / "test" / "images"
    images = [cv2.imread(str(p)) for p in sorted(image_dir.iterdir())
              if p.suffix.lower() in IMAGE_EXTENSIONS]
    h, w = images[0].shape[:2]
    print(f"Timing on {len(images)} test images ({w}x{h})\n")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
                      + (["0"] if torch.cuda.is_available() else [])
    rows = []

    weights = Path("runs") / args.model / "train" / "weights" / "best.pt"
    reference = YOLO(str(weights))
    params = get_num_params(reference.model)
    inputs = {}
    for size in args.sizes:
        # YOLO scales the long side to `size` and pads the short side up to a multiple of 32
        scale = size / max(h, w)
        net_h = -(-round(h * scale) // 32) * 32
        net_w = -(-round(w * scale) // 32) * 32
        inputs[size] = (f"{net_w}x{net_h}", get_flops(reference.model, imgsz=[net_h, net_w]))

    for device in devices:
        # A fresh model per device: Ultralytics keeps using the device of its first prediction
        yolo = YOLO(str(weights))
        for size in args.sizes:
            def run(image, size=size, device=device):
                # .cpu() waits for the GPU to finish, so the timing covers all the work
                return yolo.predict(image, imgsz=size, device=device, verbose=False)[0].boxes.conf.cpu()
            times = time_per_item(run, images)
            label, gflops = inputs[size]
            rows.append({"method": args.model, "input": label, "device": device,
                         "params": params, "GFLOPs": gflops, **summarise_times(times)})
            print(f"{args.model} {label} on {device}: {rows[-1]['median_ms']:.1f} ms/image")

    baseline = joblib.load(Path("runs") / "baseline" / "model.joblib")
    n_coef = baseline[-1].coef_.size + baseline[-1].intercept_.size
    times = time_per_item(lambda image: baseline.predict_proba(image_features(image)[None, :]), images)
    rows.append({"method": "baseline", "input": "320x240", "device": "cpu",
                 "params": n_coef, "GFLOPs": float("nan"), **summarise_times(times)})
    print(f"baseline 320x240 on cpu: {rows[-1]['median_ms']:.1f} ms/image")

    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv(Path("results") / "timing.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()