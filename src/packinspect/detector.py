from pathlib import Path

import pandas as pd
import torch

from packinspect.data import read_yolo_labels


def default_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detection_metrics(model, data_yaml, class_names, imgsz=640, device=None, out_dir=None):
    device = device or default_device()
    metric_rows, speed_rows = [], []
    for split in ("val", "test"):
        m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device,
                      project=str(out_dir) if out_dir else None,
                      name=f"eval_{split}", exist_ok=True)
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            metric_rows.append({"split": split, "class": class_names[int(c)],
                                "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
        metric_rows.append({"split": split, "class": "all", "precision": m.box.mp,
                            "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map})
        speed_rows.append({"split": split, "device": device, **m.speed})
    return pd.DataFrame(metric_rows), pd.DataFrame(speed_rows)


def export_predictions(model, split_dir, class_names, imgsz=640, device=None):
    device = device or default_device()
    split_dir = Path(split_dir)
    rows = []
    # conf=0.001 keeps almost every candidate box, so any threshold can be tested later.
    results = model.predict(source=str(split_dir / "images"), imgsz=imgsz, conf=0.001,
                            device=device, stream=True, verbose=False)
    for r in results:
        image = Path(r.path)
        boxes = read_yolo_labels(split_dir / "labels" / f"{image.stem}.txt", len(class_names))
        gt = sorted({b[0] for b in boxes})
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        row = {
            "image": image.name,
            "gt_defective": int(bool(gt)),
            "gt_classes": " ".join(class_names[c] for c in gt),
            "max_conf": float(confs.max()) if confs.size else 0.0,
        }
        for k, name in enumerate(class_names):
            mask = classes == k
            row[f"max_conf_{name}"] = float(confs[mask].max()) if mask.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)