"""Choose the pass/fail threshold on the validation set and report it on the test set.

Run from the repository root after ``evaluate_yolo.py``::

    python scripts/analyse_threshold.py
    python scripts/analyse_threshold.py --min-catch-rate 0.9   # sensitivity check

Reads ``results/<model>/predictions_{valid,test}.csv`` and writes to
``results/<model>/``:

- ``tray_level_results.csv``: catch rate, pass rate and accuracy with 95%
  Wilson intervals, for the chosen threshold, YOLO's default threshold of 0.25,
  and the trivial "fail every tray" and "pass every tray" rules
- ``threshold_sweep.csv``: catch and pass rate at thresholds 0.00 to 1.00
- ``test_errors.csv``: test trays the chosen threshold gets wrong

and the figure ``reports/figures/<model>_threshold.png``.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from packinspect.evaluate import choose_threshold, roc_auc, summarise, threshold_sweep

CATCH_COLOUR = "#2a78d6"   # blue
PASS_COLOUR = "#eb6834"    # orange
INK = "#52514e"            # axis and label text


def plot_threshold(sweeps, chosen, chosen_points, out_path):
    """Plot rates against threshold (left) and the ROC curve (right).

    ``chosen_points`` maps each split to its ``(false reject rate, catch rate)``
    at the chosen threshold, which is marked on the ROC curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    styles = {"valid": "--", "test": "-"}

    for split, sweep in sweeps.items():
        ax1.plot(sweep["threshold"], sweep["catch_rate"], styles[split],
                 color=CATCH_COLOUR, lw=2, label=f"Catch rate ({split})")
        ax1.plot(sweep["threshold"], sweep["pass_rate"], styles[split],
                 color=PASS_COLOUR, lw=2, label=f"Pass rate ({split})")
        ax2.plot(1 - sweep["pass_rate"], sweep["catch_rate"], styles[split],
                 color=CATCH_COLOUR, lw=2, label=split)
        ax2.plot(*chosen_points[split], "o", ms=8, color=CATCH_COLOUR,
                 markeredgecolor="white", markeredgewidth=2)

    ax1.axvline(chosen, color=INK, lw=1, ls=":")
    ax1.annotate(f"chosen {chosen:.2f}", (chosen, 0.04), xytext=(5, 0),
                 textcoords="offset points", color=INK, fontsize=9)
    ax1.set(xlabel="Decision threshold (highest box confidence)",
            ylabel="Share of trays", xlim=(0, 1), ylim=(-0.02, 1.02),
            title="Rates against decision threshold")
    ax1.legend(fontsize=8, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.18))

    ax2.plot([0, 1], [0, 1], color="#b0afaa", lw=1, ls=":", label="chance")
    ax2.set(xlabel="False reject rate (good trays failed)",
            ylabel="Catch rate (defective trays failed)",
            xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="ROC curve (dots: chosen threshold)")
    ax2.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.18))

    for ax in (ax1, ax2):
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.tick_params(colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    """Choose the threshold, evaluate it and save tables and the figure."""
    parser = argparse.ArgumentParser(description="Tray-level pass/fail analysis.")
    parser.add_argument("--model", default="yolo26n", help="results folder name")
    parser.add_argument("--min-catch-rate", type=float, default=0.95,
                        help="share of defective validation trays the threshold must catch")
    args = parser.parse_args()

    results_dir = Path("results") / args.model
    preds = {split: pd.read_csv(results_dir / f"predictions_{split}.csv").fillna({"gt_classes": ""})
             for split in ("valid", "test")}

    chosen = choose_threshold(preds["valid"]["gt_defective"], preds["valid"]["max_conf"],
                              min_catch_rate=args.min_catch_rate)
    print(f"Threshold chosen on validation (catch rate >= {args.min_catch_rate:.0%}): {chosen:.3f}\n")

    rules = [("chosen threshold", chosen), ("YOLO default 0.25", 0.25),
             ("fail every tray", 0.0), ("pass every tray", np.inf)]
    rows, chosen_points = [], {}
    for split, df in preds.items():
        auc = roc_auc(df["gt_defective"], df["max_conf"])
        print(f"{split}: ROC AUC = {auc:.3f}")
        for rule, t in rules:
            s = summarise(df["gt_defective"], df["max_conf"], t)
            rows.append({"split": split, "rule": rule, **s})
            if rule == "chosen threshold":
                chosen_points[split] = (1 - s["pass_rate"], s["catch_rate"])
            print(f"  {rule:18s} caught {s['caught']:3d}/{s['caught'] + s['missed']:<3d} "
                  f"{s['catch_rate']:6.1%} [{s['catch_rate_low']:.0%}-{s['catch_rate_high']:.0%}]   "
                  f"passed {s['passed']:3d}/{s['passed'] + s['false_rejects']:<3d} "
                  f"{s['pass_rate']:6.1%} [{s['pass_rate_low']:.0%}-{s['pass_rate_high']:.0%}]   "
                  f"accuracy {s['accuracy']:.1%}")
        print()
    pd.DataFrame(rows).to_csv(results_dir / "tray_level_results.csv", index=False)

    grid = np.union1d(np.round(np.linspace(0, 1, 101), 2), chosen)
    sweeps = {split: threshold_sweep(df["gt_defective"], df["max_conf"], grid)
              for split, df in preds.items()}
    pd.concat([s.assign(split=k) for k, s in sweeps.items()]).to_csv(
        results_dir / "threshold_sweep.csv", index=False)

    test = preds["test"]
    fail = test["max_conf"] >= chosen
    errors = test[fail != (test["gt_defective"] == 1)].copy()
    errors["error"] = np.where(errors["gt_defective"] == 1, "missed defect", "false reject")
    errors = errors[["image", "error", "gt_classes", "max_conf"]].sort_values("error")
    errors.to_csv(results_dir / "test_errors.csv", index=False)
    print(f"Test trays wrong at the chosen threshold: {len(errors)}")
    print(errors.to_string(index=False))

    figures = Path("reports") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_threshold(sweeps, chosen, chosen_points, figures / f"{args.model}_threshold.png")
    print(f"\nFigure saved to {figures / f'{args.model}_threshold.png'}")


if __name__ == "__main__":
    main()