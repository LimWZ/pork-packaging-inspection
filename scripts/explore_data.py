from pathlib import Path

import pandas as pd

from packinspect.data import SPLITS, find_split_leakage, load_class_names, summarise_split

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def main():
    class_names = load_class_names(DATA_DIR / "data.yaml")
    rows = []

    for split in SPLITS:
        s = summarise_split(DATA_DIR / split, class_names)
        print(f"\n{split}: {s['images']} images from {s['originals']} originals | "
              f"{s['no_defect_images']} with no defects | "
              f"{s['missing_labels']} missing label files")
        for name in class_names:
            print(f"  {name:16s} {s['boxes'][name]:5d} boxes in {s['images_with'][name]:5d} images")
            rows.append({"split": split, "class": name,
                         "boxes": s["boxes"][name], "images": s["images_with"][name]})
        rows.append({"split": split, "class": "(no defects)",
                     "boxes": 0, "images": s["no_defect_images"]})

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    leaks = find_split_leakage(DATA_DIR)
    print(f"\nFilenames appearing in more than one split (check by eye): {len(leaks)}")
    for stem, splits in list(leaks.items())[:10]:
        print(f"  {stem}: {', '.join(splits)}")


if __name__ == "__main__":
    main()