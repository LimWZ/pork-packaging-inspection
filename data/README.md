# Data

The dataset is not stored in this repository. Download it from Roboflow
before running the scripts.


## Download

1. Open https://universe.roboflow.com/hello-2aqe0/\pork-rasher-error-packaging/dataset/4
2. Click **Download Dataset**, choose **YOLOv8** format, then **Download as a ZIP**.
3. From the repository root, unzip it into this folder:

   ```bash
   unzip ~/Downloads/<downloaded-file>.zip -d data
   ```

## Expected structure

```
data/
├── data.yaml        class names and split paths
├── train/
│   ├── images/      3,349 images (augmented)
│   └── labels/
├── valid/
│   ├── images/      120 images
│   └── labels/
└── test/
    ├── images/      80 images
    └── labels/
```

Run `python scripts/explore_data.py` to check the counts match.

## Format

- **Images:** JPEG, 720 x 540 pixels, RGB.
- **Labels:** one `.txt` file per image, one line per defect:
  `class x_centre y_centre width height`, with coordinates normalised to [0, 1].
  An empty file means the tray has no labelled defects.
- **Classes:** `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

## Licence

*Pork Rasher Error Packaging*, version 4, Roboflow Universe, licensed
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The dataset is not covered by this repository's MIT licence.