# Pork Packaging Inspection

Automated pass/fail inspection of pork rasher packaging from camera images. A YOLO26n detector finds packaging defects in each tray image, and a threshold chosen on validation data turns them into a pass/fail decision per tray. A classical baseline (colour and edge features + logistic regression) is evaluated with the same code for comparison.

**Input:** 720 x 540 RGB tray image. 
**Output:** defect boxes and a pass/fail decision per tray.

MMA3001 Pork Packaging Inspection, June Lim, 2026

## Setup
Tested with Python 3.14 on macOS, Apple [chip].

```bash
git clone https://github.com/LimWZ/pork-packaging-inspection.git
cd pork-packaging-inspection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The dataset is not stored in this repository Download it as described in [data/README.md](data/README.md).

## Usage

Run from the repository root, in this order:

```bash
python scripts/explore_data.py                    
python scripts/train_yolo.py                     
python scripts/evaluate_yolo.py 
python scripts/analyse_threshold.py 
python scripts/run_baseline.py
python scripts/analyse_threshold.py --model baseline
python scripts/time_inference.py
```


Results are written to `results/` and figures to `reports/figures/`.

## Tests and documentation

```bash
python -m pytest --html=reports/test_report.html --self-contained-html
pdoc src/packinspect -o docs --docformat numpy
```

Test report: `reports/test_report.html`. Documentation: `docs/index.html`.

## AI use

Code was partially drafted, modified, verified and debugged with AI assistance (Claude). Full disclosure is in the project report.

## Licence and attribution

Code: MIT licence (see `LICENSE`).
Dataset: *Pork Rasher Error Packaging*, version 4, Roboflow Universe, CC BY 4.0,
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4