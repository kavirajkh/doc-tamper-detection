# Document Tamper Detection

Classifies an invoice image as `authentic`, `inpaint_and_rewrite`, `crop_and_replace` or `inconclusive`, and writes a heat map showing where it thinks the edit is. Everything (data, model, training) is built from scratch with open-source libraries. No pretrained weights, no external datasets.

## Layout

| Path | Purpose |
| --- | --- |
| `predict.py` | Required CLI: `--input` folder of images, `--output` folder, writes `<name>.json` and `<name>_heatmap.png` per image |
| `evaluate.py` | Scorer provided with the assignment (unchanged) |
| `tamperdoc/templates.py` | Renders synthetic invoices from 12 layout templates and records every field's bounding box |
| `tamperdoc/tamper.py` | The two forgeries (inpaint-and-rewrite, crop-and-replace) and capture/degradation simulation |
| `tamperdoc/generate.py` | Builds a labelled dataset: `images/`, `masks/`, `labels.csv` |
| `tamperdoc/model.py` | TamperNet: RGB + SRM-noise + 8x8 block-DCT branches, cross-gated fusion, 3-class segmentation head (0.7 M parameters) |
| `tamperdoc/data.py`, `tamperdoc/train.py` | Crop sampler (edits, hard negatives) and training on 192x192 crops aligned to the JPEG grid |
| `tamperdoc/infer.py` | Heat map, model and ELA signals, fused verdict with an `inconclusive` band |
| `tamperdoc/calibrate.py` | Fits the signal fusion and sets decision thresholds |
| `tamperdoc/figures.py` | Builds the report figures |
| `demo_overlay.py` | Optional demo helper: side-by-side panels (document, heat map, ground truth) for a screen share |
| `weights/` | Trained `tampernet.pt` and `calibration.json` |
| `fonts/` | Open-licence fonts used by the renderer (licences included) |
| `REPORT.md` | Approach, honest evaluation, failure cases, next steps |

## Setup

Tested on macOS 26 (Apple M1 Pro, 10 cores, 32 GB RAM), Python 3.12.4, CPU only. Python 3.10+ is required. No GPU needed.

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check the install (prints `0.702291 M params`):

```
python -m tamperdoc.model
```

## Run the trained model

```
python predict.py --input <folder of images> --output results
```

Each input `name.jpg` produces `results/name.json` (verdict, confidence, tamper_score, signals, reason) and `results/name_heatmap.png` (8-bit, same size as the input, 255 = most likely tampered). Inference takes about 0.5 s per page on the hardware above.

To try it on labelled data and score it, generate a small demo set first (see the next section), then:

```
python predict.py --input data/demo/images --output results_demo
python evaluate.py --pred results_demo --gt data/demo
```

(`evaluate.py` expects `labels.csv` and `masks/<id>.png` in the `--gt` folder, which the generator writes.)

## Generate data

```
python -m tamperdoc.generate --out data/demo --n 20 --templates 0-5 --seed 1
```

Output: `data/demo/images/*.jpg`, `data/demo/masks/*.png` (255 where the forger edited, all black for authentic pages), `data/demo/labels.csv`. Generation is deterministic for a given `--seed`.

## Rebuild everything from scratch

About 75 minutes on a 2-core CPU, nearly all of it training. CPU training is not bit-for-bit deterministic, so results land close to, not exactly on, the shipped weights. The `data/val` set (seed 2) regenerates byte-identically, but the shipped `calibration.json` was fit on a calibration set that `--seed 3` does not rebuild exactly, so a fresh `tamperdoc.calibrate` run gives somewhat different thresholds (see REPORT.md). Held-out numbers in the report use the shipped `weights/`.

```
python -m tamperdoc.generate --out data/train --n 1800 --templates 0-5 --seed 1 --save_fields   # training
python -m tamperdoc.generate --out data/calib --n 300  --templates 0-5 --seed 3                  # checkpoint pick + calibration
python -m tamperdoc.generate --out data/val   --n 300  --templates 6-7 --seed 2                  # held-out templates
python -m tamperdoc.train --train data/train --select data/calib --steps 2400 --eval_every 400
python -m tamperdoc.calibrate --data data/calib
python predict.py --input data/val/images --output results_val
python evaluate.py --pred results_val --gt data/val
```

Quick smoke test of the training loop (about 1-2 minutes on CPU; the result is not a usable model):

```
python -m tamperdoc.generate --out data/mini --n 60 --templates 0-5 --seed 1 --save_fields
python -m tamperdoc.train --train data/mini --select data/mini --steps 25 --eval_every 25 --out weights_mini.pt
```

## Demo overlay

Renders [document | heat map | ground truth] panels for any labelled folder (needs only numpy and Pillow):

```
python demo_overlay.py --images data/demo/images --pred results_demo --gt data/demo --out demo_panels
```

Output: `demo_panels/<id>_panel.png` per image and `demo_panels/contact_sheet.png`.

## Reproducing the reported numbers

The numbers in `REPORT.md` come from the shipped `weights/` and these two commands (the datasets regenerate byte-identically from their seeds):

```
python -m tamperdoc.generate --out data/stress --n 180 --templates 8-11 --seed 5 --profile private --prefix case
python predict.py --input data/stress/images --output results_stress && python evaluate.py --pred results_stress --gt data/stress
```

plus the `data/val` commands in the section above. If you retrain or recalibrate, the numbers will change; update `REPORT.md` and regenerate `figures/` with `python -m tamperdoc.figures --pred results_val --gt data/val --out figures`.

## Template split

| Templates | Used for |
| --- | --- |
| 0-5 | Training, checkpoint selection and calibration |
| 6-7 | Held-out evaluation (never used for any decision) |
| 8-11 | Unseen-font/layout stress test; `--profile private` adds phone-photo and print-and-scan capture |

## Third-party components

No pretrained models and no external datasets. Libraries: numpy (BSD-3), Pillow (HPND), opencv-python-headless (Apache-2.0), scikit-learn (BSD-3), PyTorch (BSD-3). Fonts: DejaVu (free Bitstream Vera-derived licence), Liberation (SIL OFL 1.1), GNU FreeFont (GPLv3 with font exception); licence texts are in `fonts/`.
