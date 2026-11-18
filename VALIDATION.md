# Validation guide

Step-by-step checks for every claim made about this project, with the commands to run and the results I recorded. Anyone can repeat them on a clean machine.

**Recorded run:** 2026-11-18, fresh `git clone`, macOS (Apple M1 Pro), Python 3.12.4, CPU only. Installed: numpy 2.5.5, Pillow 13.0.0, opencv-python-headless 5.1.0.95, scikit-learn 1.9.1, torch 2.15.0.

Notes for running the commands:

- Run them one at a time, from the repository root with the virtual environment active.
- Do not pipe `predict.py` into `head`; it stops early with `BrokenPipeError` and leaves a partial results folder. `| tail -1` is fine.
- `evaluate.py` scores any missing prediction as "authentic". If it prints `WARNING: ... predictions missing`, the predictions folder is incomplete: delete it and run `predict.py` again.

| # | Claim | Result |
| --- | --- | --- |
| 1 | Clean clone installs and runs | Pass |
| 2 | 12 layouts | Pass |
| 3 | Masks match the images and are empty for authentic pages | Pass |
| 4 | Degradation is applied to every class | Pass |
| 5 | A fixed seed rebuilds the same data | Pass |
| 6 | Held-out templates are disjoint from training templates | Pass |
| 7 | No pretrained models or downloads | Pass |
| 8 | `predict.py` meets the output contract | Pass |
| 9 | About 0.5 s per page on CPU | Pass |
| 10 | Held-out results match the report | Pass |
| 11 | Unseen-template results match the report | Pass |
| 12 | Calibration thresholds | See caveat in section 12 |

---

## 1. Clean clone, install, model check

```
git clone https://github.com/kavirajkh/doc-tamper-detection
cd doc-tamper-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tamperdoc.model
```

Expected: `0.702291 M params` and `torch.Size([2, 3, 128, 128])`.

## 2. Twelve templates

```
python -c "from tamperdoc.templates import TEMPLATES; print(len(TEMPLATES))"
```

Expected: `12`.

## 3. Generate and score a small demo set

```
python -m tamperdoc.generate --out data/demo --n 20 --templates 0-5 --seed 1
python predict.py --input data/demo/images --output results_demo
python evaluate.py --pred results_demo --gt data/demo
```

Recorded: 20 images, mean 0.52 s, max 0.78 s; AUC 0.8681, false-positive rate 0.0, detection recall 0.1538, inconclusive rate 0.65, pixel-F1 0.35. Twenty images is too few for real metrics; this is a plumbing check.

## 4. Masks are exact

```
python -m tamperdoc.generate --out data/chk --n 60 --templates 0-5 --seed 1
python - <<'EOF'
import csv, numpy as np
from PIL import Image
for r in csv.DictReader(open('data/chk/labels.csv')):
    m = np.array(Image.open(f"data/chk/masks/{r['id']}.png"))
    im = Image.open(f"data/chk/images/{r['id']}.jpg")
    assert m.shape == im.size[::-1], r['id']
    assert (m.max() > 0) == (r['label'] != 'authentic'), r['id']
print("mask checks ok")
EOF
```

Expected: `mask checks ok`. Then look at a few images next to their masks (`open data/chk/images` on macOS). The white box should sit on the edited field.

## 5. Degradation on every class

```
python - <<'EOF'
import csv, collections
c = collections.defaultdict(lambda: [0, 0])
for r in csv.DictReader(open('data/chk/labels.csv')):
    c[r['label']][0] += 1
    c[r['label']][1] += 'jpeg' in r['degradations']
print(dict(c))
EOF
```

Expected: every label has equal counts in both positions, e.g. `{'authentic': [20, 20], 'inpaint_and_rewrite': [20, 20], 'crop_and_replace': [20, 20]}`.

## 6. Fixed seed gives identical data

```
python -m tamperdoc.generate --out data/chk2 --n 60 --templates 0-5 --seed 1
diff -rq -x .DS_Store data/chk data/chk2 && echo IDENTICAL
```

Expected: `IDENTICAL`.

## 7. Held-out templates are really held out

```
python -m tamperdoc.generate --out data/val --n 300 --templates 6-7 --seed 2
python - <<'EOF'
import csv
t = lambda p: {r['template'] for r in csv.DictReader(open(p))}
print("train", sorted(t('data/chk/labels.csv')), "val", sorted(t('data/val/labels.csv')))
EOF
```

Expected: `train ['0', '1', '2', '3', '4', '5'] val ['6', '7']`.

## 8. No pretrained models or downloads

```
grep -rniE "pretrained|torchvision|hub\.load|from_pretrained|urlopen|requests\.|http" tamperdoc predict.py
head -2 weights/train_log.txt
```

Expected: the grep prints nothing; the log starts at a high loss (`step 25 loss 1.4869`, `step 50 loss 1.2539`), consistent with training from scratch.

## 9. Output contract

```
python - <<'EOF'
import json, glob, os
from PIL import Image
for j in glob.glob('results_demo/*.json'):
    d = json.load(open(j)); n = os.path.basename(j)[:-5]
    assert {'image','verdict','confidence','tamper_score'} <= d.keys()
    assert d['verdict'] in ('authentic','inpaint_and_rewrite','crop_and_replace','inconclusive')
    h = Image.open(f'results_demo/{n}_heatmap.png'); i = Image.open(f'data/demo/images/{n}.jpg')
    assert h.mode == 'L' and h.size == i.size
    assert 0 <= d['confidence'] <= 1 and 0 <= d['tamper_score'] <= 1 and d.get('reason')
print("contract ok")
EOF
```

Expected: `contract ok`.

## 10. Held-out results and speed

```
python predict.py --input data/val/images --output results_val | tail -1
ls results_val | wc -l
python evaluate.py --pred results_val --gt data/val | sed -n 1,10p
```

Expected: `300 images, mean ~0.5s, max < 1s`, then `600`, then:

| Metric | Value |
| --- | --- |
| AUC | 0.6146 |
| Macro-F1 | 0.1545 |
| Detection recall | 0.06 |
| False-positive rate | 0.04 |
| Inconclusive rate | 0.8133 |
| Pixel-F1 / IoU | 0.2784 / 0.2013 |

Recorded timing: mean 0.49 s, max 0.61 s per image, CPU only (`torch.cuda.is_available()` is `False`).

## 11. Unseen templates (stress set with phone-photo and print-and-scan capture)

```
python -m tamperdoc.generate --out data/stress --n 180 --templates 8-11 --seed 5 --profile private --prefix case
python predict.py --input data/stress/images --output results_stress | tail -1
python evaluate.py --pred results_stress --gt data/stress
```

Recorded: AUC 0.459, false-positive rate 0.0, detection recall 0.0167, inconclusive rate 0.70, pixel-F1 0.0904. Recorded timing in the clean-clone run: mean 0.55 s, max 0.80 s per image. By capture condition, AUC is 0.339 for phone photos and 0.419 for print-and-scan, both below chance.

## 12. Calibration (read the caveat)

```
python -m tamperdoc.generate --out data/calib --n 300 --templates 0-5 --seed 3
python -m tamperdoc.calibrate --data data/calib --out /tmp/calib_check.json
cat /tmp/calib_check.json
cat weights/calibration.json
```

Recorded on the regenerated calibration set: model coefficient 1.9614, ELA 0.5626, bias -0.953, band [0.5452, 0.7721]; on the calibration set, 2.0% of authentic pages are called tampered, 15.0% of tampered pages are called tampered and 77% are inconclusive; single-signal AUC is 0.651 for the model, 0.54 for ELA and 0.651 fused.

The shipped `weights/calibration.json` has model 1.8774, ELA 0.6985, bias -1.0867, band [0.5706, 0.786]. They differ because the shipped file was fit on a calibration set that the `--seed 3` command does not rebuild exactly (different page-to-template assignment). This is documented in `REPORT.md`. The held-out and unseen-template results above all use the shipped file. For comparison, scoring `data/val` with the regenerated calibration gives AUC 0.617, false-positive rate 0.11 and recall 0.21, which shows how sensitive the 2% target is to the calibration data:

```
python predict.py --input data/val/images --output results_val_regen --calibration /tmp/calib_check.json | tail -1
python evaluate.py --pred results_val_regen --gt data/val | sed -n 1,10p
```

## 13. Heat maps on the edits

```
python demo_overlay.py --images data/val/images --pred results_val --gt data/val --out demo_panels
open demo_panels/contact_sheet.png
```

By default only the first 12 images are processed (`--limit 0` does all, but the resulting contact sheet is far too tall for an image viewer to open; individual panels are `demo_panels/<id>_panel.png`). Each panel shows document, heat map overlay and ground truth. Compare with `figures/correct.png` and `figures/failures.png`. The report's failures are `doc_00013`, `doc_00017` and `doc_00126`.

## 14. Repository hygiene

```
git log --format='%ae' | sort -u
git ls-files | grep -iE "venv|grader|ASSIGNMENT|examples/"
git grep -niE "apple\.com"
```

Expected: one no-reply author address, and no output from the other two commands.

---

## Known limits

- The detector is a weak baseline: AUC 0.61 on held-out templates and chance level on unseen templates with phone and print-and-scan capture. It answers `inconclusive` for about 80% of pages by design.
- A full retrain is not bit-for-bit deterministic on CPU.
- Results depend on the calibration set (section 12).
