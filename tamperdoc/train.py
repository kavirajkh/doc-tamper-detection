"""Train TamperNet on random crops.

PRIMARY GOAL
    Optimise the network to segment tampered pixels, and keep the best checkpoint chosen on a set that is
    NOT the held-out evaluation set (so reported numbers stay honest).

MAJOR FUNCTIONS
    loss_fn()       cross-entropy (class-weighted) + 0.3 x Dice on P(tampered).
    select_score()  full-page validation used for checkpoint selection: pixel-F1 minus false-alarm area.
    main()          builds model/optimiser/loader, runs the step loop, saves the best checkpoint.

    python -m tamperdoc.train --train data/train --select data/calib --steps 1200

--select is used only to pick the best checkpoint. It comes from the training
templates (a different seed), so the held-out templates stay untouched until
the final evaluation.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import CropSampler, load, read_labels
from .infer import predict_maps
from .metrics import pixel_f1
from .model import TamperNet


def loss_fn(logits, target, class_w):
    """Segmentation loss = class-weighted cross-entropy + 0.3 x Dice.

    logits: B x 3 x H/2 x W/2 (half resolution). target: B x H x W class ids 0/1/2.
    """
    # Logits are at 1/2 resolution: subsample targets by striding (nearest neighbour keeps class ids intact).
    tgt = target[:, ::2, ::2]  # logits are at 1/2 resolution; nearest-neighbour subsample keeps labels crisp
    # Class-weighted cross-entropy: tampered classes weigh 2x because they are a tiny share of pixels.
    ce = F.cross_entropy(logits, tgt, weight=class_w)
    # Dice term on P(tampered) = 1 - P(background): directly optimises overlap and is insensitive to the
    # background/edit imbalance that lets CE settle for 'predict background everywhere'.
    p_t = 1 - torch.softmax(logits, 1)[:, 0]
    g = (tgt > 0).float()
    inter = (p_t * g).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p_t.sum((1, 2)) + g.sum((1, 2)) + 1)
    # Total loss = CE + 0.3 x (1 - Dice).
    return ce + 0.3 * dice.mean()


@torch.no_grad()
def select_score(model, root, rows):
    """Checkpoint-selection score on whole pages: mean pixel-F1 over tampered pages and mean fraction of authentic
    pixels falsely lit. Called with the CALIBRATION set, never the held-out templates.
    """
    model.eval()
    f1s, fp = [], []
    for r in rows:
        img, mask = load(root, r)
        heat, _ = predict_maps(model, img)
        if r["label"] == "authentic":
            fp.append((heat > 0.5).mean())
        else:
            f1s.append(pixel_f1(heat, mask))
    model.train()
    return float(np.mean(f1s)), float(np.mean(fp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--select", default="data/calib")
    ap.add_argument("--out", default="weights/tampernet.pt")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from these weights (fine-tuning)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    model = TamperNet()
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    # AdamW + OneCycle: ramps the learning rate up for 10% of steps, then anneals; works well for small nets with a high peak LR.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    # Infinite crop stream; training is step-based, not epoch-based.
    loader = torch.utils.data.DataLoader(CropSampler(a.train, a.crop, seed=a.seed), batch_size=a.batch,
                                         num_workers=1)
    # Class weights: background 1, inpaint 2, crop 2.
    class_w = torch.tensor([1.0, 2.0, 2.0])
    sel_rows = read_labels(a.select)[:45]
    best, t0, run = -1.0, time.time(), []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # TRAINING LOOP: forward -> loss -> backward -> AdamW step -> LR schedule step.
    for step, (x, y) in enumerate(loader, 1):
        loss = loss_fn(model(x), y, class_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 25 == 0:
            print(f"step {step} loss {np.mean(run[-25:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        # Periodically score on the selection set and keep ONLY the best checkpoint (score = F1 - false-alarm area),
        # so a worse later step cannot overwrite a better one (the shipped weights are step 2000, not 2400).
        if step % a.eval_every == 0 or step == a.steps:
            f1, fp = select_score(model, a.select, sel_rows)
            score = f1 - fp
            print(f"  select: pixel-F1 {f1:.3f}  authentic false-alarm pixels {fp:.4f}", flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), a.out)
                print(f"  saved {a.out}", flush=True)
        if step >= a.steps:
            break
    print(f"done in {time.time() - t0:.0f}s, best select score {best:.3f}")


if __name__ == "__main__":
    main()
