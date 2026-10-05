"""Link the validation slice of each training set (every 20th image, the
same slice train.py holds out) as eval/data/<name>-VAL, so the JS pipeline
can score and tune on it without touching the evaluation sets.

Usage: python split.py
"""

import numpy as np

from train import ROOT, SOURCES

DATA = ROOT.parent / "eval" / "data"

for name in SOURCES:
    names = np.load(ROOT / "cache" / f"{name}.npz")["names"][::20]
    src = DATA / name
    dst = DATA / f"{name}-VAL"
    truth = "fixations" if (src / "fixations").exists() else "masks"
    for sub, ext in (("images", "jpg"), (truth, "png")):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            link = dst / sub / f"{n}.{ext}"
            if not link.exists():
                link.symlink_to(src / sub / f"{n}.{ext}")
    print(f"{dst.name}: {len(names)} images")
