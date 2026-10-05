"""Raise every detected head to full importance in a cache's targets.

Cutting a head is the worst crop there is, so heads found by faces.py become
the strongest part of each target map, and the --head loss then trains crops
to keep them whole. Writes cache/<name>-F.npz next to cache/<name>.npz.

Usage: python boost.py <cache-name>:<dataset-dir>...
  e.g. python boost.py SALICON-TR-T:../eval/data/SALICON-TR
"""

import json
import sys
from pathlib import Path

import numpy as np

from train import ROOT


def boost(name, directory):
    d = dict(np.load(ROOT / "cache" / f"{name}.npz"))
    faces = json.loads((Path(directory) / "faces.json").read_text())
    targets = d["targets"].astype(np.float32)
    size = targets.shape[-1]
    grid = (np.arange(size) + 0.5) / size
    count = 0
    for i, n in enumerate(d["names"]):
        for x0, y0, x1, y1, _ in faces.get(str(n), []):
            # Soft-edged box, feathered over about two pixels
            edge = 2 / size
            wx = np.clip(np.minimum(grid - x0, x1 - grid) / edge + 0.5, 0, 1)
            wy = np.clip(np.minimum(grid - y0, y1 - grid) / edge + 0.5, 0, 1)
            targets[i] = np.maximum(targets[i], 255 * wy[:, None] * wx[None, :])
            count += 1
    d["targets"] = targets.round().clip(0, 255).astype(np.uint8)
    d["attention"] = True
    np.savez(ROOT / "cache" / f"{name}-F.npz", **d)
    print(f"{name}-F: {count} heads in {sum(1 for n in d['names'] if faces.get(str(n)))} of {len(d['names'])} images")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, directory = arg.split(":")
        boost(name, directory)
