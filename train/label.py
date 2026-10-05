"""Label images with the teacher for distillation.

Runs a teacher run on cache/<name>@<its cache size>.npz and writes its maps,
peak 255, as the targets of cache/<name>.npz (the 128 px cache the small
model trains on), marked as attention.

Usage: python label.py <teacher-run> <dataset>...
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import importance
from train import ROOT, batch, build, config, device, scale

run, *names = sys.argv[1:]
c = config(run)
dev = device()
net = build(c).to(dev).eval()
net.load_state_dict(torch.load(ROOT / "runs" / run / "best.pt", map_location=dev))
for name in names:
    big = np.load(ROOT / "cache" / f"{name}@{c['cache']}.npz")
    path = ROOT / "cache" / f"{name}.npz"
    small = dict(np.load(path))
    assert (small["names"] == big["names"]).all()
    images, targets = big["images"], big["targets"]
    out = []
    with torch.no_grad():
        for i in range(0, len(images), 128):
            # Chunks go to the device one at a time: the cache can be many GB
            chunk = lambda a: torch.from_numpy(a[i : i + 128]).to(dev)
            x, _ = batch(chunk(images), chunk(targets), False, size=c["input"], out=c["input"] // scale(c))
            m = importance(net(x))[:, None]
            out.append((F.interpolate(m, size=128, mode="bilinear", align_corners=False)[:, 0] * 255).round().clamp(0, 255).byte().cpu())
    small["targets"] = torch.cat(out).numpy()
    small["attention"] = True
    np.savez(path, **small)
    print(f"{name}: {len(small['targets'])} labelled")
