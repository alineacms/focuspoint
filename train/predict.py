"""Write a run's importance maps for datasets, for `bun eval/run.ts --maps`.

Maps go to eval/predictions/<run>/<dataset>/<image>.f32: width and height as
uint32, then float32 values (peak 1), little endian. With --truth the ground
truth is written instead, at the model's output size, to check the plumbing.

Usage: python predict.py <run> <dataset>... [--checkpoint path] [--truth]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import importance
from train import ROOT, batch, build, config, device, scale

PREDICTIONS = ROOT.parent / "eval" / "predictions"


def write(path, m):
    m = np.asarray(m, dtype="<f4")
    with open(path, "wb") as f:
        f.write(np.array(m.shape[::-1], dtype="<u4").tobytes())
        f.write(m.tobytes())


@torch.no_grad()
def maps(net, images, targets, truth, size, out):
    for i in range(0, len(images), 256):
        x, t = batch(images[i : i + 256], targets[i : i + 256], False, size=size, out=out)
        if truth:
            yield from (t / t.flatten(1).max(1).values.clamp_min(1e-6)[:, None, None]).cpu().numpy()
        else:
            yield from importance(net(x)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--checkpoint")
    ap.add_argument("--truth", action="store_true")
    args = ap.parse_args()
    dev = device()
    c = config(args.run)
    net = None
    if not args.truth:
        net = build(c).to(dev).eval()
        net.load_state_dict(torch.load(args.checkpoint or ROOT / "runs" / args.run / "best.pt", map_location=dev))
    for name in args.datasets:
        d = np.load(ROOT / "cache" / (f"{name}.npz" if c["cache"] == 128 else f"{name}@{c['cache']}.npz"))
        out = PREDICTIONS / args.run / name
        out.mkdir(parents=True, exist_ok=True)
        images = torch.from_numpy(d["images"]).to(dev)
        targets = torch.from_numpy(d["targets"]).to(dev)
        for n, m in zip(d["names"], maps(net, images, targets, args.truth, c["input"], c["input"] // scale(c))):
            write(out / f"{n}.f32", m)
        print(f"{name}: {len(d['names'])} maps")


if __name__ == "__main__":
    main()
