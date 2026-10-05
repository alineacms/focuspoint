"""Train the importance model on SALICON train (attention) and DUTS-TR (masks).

Every 20th image of each set is held out for validation and early stopping.
The evaluation sets are never touched here.

Usage: python train.py <run-name> [--epochs 40] [--lr 0.004] [--peak 1] [--kept 1]
       python train.py <run-name> --init <run> --qat 5 [--epochs 4] [--lr 0.0003]

The loss is KL divergence to the normalised target, plus optionally
differentiable versions of the two crop metrics (see eval/metrics.ts): for
each container shape, crop positions are soft-selected by how much of the
predicted map they keep, as src/place.ts does, and scored on the ground
truth (kept) or on the ground truth's most important spot (peak, the main
metric). Runs are selected on validation peak when it is trained.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from codec import log_scale

from model import Net, Teacher

ROOT = Path(__file__).parent
SOURCES = ["SALICON-TR", "DUTS-TR"]


def config(run):
    """Input size and widths of a run, as train.py saved them."""
    path = ROOT / "runs" / run / "config.json"
    c = json.loads(path.read_text()) if path.exists() else {}
    return {
        "input": c.get("input", 64),
        "widths": c.get("widths", [16, 24, 48, 64]),
        "arch": c.get("arch", "tiny"),
        "cache": c.get("cache", 128),
    }


def build(c):
    """The network a run config describes."""
    return Teacher() if c["arch"] == "teacher" else Net(c["widths"])


def scale(c):
    """Output size relative to input size."""
    return 4 if c["arch"] == "teacher" else 2


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(name, dev, cache=128):
    d = np.load(ROOT / "cache" / (f"{name}.npz" if cache == 128 else f"{name}@{cache}.npz"))
    images = torch.from_numpy(d["images"]).to(dev)
    targets = torch.from_numpy(d["targets"]).to(dev)
    sizes = torch.from_numpy(d["sizes"]).float().to(dev)
    aspects = sizes[:, 0] / sizes[:, 1]
    val = torch.zeros(len(images), dtype=torch.bool, device=dev)
    val[::20] = True
    return {
        "train": (images[~val], targets[~val], aspects[~val]),
        "val": (images[val], targets[val], aspects[val]),
    }


def linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def batch(images, targets, augment, aspects=None, size=64, out=None):
    """uint8 caches (N,128,128,3) and (N,128,128) to model input and target.
    With aspects (width / height per image), also returns the aspects after
    cropping."""
    x = linear(images.permute(0, 3, 1, 2).float() / 255)
    t = targets[:, None].float() / 255
    n = x.shape[0]
    if augment:
        dev = x.device
        # Crop a random window, often with a different aspect, and flip
        sw = torch.empty(n, device=dev).uniform_(0.65, 1)
        sh = torch.empty(n, device=dev).uniform_(0.65, 1)
        full = torch.rand(n, device=dev) < 0.3
        sw = torch.where(full, torch.ones_like(sw), sw)
        sh = torch.where(full, torch.ones_like(sh), sh)
        cx = (torch.rand(n, device=dev) * 2 - 1) * (1 - sw)
        cy = (torch.rand(n, device=dev) * 2 - 1) * (1 - sh)
        flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0] = sw * flip
        theta[:, 0, 2] = cx
        theta[:, 1, 1] = sh
        theta[:, 1, 2] = cy
        res = images.shape[1]
        grid = F.affine_grid(theta, (n, 1, res, res), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
        t = F.grid_sample(t, grid, padding_mode="border", align_corners=False)
        if aspects is not None:
            aspects = aspects * sw / sh
    x = srgb(shrink(x, size))
    t = shrink(t, out or size // 2)[:, 0]
    if augment:
        # Brightness, contrast and saturation jitter, sometimes greyscale
        r = lambda lo, hi: torch.empty(n, 1, 1, 1, device=x.device).uniform_(lo, hi)
        grey = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
        sat = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < 0.1, torch.zeros_like(r(0, 1)), r(0.6, 1.4))
        x = grey + (x - grey) * sat
        mean = x.mean((1, 2, 3), keepdim=True)
        x = (x - mean) * r(0.7, 1.3) + mean * r(0.75, 1.25)
        x = x.clamp(0, 1)
    return (x, t) if aspects is None else (x, t, aspects)


class FakeQuant(nn.Module):
    """Quantise weights exactly as export.py stores them, with a straight-
    through gradient, for quantisation-aware fine-tuning."""

    def __init__(self, bits):
        super().__init__()
        self.top = 2 ** (bits - 1) - 1

    def forward(self, w):
        flat = w.reshape(w.shape[0], -1)
        scale = log_scale(flat.detach().abs().max(1).values.clamp_min(1e-12) / self.top)[1]
        q = torch.round(flat / scale[:, None]).clamp(-self.top, self.top) * scale[:, None]
        return w + (q.view_as(w) - w).detach()


def prepare_qat(net, bits):
    """Fold batch norm into the convolutions (it becomes a trainable bias)
    and quantise their weights in every forward pass."""
    from export import fold, layers

    for conv, bn in layers(net):
        w, b = fold(conv, bn)
        with torch.no_grad():
            conv.weight.copy_(w)
            if bn is None:
                conv.bias.copy_(b)
            else:
                bn.weight.fill_(1)
                bn.weight.requires_grad_(False)
                bn.bias.copy_(b)
                bn.running_mean.zero_()
                bn.running_var.fill_(1 - bn.eps)
        parametrize.register_parametrization(conv, "weight", FakeQuant(bits))


def freeze_norms(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def shrink(x, size):
    """Area-average to size x size. MPS only pools by whole factors, so other
    sizes go through a bilinear step to twice the size first."""
    if x.shape[-1] % size:
        x = F.interpolate(x, size=2 * size, mode="bilinear", align_corners=False)
    return F.avg_pool2d(x, x.shape[-1] // size)


def kl(logits, t):
    p = t.flatten(1)
    mass = p.sum(1)
    keep = mass > 1e-6
    p = p / mass.clamp_min(1e-6)[:, None]
    logq = F.log_softmax(logits.flatten(1), 1)
    loss = (p * (torch.log(p + 1e-12) - logq)).sum(1)
    return (loss * keep).sum() / keep.sum().clamp_min(1)


CONTAINERS = [3, 16 / 9, 1, 4 / 5, 9 / 16]
POSITIONS = torch.linspace(0, 1, 41)


def windows(profile, frac, pos):
    """Share of a profile (N,B) inside windows of width frac (N,) centred
    on pos (K,) and clamped to 0..1, with linear interpolation: (N,K)."""
    n, bins = profile.shape
    cum = F.pad(profile.cumsum(1), (1, 0))
    start = (pos[None] - frac[:, None] / 2).clamp_min(0)
    start = torch.minimum(start, (1 - frac)[:, None])

    def at(t):
        f = (t * bins).clamp(0, bins)
        i = f.floor().clamp(max=bins - 1)
        lo = cum.gather(1, i.long())
        hi = cum.gather(1, i.long() + 1)
        return lo + (hi - lo) * (f - i)

    return at(start + frac[:, None]) - at(start)


def kept(logits, t, aspects, tau=0.01):
    """Differentiable share of ground truth kept by the crops the predicted
    map would choose, averaged over container shapes."""
    n = logits.shape[0]
    q = F.softmax(logits.flatten(1), 1).view_as(logits)
    p = t / t.flatten(1).sum(1).clamp_min(1e-6)[:, None, None]
    pos = POSITIONS.to(logits.device)
    total = 0
    for a in CONTAINERS:
        a = torch.full_like(aspects, a)
        # Narrower container: keep a/A of the width; wider: A/a of the height
        narrow = a < aspects
        frac = torch.where(narrow, a / aspects, aspects / a).clamp(max=1)
        qp = torch.where(narrow[:, None], q.sum(1), q.sum(2))
        pp = torch.where(narrow[:, None], p.sum(1), p.sum(2))
        w = F.softmax(windows(qp, frac, pos) / tau, 1)
        total = total + (w * windows(pp, frac, pos)).sum(1)
    return total / len(CONTAINERS)


def spot(t, sigma=0.75):
    """A small blob at the most important spot of each target, found as
    eval/metrics.ts does: the peak of the lightly blurred map, with ties
    (flat masks) going to the tied position nearest the tied area's middle."""
    n, h, w = t.shape
    blurred = F.avg_pool2d(t[:, None], 3, 1, 1, count_include_pad=False)[:, 0].flatten(1)
    tied = blurred >= blurred.max(1, keepdim=True).values * 0.99
    gy, gx = torch.meshgrid(torch.arange(h, device=t.device).float(), torch.arange(w, device=t.device).float(), indexing="ij")
    gy, gx = gy.flatten()[None], gx.flatten()[None]
    count = tied.sum(1, keepdim=True)
    cy = (gy * tied).sum(1, keepdim=True) / count
    cx = (gx * tied).sum(1, keepdim=True) / count
    d = torch.where(tied, (gy - cy) ** 2 + (gx - cx) ** 2, torch.full_like(blurred, float("inf")))
    i = d.argmin(1)
    y, x = (i // w).float(), (i % w).float()
    return torch.exp(-((gy - y[:, None]) ** 2 + (gx - x[:, None]) ** 2) / (2 * sigma**2)).view(n, h, w)


def region(t, share=0.5):
    """The most important region of each target: where it stays above half
    its peak (a face, for attention-like maps)."""
    return (t >= t.flatten(1).max(1).values[:, None, None] * share).float()


@torch.no_grad()
def validate(net, data, size, res):
    net.eval()
    out = {}
    for name, d in data.items():
        images, targets, aspects = d["val"]
        losses, keeps, peaks = [], [], []
        for i in range(0, len(images), 256):
            x, t = batch(images[i : i + 256], targets[i : i + 256], False, size=size, out=res)
            logits = net(x)
            a = aspects[i : i + 256]
            losses.append(kl(logits, t).item() * len(x))
            keeps.append(kept(logits, t, a, tau=1e-3).sum().item())
            peaks.append(kept(logits, spot(t), a, tau=1e-3).sum().item())
        out[name] = sum(losses) / len(images)
        out[name + " kept"] = sum(keeps) / len(images)
        out[name + " peak"] = sum(peaks) / len(images)
    net.train()
    return out


def plain(net):
    """State dict without quantisation parametrisations: the float weights
    behind them, which export.py quantises the same way again."""
    state = net.state_dict()
    return {k.replace("parametrizations.weight.original", "weight"): v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kept", type=float, default=0)
    ap.add_argument("--peak", type=float, default=0)
    ap.add_argument("--head", type=float, default=0, help="weight of keeping the whole top region in every crop")
    ap.add_argument("--input", type=int, default=64)
    ap.add_argument("--widths", default="16,24,48,64")
    ap.add_argument("--arch", default="tiny", choices=["tiny", "teacher"])
    ap.add_argument("--cache", type=int, default=128)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--val", help="validate and select on these sets instead (default: the sources)")
    ap.add_argument("--init", help="start from this run's best checkpoint")
    ap.add_argument("--checkpoint", help="with --init: a specific checkpoint file instead")
    ap.add_argument("--qat", type=int, default=0, help="quantisation-aware training at this many bits")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = device()
    data = {name: load(name, dev, args.cache) for name in args.sources.split(",")}
    checks = {name: data.get(name) or load(name, dev, args.cache) for name in (args.val or args.sources).split(",")}
    widths = [int(w) for w in args.widths.split(",")]
    c = {**vars(args), "input": args.input, "widths": widths}
    if args.init:
        c = {**config(args.init), **{k: v for k, v in c.items() if k not in ("input", "widths", "arch", "cache")}}
    net = build(c)
    if args.init:
        net.load_state_dict(torch.load(args.checkpoint or ROOT / "runs" / args.init / "best.pt", map_location="cpu"))
    if args.qat:
        prepare_qat(net, args.qat)
    net = net.to(dev)
    res = c["input"] // scale(c)
    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(c))
    steps = args.epochs * sum(len(d["train"][0]) for d in data.values()) // args.batch
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
    )
    half = args.batch // len(SOURCES)
    best = math.inf
    log = open(out / "log.txt", "a")
    t0 = time.time()
    per_epoch = steps // args.epochs
    for step in range(steps):
        xs, ts, aspect = [], [], []
        for d in data.values():
            images, targets, aspects = d["train"]
            idx = torch.randint(len(images), (half,), device=dev)
            x, t, a = batch(images[idx], targets[idx], True, aspects[idx], c["input"], res)
            xs.append(x)
            ts.append(t)
            aspect.append(a)
        if args.qat:
            freeze_norms(net)
        logits, t = net(torch.cat(xs)), torch.cat(ts)
        loss = kl(logits, t)
        if args.kept:
            loss = loss + args.kept * (1 - kept(logits, t, torch.cat(aspect)).mean())
        if args.peak:
            loss = loss + args.peak * (1 - kept(logits, spot(t), torch.cat(aspect)).mean())
        if args.head:
            # Any share of the region cut off costs, so crops learn to keep it whole
            loss = loss + args.head * (1 - kept(logits, region(t), torch.cat(aspect))).clamp_min(0).pow(0.5).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % per_epoch == 0:
            v = validate(net, checks, c["input"], res)
            # Select on peak, or kept, when trained, else on KL
            metric = "peak" if args.peak else "kept" if args.kept else None
            score = -sum(x for k, x in v.items() if k.endswith(f" {metric}")) if metric else sum(x for k, x in v.items() if " " not in k)
            line = f"epoch {(step + 1) // per_epoch} loss {loss.item():.4f} val {json.dumps({k: round(x, 4) for k, x in v.items()})} {time.time() - t0:.0f}s"
            torch.save(plain(net), out / "last.pt")
            if score < best:
                best = score
                torch.save(plain(net), out / "best.pt")
                line += " *"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
