"""Cache a dataset as squashed square arrays for fast training.

Images are squashed (not padded) to SIZE x SIZE, averaging in linear light
like src/image.ts, and stored as sRGB uint8. Ground truth is stored the same
way: subject masks as is, attention maps scaled so their peak is 255. Sets
without ground truth get empty targets, for train/label.py to fill.

Usage: python prep.py [--size 128] <dataset-dir>...
Writes cache/<name>.npz, or cache/<name>@<size>.npz for other sizes.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SIZE = 128
CACHE = Path(__file__).parent / "cache"


def to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255


def area(a, size):
    """Area-average an HxWxC float array to size x size (fractional boxes)."""
    h, w = a.shape[:2]

    def weights(n):
        # m[i, j]: share of source pixel j in output pixel i
        edges = np.linspace(0, n, size + 1)
        m = np.zeros((size, n))
        for i in range(size):
            lo, hi = edges[i], edges[i + 1]
            for j in range(int(lo), min(n, int(np.ceil(hi)))):
                m[i, j] = min(hi, j + 1) - max(lo, j)
            m[i] /= m[i].sum()
        return m

    rows = np.tensordot(weights(h), a, axes=(1, 0))
    return np.tensordot(rows, weights(w), axes=(1, 1)).transpose(0, 2, 1)


def shrink(img, limit):
    # Integer pre-reduction keeps the exact area average cheap
    factor = max(1, min(img.size) // limit)
    return img.reduce(factor) if factor > 1 else img


def safe(args):
    try:
        return load(args)
    except Exception as e:
        print(f"skipping {args[0]}: {e}", file=sys.stderr)
        return None


def load(args):
    # The size travels with the job: spawned workers don't see main's globals
    image_path, truth_path, attention, size = args
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    rgb = area(to_linear(np.asarray(shrink(img, 2 * size), dtype=np.float64)), size)
    if truth_path is None:
        return to_srgb(rgb).round().astype(np.uint8), np.zeros((size, size), np.uint8), img.size
    truth = Image.open(truth_path).convert("L")
    t = area(np.asarray(shrink(truth, 2 * size), dtype=np.float64)[..., None], size)[..., 0]
    peak = t.max() if attention else 255.0
    t = t / peak * 255 if peak > 0 else t
    return to_srgb(rgb).round().astype(np.uint8), np.clip(t, 0, 255).round().astype(np.uint8), img.size


def prep(directory):
    directory = Path(directory)
    attention = (directory / "fixations").exists()
    truth = directory / ("fixations" if attention else "masks")
    images = sorted(p.stem for p in (directory / "images").iterdir() if p.suffix == ".jpg")
    if truth.exists():
        masks = {p.stem: p for p in truth.iterdir() if p.suffix == ".png"}
        names = [n for n in images if n in masks]
    else:
        masks = {}
        names = images
    jobs = [(directory / "images" / f"{n}.jpg", masks.get(n), attention, SIZE) for n in names]
    with Pool() as pool:
        out = pool.map(safe, jobs, chunksize=32)
    kept = [i for i, o in enumerate(out) if o is not None]
    names = [names[i] for i in kept]
    out = [out[i] for i in kept]
    CACHE.mkdir(exist_ok=True)
    np.savez(
        CACHE / (f"{directory.name}.npz" if SIZE == 128 else f"{directory.name}@{SIZE}.npz"),
        names=np.array(names),
        images=np.stack([o[0] for o in out]),
        targets=np.stack([o[1] for o in out]),
        attention=attention,
        sizes=np.array([o[2] for o in out], dtype=np.int32),
    )
    print(f"{directory.name}: {len(names)} images")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--size"]:
        SIZE = int(args[1])
        args = args[2:]
    for d in args:
        prep(d)
