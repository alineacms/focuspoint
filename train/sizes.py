"""Compare ways to store a run's weights as JavaScript, by compressed size.

Each candidate is the string literal that would ship in weights.ts. Sizes
are for gzip -9 and brotli -q 11, as a bundler or CDN would serve them.

Usage: python sizes.py <run>
"""

import base64
import gzip
import sys

import brotli
import numpy as np
import torch

from codec import ALPHABET, chars, log_scale, zigzag
from export import fold, layers
from train import build, config

def quantise(w, bits):
    top = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().max(1).values.clamp_min(1e-12) / top
    q = torch.round(flat / scale[:, None]).clamp(-top, top)
    return q.reshape(w.shape), scale


def pack(q, bits):
    """Signed integers to a little-endian bitstream, offset to unsigned."""
    u = (q.flatten().to(torch.int64) + 2 ** (bits - 1)).tolist()
    out = bytearray()
    acc = n = 0
    for v in u:
        acc |= v << n
        n += bits
        while n >= 8:
            out.append(acc & 255)
            acc >>= 8
            n -= 8
    if n:
        out.append(acc & 255)
    return bytes(out)


def candidates(net):
    out = {}
    for bits in (4, 5, 6, 8):
        packed, text, text_log = bytearray(), [], []
        for conv, bn in layers(net):
            w, b = fold(conv, bn)
            q, scale = quantise(w, bits)
            f16 = scale.numpy().astype("<f2").tobytes() + b.numpy().astype("<f2").tobytes()
            packed += pack(q, bits) + f16
            if bits <= 6:
                text.append(chars(zigzag(q)) + base64.b64encode(f16).decode())
                # All text: log scales, weights requantised to them, 12-bit biases
                codes, rounded = log_scale(scale)
                top = 2 ** (bits - 1) - 1
                q2 = torch.round(w.reshape(w.shape[0], -1) / rounded[:, None]).clamp(-top, top)
                bias_scale = b.abs().max().clamp_min(1e-9) / 2047
                bq = zigzag(torch.round(b / bias_scale))
                text_log.append(chars(zigzag(q2)) + chars(codes) + "".join(ALPHABET[v >> 6] + ALPHABET[v & 63] for v in bq))
        out[f"packed {bits}-bit, base64"] = base64.b64encode(bytes(packed)).decode()
        if bits <= 6:
            out[f"chars {bits}-bit, float16 scales"] = "".join(text)
            out[f"chars {bits}-bit, all text"] = "".join(text_log)
    return out


def main():
    run = sys.argv[1]
    net = build(config(run))
    net.load_state_dict(torch.load(f"runs/{run}/best.pt", map_location="cpu"))
    net.eval()
    print(f"{run}: {sum(p.numel() for p in net.parameters())} parameters")
    print(f"{'encoding':34} {'raw':>8} {'gzip':>8} {'brotli':>8}")
    for name, s in candidates(net).items():
        literal = f"export const weights='{s}'\n".encode()
        print(f"{name:34} {len(literal):8} {len(gzip.compress(literal, 9)):8} {len(brotli.compress(literal, quality=11)):8}")


if __name__ == "__main__":
    main()
