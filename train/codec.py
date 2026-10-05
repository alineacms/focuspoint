"""Text encoding for weights: every value is one base64url character, which
compresses far better than a packed bitstream (see sizes.py)."""

import torch

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def zigzag(q):
    q = q.flatten().to(torch.int64)
    return torch.where(q >= 0, 2 * q, -2 * q - 1).tolist()


def chars(values):
    return "".join(ALPHABET[v] for v in values)


def log_scale(scale):
    """Scales as one character each: quarter-octave steps, rounded up so no
    weight overflows its range."""
    e = torch.ceil(torch.log2(scale) * 4).clamp(-56, 7) + 56
    return e.long().tolist(), 2 ** ((e - 56) / 4)
