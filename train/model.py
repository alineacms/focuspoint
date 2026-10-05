"""A tiny importance-map network that is easy to port to plain TypeScript.

Only a few operations are used: 3x3 convolution (stem), 3x3 depthwise
convolution, 1x1 convolution, ReLU, 2x bilinear upsampling, channel concat
and a global mean. Batch norm is folded into the convolutions at export.

Input: a square sRGB thumbnail in 0..1 (64 or 96 px), squashed from the
image whatever its aspect. Output: logits at half that size; softmax over
all positions gives the importance map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def coords(n, size, device):
    r = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([x, y]).expand(n, 2, size, size)


class Conv(nn.Sequential):
    """Convolution, batch norm, optional ReLU."""

    def __init__(self, cin, cout, k=1, stride=1, groups=1, relu=True):
        layers = [
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class Block(nn.Sequential):
    """Depthwise-separable block: 3x3 depthwise then 1x1."""

    def __init__(self, cin, cout, stride=1):
        super().__init__(Conv(cin, cin, 3, stride, groups=cin), Conv(cin, cout))


class Net(nn.Module):
    def __init__(self, w=(16, 24, 48, 64)):
        super().__init__()
        a, b, c, d = w
        self.stem = Conv(5, a, 3, 2)  # 32
        self.e1 = nn.Sequential(Block(a, b, 2), Block(b, b))  # 16
        self.e2 = nn.Sequential(Block(b, c, 2), Block(c, c))  # 8
        self.e3 = nn.Sequential(Block(c, d, 2), Block(d, d))  # 4
        self.context = Conv(d, c)  # global mean, broadcast into the decoder
        self.d2 = nn.Sequential(Conv(d + c, c), Block(c, c))  # 8
        self.d1 = nn.Sequential(Conv(c + b, b), Block(b, b))  # 16
        self.d0 = nn.Sequential(Conv(b + a, a), Block(a, a))  # 32
        self.head = nn.Conv2d(a, 1, 1)

    def forward(self, x):
        n = x.shape[0]
        x = torch.cat([x * 2 - 1, coords(n, x.shape[-1], x.device)], 1)
        s = self.stem(x)
        f1 = self.e1(s)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        g = self.context(f3.mean((2, 3), keepdim=True))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        y = self.d2(torch.cat([up(f3), f2], 1)) + g
        y = self.d1(torch.cat([up(y), f1], 1))
        y = self.d0(torch.cat([up(y), s], 1))
        return self.head(y)[:, 0]


class Teacher(nn.Module):
    """A larger model on an ImageNet-pretrained MobileNetV3 backbone, used
    only to label extra images for the small model. Output: logits at a
    quarter of the input size."""

    taps = (3, 6, 12, 16)

    def __init__(self, width=64):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        self.features = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        self.lateral = nn.ModuleList(Conv(c, width) for c in (24, 40, 112, 960))
        self.smooth = nn.Sequential(Conv(width + 2, width, 3), Conv(width, width, 3))
        self.head = nn.Conv2d(width, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.taps:
                feats.append(x)
        y = self.lateral[3](feats[3])
        for k in (2, 1, 0):
            f = feats[k]
            y = F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False) + self.lateral[k](f)
        y = torch.cat([y, coords(len(y), y.shape[-1], y.device)], 1)
        return self.head(self.smooth(y))[:, 0]


def importance(logits):
    """Logits to a map with peak 1."""
    flat = logits.flatten(1)
    return torch.exp(flat - flat.max(1, keepdim=True).values).view_as(logits)


if __name__ == "__main__":
    net = Net()
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    macs = 0

    def count(m, i, o):
        global macs
        if isinstance(m, nn.Conv2d):
            macs += o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]

    for m in net.modules():
        m.register_forward_hook(count)
    net.eval()(torch.rand(1, 3, 64, 64))
    print(f"{params} parameters, {macs / 1e6:.2f}M MACs")
