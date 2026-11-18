"""Three-branch tamper segmentation network (RGB + SRM noise + block DCT).

PRIMARY GOAL
    Map an image to a per-pixel map of {background, inpaint_and_rewrite, crop_and_replace}, using three
    different views of the page because each tamper type leaves different traces (see below).

MAJOR FUNCTIONS / CLASSES
    srm_kernels()   3 fixed high-pass filters that expose the noise residual.
    dct_matrix()    8x8 DCT basis used to compute JPEG-style block coefficients.
    Encoder         small 4-stage CNN (strides 1,2,2,2) returning features at 1/2, 1/4, 1/8 scale.
    TamperNet       the full model; forward() = three branches -> cross-gated fusion -> U-Net decoder.
    TamperNet.block_dct()  vectorised 8x8 block DCT of luminance -> 64 channels at 1/8 resolution.

    RGB branch       - what the document looks like (fonts, edges, layout)
    Noise branch     - fixed SRM high-pass filters: paper grain and sensor/scan noise,
                       which inpainting wipes out and pasting replaces
    DCT branch       - log-magnitude of the 8x8 block DCT of luminance, aligned to the
                       JPEG grid: exposes regions with a different compression history

The branches are fused at 1/8 resolution with cross-gating (each branch is scaled
by a sigmoid gate computed from the others), then decoded to a 3-class map at
1/2 resolution: background, inpaint_and_rewrite, crop_and_replace.

Small on purpose (~0.6M parameters) so it trains on a laptop CPU in under an hour.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["background", "inpaint_and_rewrite", "crop_and_replace"]


def srm_kernels() -> torch.Tensor:
    """Three fixed Spatial-Rich-Model style high-pass kernels (3x3 edge, 5x5 'square', horizontal 2nd derivative).
    They cancel image content (text) and keep noise. Returned as a (3,1,5,5) conv weight; never trained.
    """
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]) / 12.0
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0
    return torch.tensor(np.stack([k1, k2, k3])[:, None], dtype=torch.float32)


def dct_matrix(n: int = 8) -> torch.Tensor:
    """Orthonormal DCT-II basis matrix C: for an 8x8 block X, the DCT is C @ X @ C.T."""
    m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            a = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            m[k, i] = a * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return m


def conv(cin, cout, stride=1):
    """Conv3x3 -> BatchNorm -> ReLU, the basic building block."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, cin, w=(16, 32, 48, 64)):
        super().__init__()
        self.s1 = nn.Sequential(conv(cin, w[0]), conv(w[0], w[0]))          # 1/1
        self.s2 = nn.Sequential(conv(w[0], w[1], 2), conv(w[1], w[1]))      # 1/2
        self.s3 = nn.Sequential(conv(w[1], w[2], 2), conv(w[2], w[2]))      # 1/4
        self.s4 = nn.Sequential(conv(w[2], w[3], 2), conv(w[3], w[3]))      # 1/8

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f2, f3, f4


class TamperNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Buffers = saved with the weights and moved with .to(device), but NOT trained (fixed filters).
        self.register_buffer("srm", srm_kernels())
        self.register_buffer("dct", dct_matrix())
        # Three branches: RGB sees appearance, noise branch sees the SRM residual, frequency branch sees JPEG-block DCT.
        self.rgb = Encoder(3)
        self.noise = Encoder(3, (16, 24, 32, 48))
        self.freq = nn.Sequential(conv(64, 64), conv(64, 48))  # operates directly at 1/8
        # Channels after concatenating the three 1/8-scale feature maps (160).
        fused = 64 + 48 + 48
        # Cross-gating: each branch gets a 1x1-conv gate computed from ALL three, so the net learns where to trust which view.
        self.gate_rgb = nn.Conv2d(fused, 64, 1)
        self.gate_noise = nn.Conv2d(fused, 48, 1)
        self.gate_freq = nn.Conv2d(fused, 48, 1)
        self.mid = nn.Sequential(conv(fused, 96), conv(96, 96))
        self.up3 = nn.Sequential(conv(96 + 48 + 32, 64), conv(64, 64))   # 1/4
        self.up2 = nn.Sequential(conv(64 + 32 + 24, 48), conv(48, 48))   # 1/2
        self.head = nn.Conv2d(48, len(CLASSES), 1)

    def block_dct(self, gray: torch.Tensor) -> torch.Tensor:
        """gray: B x 1 x H x W in [0,255], H and W multiples of 8 -> B x 64 x H/8 x W/8."""
        b, _, h, w = gray.shape
        x = gray - 128.0
        x = x.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)        # B,h8,w8,8,8
        c = self.dct @ x @ self.dct.T
        c = c.reshape(b, h // 8, w // 8, 64).permute(0, 3, 1, 2)
        return torch.log1p(c.abs())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: B x 3 x H x W, float in [0,255], H and W multiples of 8. Returns logits at 1/2 res."""
        # BRANCH INPUTS. Luminance (ITU-R BT.601 weights) feeds both the noise and DCT views.
        gray = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
        # Noise view: apply the 3 SRM kernels -> 3 residual channels (scaled to a sane magnitude).
        noise = F.conv2d(gray, self.srm, padding=2) / 8.0
        # Run each branch. RGB is normalised to [-1, 1]; noise and RGB encoders return features at 1/2, 1/4, 1/8.
        r2, r3, r4 = self.rgb(img / 127.5 - 1.0)
        n2, n3, n4 = self.noise(noise)
        # Frequency view: 8x8 block DCT (64 coefficient channels at 1/8 res) -> two convs.
        q = self.freq(self.block_dct(gray) / 4.0)
        # FUSION. Gate each branch by sigmoid(1x1 conv of the concatenation), then process the gated features together.
        cat = torch.cat([r4, n4, q], 1)
        r4 = r4 * torch.sigmoid(self.gate_rgb(cat))
        n4 = n4 * torch.sigmoid(self.gate_noise(cat))
        q = q * torch.sigmoid(self.gate_freq(cat))
        x = self.mid(torch.cat([r4, n4, q], 1))
        # DECODER (U-Net style): upsample, concatenate the same-scale RGB + noise skip features, convolve; repeat
        # once more to reach 1/2 resolution. Skips restore fine edges lost by downsampling.
        x = F.interpolate(x, size=r3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, r3, n3], 1))
        x = F.interpolate(x, size=r2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, r2, n2], 1))
        # 1x1 conv -> 3 class logits per pixel at 1/2 input resolution (upsampled later at inference).
        return self.head(x)


if __name__ == "__main__":
    m = TamperNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    print(m(torch.rand(2, 3, 256, 256) * 255).shape)
