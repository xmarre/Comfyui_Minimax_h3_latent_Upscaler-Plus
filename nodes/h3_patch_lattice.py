"""Continuous dense-feature transport with H3 physical patch centers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

H3_PATCH_LATTICE = "h3_dense_patch_center_lattice_v2"
HALF_PIXEL_LATTICE = "half_pixel_latent_v1"


def _axis(grid_h, grid_w, axis):
    area = math.sqrt(grid_h * grid_w)
    length = (grid_h, grid_w)[axis]
    step = 32 / area
    return length, (1 - length / area) * 16 - step / 2, step


def resize_h3_patch_lattice(value, target_h, target_w):
    """Resample dense cells without separating even/odd spatial phases.

    The mean coordinate of each adjacent pair is the native H3 patch coordinate.
    Encoder Conv3d features are a dense field, not four independent patch lanes.
    Chunk only the interpolation scratch, after all temporal encoder blocks and
    before all temporal decoder blocks. The learned network still runs once on
    the complete temporal sequence.
    """
    if value.ndim != 5 or not value.is_floating_point():
        raise ValueError("H3 lattice resize expects floating BxCxTxHxW features")
    b, c, t, h, w = value.shape
    if min(h, w, target_h, target_w) <= 0 or any(
        v % 2 for v in (h, w, target_h, target_w)
    ):
        raise ValueError("H3 lattice resize requires positive even spatial axes")
    if (h, w) == (target_h, target_w):
        return value
    axes = []
    for axis in (0, 1):
        sn, s0, ds = _axis(h, w, axis)
        tn, t0, dt = _axis(target_h, target_w, axis)
        index = (
            t0 + torch.arange(tn, device=value.device, dtype=torch.float32) * dt - s0
        ) / ds
        axes.append(2 * index / (sn - 1) - 1 if sn > 1 else torch.zeros_like(index))
    yy, xx = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None]
    frames = value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    output = value.new_empty((b * t, c, target_h, target_w))
    # Bound fp32 source+target interpolation scratch to roughly 64 MiB per chunk.
    chunk = max(1, (64 << 20) // (c * (h * w + target_h * target_w) * 4))
    for first in range(0, b * t, chunk):
        last = min(first + chunk, b * t)
        mapped = F.grid_sample(
            frames[first:last].float(),
            grid.expand(last - first, -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        output[first:last] = mapped.to(value.dtype)
    return output.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
