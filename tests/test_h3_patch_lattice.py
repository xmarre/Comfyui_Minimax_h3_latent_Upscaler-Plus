from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PACKAGE = "h3_lattice_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes")]
sys.modules[PACKAGE] = package
lattice = importlib.import_module(f"{PACKAGE}.h3_patch_lattice")
upscaler = importlib.import_module(f"{PACKAGE}.minimax_h3_latent_upscaler_3d")
provider_module = importlib.import_module(f"{PACKAGE}.minimax_h3_handoff_provider")


def _coordinates(h, w):
    gh, gw = h // 2, w // 2
    area = math.sqrt(gh * gw)
    yy, xx = torch.meshgrid(
        torch.arange(gh) * 32 / area + (1 - gh / area) * 16,
        torch.arange(gw) * 32 / area + (1 - gw / area) * 16,
        indexing="ij",
    )
    return (
        torch.stack((yy, xx))[None, :, None]
        .repeat_interleave(2, -2)
        .repeat_interleave(2, -1)
    )


@pytest.mark.parametrize(
    "source,target", [((46, 40), (66, 58)), ((54, 36), (76, 50)), ((32, 32), (48, 48))]
)
def test_physical_roundtrip_preserves_coordinate_fields_and_half_pixel_does_not(
    source, target
):
    exact = _coordinates(*target)
    low = lattice.resize_h3_patch_lattice(exact, *source)
    correct = lattice.resize_h3_patch_lattice(low, *target)
    wrong = F.interpolate(low, size=(1, *target), mode="trilinear", align_corners=False)
    interior = (..., slice(6, -6), slice(6, -6))
    torch.testing.assert_close(correct[interior], exact[interior], rtol=0, atol=1e-5)
    assert float((wrong[interior] - exact[interior]).square().mean().sqrt()) > 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_patch_features_batch_and_temporal_identity_survive_transport(dtype):
    value = torch.zeros(2, 3, 4, 8, 10, dtype=dtype)
    for y in range(2):
        for x in range(2):
            value[..., y::2, x::2] = (
                torch.arange(4)[None, None, :, None, None] + y * 10 + x * 20
            )
    before = value.clone()
    result = lattice.resize_h3_patch_lattice(value, 12, 16)
    assert result.dtype == dtype
    for y in range(2):
        for x in range(2):
            expected = value[..., y : y + 1, x : x + 1].expand(2, 3, 4, 6, 8)
            torch.testing.assert_close(
                result[..., y::2, x::2], expected, rtol=0, atol=1e-5
            )
    assert torch.equal(value, before)
    assert lattice.resize_h3_patch_lattice(value, 8, 10) is value


def test_network_transports_encoded_features_before_decoder_and_keeps_default(
    monkeypatch,
):
    torch.manual_seed(813)
    model = upscaler.LatentResizer3D(
        in_blocks=1,
        out_blocks=1,
        channels=32,
        dropout=0,
        temporal_every=1,
        temporal_kernel=3,
    ).eval()
    video = torch.randn(1, 24, 3, 8, 10)
    encoded, decoder_inputs = [], []
    handle_in = model.in_blocks[-1].register_forward_hook(
        lambda _m, _i, output: encoded.append(output.clone())
    )
    handle_out = model.out_blocks[0].register_forward_pre_hook(
        lambda _m, inputs: decoder_inputs.append(inputs[0].clone())
    )
    try:
        with torch.no_grad():
            native = model(video, scale=1.5, target_size=(3, 12, 16))
            physical = model(
                video,
                scale=1.5,
                target_size=(3, 12, 16),
                spatial_lattice=lattice.H3_PATCH_LATTICE,
            )
        torch.testing.assert_close(
            decoder_inputs[0],
            F.interpolate(
                encoded[0], size=(3, 12, 16), mode="trilinear", align_corners=False
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            decoder_inputs[1],
            lattice.resize_h3_patch_lattice(encoded[1], 12, 16),
            rtol=0,
            atol=0,
        )
        assert native.shape == physical.shape == (1, 24, 3, 12, 16)
        assert not torch.equal(native, physical)
        torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
    finally:
        handle_in.remove()
        handle_out.remove()


def test_provider_passes_lattice_capability_explicitly_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        provider_module,
        "_lbh_module",
        lambda: types.SimpleNamespace(
            upscale_clean_video_exact=lambda value, **kwargs: (
                calls.append(kwargs) or value
            )
        ),
    )
    provider = provider_module.H3LatentUpscalerProvider("m", device="cpu")
    provider.upscale_clean_video_h3_patch_lattice(
        torch.zeros(1, 24, 2, 4, 4), target_h=6, target_w=6
    )
    assert provider.h3_patch_lattice_api == 1
    assert len(calls) == 1
    assert calls[0]["spatial_lattice"] == lattice.H3_PATCH_LATTICE
