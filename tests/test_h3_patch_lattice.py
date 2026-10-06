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
    area = math.sqrt(h * w)
    yy, xx = torch.meshgrid(
        (torch.arange(h) - 0.5) * 32 / area + (1 - h / area) * 16,
        (torch.arange(w) - 0.5) * 32 / area + (1 - w / area) * 16,
        indexing="ij",
    )
    return torch.stack((yy, xx))[None, :, None]


@pytest.mark.parametrize(
    "source,target",
    [
        ((46, 40), (66, 58)),
        ((54, 36), (76, 50)),
        ((36, 54), (50, 76)),
        ((32, 32), (48, 48)),
    ],
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
def test_dense_features_batch_and_temporal_identity_survive_transport(dtype):
    value = torch.arange(24, dtype=dtype).reshape(2, 3, 4, 1, 1).expand(2, 3, 4, 8, 10)
    before = value.clone()
    result = lattice.resize_h3_patch_lattice(value, 12, 16)
    assert result.dtype == dtype
    expected = value[..., :1, :1].expand(2, 3, 4, 12, 16)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-5)
    assert torch.equal(value, before)
    assert lattice.resize_h3_patch_lattice(value, 8, 10) is value


@pytest.mark.parametrize("source,target", [(36, 50), (54, 76), (40, 58)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("phase", [0, 1])
def test_single_edge_has_one_contiguous_unimodal_support(source, target, axis, phase):
    value = torch.zeros(1, 1, 1, source, source)
    position = source // 2 // 2 * 2 + phase
    if axis == 0:
        value[..., position, :] = 1
    else:
        value[..., :, position] = 1
    result = lattice.resize_h3_patch_lattice(value, target, target)[0, 0, 0]
    profile = result[:, target // 2] if axis == 0 else result[target // 2]
    support = (profile > 1e-5).nonzero().flatten()
    assert support.numel() >= 1
    assert int(support[-1] - support[0] + 1) == support.numel()
    peak = int(profile.argmax())
    assert bool((profile[: peak + 1].diff() >= -1e-5).all())
    assert bool((profile[peak:].diff() <= 1e-5).all())


@pytest.mark.parametrize("source,target", [((36, 54), (50, 76)), ((54, 36), (76, 50))])
def test_dense_coordinate_spacing_is_uniform_and_patch_centers_match(source, target):
    mapped = lattice.resize_h3_patch_lattice(_coordinates(*source), *target)
    expected = _coordinates(*target)
    interior = (..., slice(4, -4), slice(4, -4))
    torch.testing.assert_close(mapped[interior], expected[interior], rtol=0, atol=1e-5)
    patch_centers = F.avg_pool2d(mapped[0, :, 0], 2)
    gh, gw = target[0] // 2, target[1] // 2
    area = math.sqrt(gh * gw)
    yy, xx = torch.meshgrid(
        torch.arange(gh) * 32 / area + (1 - gh / area) * 16,
        torch.arange(gw) * 32 / area + (1 - gw / area) * 16,
        indexing="ij",
    )
    torch.testing.assert_close(
        patch_centers[:, 2:-2, 2:-2],
        torch.stack((yy, xx))[:, 2:-2, 2:-2],
        rtol=0,
        atol=2e-5,
    )


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
    assert provider.h3_patch_lattice_api == 2
    assert len(calls) == 1
    assert calls[0]["spatial_lattice"] == lattice.H3_PATCH_LATTICE


def test_cached_network_uses_current_geometry_and_repeats_without_state_drift(
    monkeypatch,
):
    torch.manual_seed(816)
    model = (
        upscaler.LatentResizer3D(
            in_blocks=1,
            out_blocks=1,
            channels=32,
            dropout=0,
            temporal_every=1,
            temporal_kernel=3,
        )
        .eval()
        .requires_grad_(False)
    )
    loads = []
    monkeypatch.setattr(
        upscaler, "load_model", lambda *args: loads.append(args) or model
    )
    source = torch.randn(1, 24, 3, 8, 12)
    before = source.clone()
    provider = provider_module.H3LatentUpscalerProvider(
        "cached", device="cpu", precision="fp32"
    )
    monkeypatch.setattr(provider_module, "_lbh_module", lambda: upscaler)
    first = provider.upscale_clean_video_h3_patch_lattice(
        source, target_h=12, target_w=18
    )
    landscape = provider.upscale_clean_video_h3_patch_lattice(
        source.transpose(-1, -2), target_h=18, target_w=12
    )
    repeated = provider.upscale_clean_video_h3_patch_lattice(
        source, target_h=12, target_w=18
    )
    assert first.shape == repeated.shape == (1, 24, 3, 12, 18)
    assert landscape.shape == (1, 24, 3, 18, 12)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert torch.equal(source, before)
    assert len(loads) == 3 and all(args == loads[0] for args in loads)


@pytest.mark.parametrize(
    "source,target", [((32, 48), (48, 72)), ((48, 32), (72, 48)), ((16, 16), (30, 30))]
)
def test_rope_box_lattice_is_the_trained_half_pixel_map_for_equal_aspect(
    source, target
):
    torch.manual_seed(931)
    value = torch.randn(1, 3, 2, *source)
    box = lattice.resize_h3_patch_lattice(
        value, *target, lattice=lattice.H3_ROPE_BOX_LATTICE
    )
    half_pixel = F.interpolate(
        value, size=(2, *target), mode="trilinear", align_corners=False
    )
    # Only float32 coordinate rounding separates the two constructions.
    torch.testing.assert_close(box, half_pixel, rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    "source,target", [((32, 44), (54, 72)), ((44, 32), (72, 54)), ((36, 54), (50, 76))]
)
def test_rope_box_and_patch_center_lattices_differ_by_one_constant_translation(
    source, target
):
    rows = torch.arange(source[0], dtype=torch.float32)[:, None].expand(*source)
    cols = torch.arange(source[1], dtype=torch.float32)[None, :].expand(*source)
    index = torch.stack((rows, cols))[None, :, None]
    center = lattice.resize_h3_patch_lattice(index, *target)
    box = lattice.resize_h3_patch_lattice(
        index, *target, lattice=lattice.H3_ROPE_BOX_LATTICE
    )
    # Away from the border clamp, both maps are affine in the source index.
    interior = (..., slice(3, -3), slice(3, -3))
    delta = (box - center)[interior]
    area_s = math.sqrt(source[0] * source[1])
    area_t = math.sqrt(target[0] * target[1])
    expected = 1.0 - area_s / area_t  # (1 - target_step / source_step) source cells
    torch.testing.assert_close(
        delta, torch.full_like(delta, -expected), rtol=0, atol=1e-4
    )


def test_unknown_transport_lattice_is_rejected():
    with pytest.raises(ValueError, match="Unsupported H3 transport lattice"):
        lattice.resize_h3_patch_lattice(
            torch.zeros(1, 1, 1, 4, 4), 6, 6, lattice=lattice.HALF_PIXEL_LATTICE
        )


def test_provider_selects_the_requested_transport_lattice(monkeypatch):
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
    assert provider.h3_transport_lattices == (
        lattice.H3_PATCH_LATTICE,
        lattice.H3_ROPE_BOX_LATTICE,
    )
    provider.upscale_clean_video_h3_patch_lattice(
        torch.zeros(1, 24, 2, 4, 4),
        target_h=6,
        target_w=6,
        spatial_lattice=lattice.H3_ROPE_BOX_LATTICE,
    )
    assert calls[0]["spatial_lattice"] == lattice.H3_ROPE_BOX_LATTICE
    with pytest.raises(ValueError, match="Unsupported H3 transport lattice"):
        provider.upscale_clean_video_h3_patch_lattice(
            torch.zeros(1, 24, 2, 4, 4),
            target_h=6,
            target_w=6,
            spatial_lattice=lattice.HALF_PIXEL_LATTICE,
        )
    assert len(calls) == 1


def test_network_transports_encoded_features_on_the_rope_box_lattice():
    torch.manual_seed(814)
    model = upscaler.LatentResizer3D(
        in_blocks=1,
        out_blocks=1,
        channels=32,
        dropout=0,
        temporal_every=1,
        temporal_kernel=3,
    ).eval()
    encoded, decoder_inputs = [], []
    handle_in = model.in_blocks[-1].register_forward_hook(
        lambda _m, _i, output: encoded.append(output.clone())
    )
    handle_out = model.out_blocks[0].register_forward_pre_hook(
        lambda _m, inputs: decoder_inputs.append(inputs[0].clone())
    )
    try:
        with torch.no_grad():
            model(
                torch.randn(1, 24, 3, 8, 10),
                scale=1.5,
                target_size=(3, 12, 16),
                spatial_lattice=lattice.H3_ROPE_BOX_LATTICE,
            )
        torch.testing.assert_close(
            decoder_inputs[0],
            lattice.resize_h3_patch_lattice(
                encoded[0], 12, 16, lattice=lattice.H3_ROPE_BOX_LATTICE
            ),
            rtol=0,
            atol=0,
        )
    finally:
        handle_in.remove()
        handle_out.remove()
