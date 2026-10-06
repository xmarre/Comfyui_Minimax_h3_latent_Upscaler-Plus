"""
Minimax H3 Latent Upscaler - ComfyUI inference node (pure 3D conv version)
- New ComfyUI API (comfy_api.latest)
- 3 resize modes: scale by multiplier / target dimensions / megapixels
- Pixel-space alignment + aspect-ratio lock (no distortion)
- Auto-detects model architecture (channels, blocks, temporal layout)
- FP32 / FP16 / BF16 inference, VRAM-optimized
"""
import math
import logging
from numbers import Integral
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
from einops import rearrange
from enum import Enum
from typing import TypedDict

from .h3_patch_lattice import H3_TRANSPORT_LATTICES, HALF_PIXEL_LATTICE, resize_h3_patch_lattice

# Try to import the new API
try:
    from comfy_api.latest import ComfyExtension, io
    from typing_extensions import override
    USE_NEW_API = True
except ImportError:
    USE_NEW_API = False
    # Dummy io module so the file still parses on old ComfyUI versions
    class io:
        class ComfyNode: pass
        class Schema: pass
        class NodeOutput: pass
        class AnyType:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "ANY"
            @staticmethod
            def Output(*args, **kwargs): return args[0] if args else "ANY"
        class Combo:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "COMBO"
        class Float:
            @staticmethod
            def Input(*args, **kwargs): return "FLOAT"
        class Int:
            @staticmethod
            def Input(*args, **kwargs): return "INT"
        class Boolean:
            @staticmethod
            def Input(*args, **kwargs): return "BOOLEAN"
        class DynamicCombo:
            @staticmethod
            def Input(*args, **kwargs): return args[0] if args else "DYNAMIC"
            class Option: pass
    class ComfyExtension: pass
    def override(func): return func

# ==========================================
# Register model folder
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

# Spatial compression factor of the Minimax H3 3D VAE (16x).
# Hidden from the UI on purpose: 1280x704 px -> 80x44 latent.
VAE_DOWNSAMPLE = 16


def _alignment_grid(align):
    """Return a pixel grid compatible with both the requested alignment and H3 VAE."""
    return math.lcm(max(1, int(align)), VAE_DOWNSAMPLE)


def _aligned_pixel_size(w_target, h_target, w_in, h_in, align, keep_proportion):
    """Choose final pixel dimensions on a dual-axis H3-safe alignment grid.

    When aspect-ratio lock is enabled, preserve the historical width-driven
    semantics but choose the nearby grid pair with the smallest aspect-ratio and
    size error instead of letting the second axis fall off the requested grid.
    """
    grid = _alignment_grid(align)
    w_target = max(float(grid), float(w_target))
    h_target = max(float(grid), float(h_target))

    if not keep_proportion:
        return (
            max(grid, int(round(w_target / grid)) * grid),
            max(grid, int(round(h_target / grid)) * grid),
        )

    aspect = float(w_in) / float(h_in)
    ideal_h = w_target / aspect
    w_center = max(1, int(round(w_target / grid)))
    h_center = max(1, int(round(ideal_h / grid)))

    best = None
    for w_units in range(max(1, w_center - 2), w_center + 3):
        for h_units in range(max(1, h_center - 2), h_center + 3):
            width = w_units * grid
            height = h_units * grid
            ratio_error = abs(math.log((width / height) / aspect))
            size_error = abs(width / w_target - 1.0) + abs(height / ideal_h - 1.0)
            score = (
                ratio_error * 4.0 + size_error,
                ratio_error,
                size_error,
                abs(width - w_target) + abs(height - ideal_h),
            )
            if best is None or score < best[0]:
                best = (score, width, height)

    return best[1], best[2]


# ==========================================
# Minimax H3 latent normalization stats (24 channels, from training code)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

# ==========================================
# 3D network components (identical to training code)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

# ==========================================
# Pure-3D backbone (identical to training code)
# ==========================================
class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, spatial_lattice=HALF_PIXEL_LATTICE):
        if spatial_lattice != HALF_PIXEL_LATTICE and spatial_lattice not in H3_TRANSPORT_LATTICES:
            raise ValueError(f"Unsupported spatial lattice: {spatial_lattice!r}")
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        if spatial_lattice in H3_TRANSPORT_LATTICES:
            if size[0] != x.shape[2]:
                raise ValueError("H3 patch-lattice transport preserves the temporal axis")
            logging.getLogger(__name__).info(
                "[H3 learned transfer] spatial_lattice=%s source_hw=%s target_hw=%s temporal=%d "
                "resample_position=encoder_to_decoder",
                spatial_lattice, tuple(x.shape[-2:]), tuple(size[1:]), size[0],
            )
            x = resize_h3_patch_lattice(x, size[1], size[2], lattice=spatial_lattice)
        else:
            x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# Model loading
# ==========================================
MODEL_CACHE = {}
_PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]

def scan_models():
    files = []
    model_dir = get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
    return names if names else [f"(place models in: {model_dir})"]

def _convert_state_tensor(tensor, dtype):
    if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor

def _load_raw_sd(path, device, dtype):
    if path.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(path, framework='pt', device=str(device)) as f:
            keys = list(f.keys())
            has_prefix = any(k.startswith("upscaler.") for k in keys)
            sd = {}
            for k in keys:
                if has_prefix and not k.startswith("upscaler."):
                    continue
                out_key = k[len("upscaler."):] if has_prefix else k
                sd[out_key] = _convert_state_tensor(f.get_tensor(k), dtype)
        return sd

    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = _extract_upscaler_sd(sd)
    return {k: _convert_state_tensor(v, dtype) for k, v in sd.items()}

def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    if any('attn' in k for k in sd): cfg["attn"] = True
    cfg["attn"] = False  # force off at inference for speed/stability
    return cfg

def load_model(name, device, precision):
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in MODEL_CACHE:
        # A cached CUDA model may have been explicitly offloaded after the prior
        # execution. Moving an already-correct model is a no-op in PyTorch.
        return MODEL_CACHE[cache_key].to(device)

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    dtype = _PRECISION_DTYPES.get(precision, torch.float32)
    up_sd = _load_raw_sd(path, device, dtype)
    cfg = _detect_arch(up_sd)

    # Meta construction avoids a second full FP32 CPU model before CUDA inference.
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(up_sd, strict=True, assign=True)
    model.eval().requires_grad_(False)
    del up_sd

    MODEL_CACHE[cache_key] = model
    print(f"[MinimaxH3-3D] Loaded upscale model: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
          f"Precision: {precision} | Device: {device}")
    return model


def upscale_clean_video_exact(
    video,
    *,
    model_name,
    target_h,
    target_w,
    device,
    precision,
    offload_after_upscale=False,
    output_device=None,
    clear_cuda_cache=False,
    scale_embedding=None,
    spatial_lattice=HALF_PIXEL_LATTICE,
):
    """Learned spatial resize to an exact latent H/W while preserving B/C/T.

    This is the public low-level boundary for callers that already resolved their
    target geometry.  It owns the checkpoint normalization exactly once and does
    not accept or produce an H3 audio stream.
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError("MiniMax H3 learned upscale input must be a torch.Tensor")
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(
            "MiniMax H3 learned upscale input must be Bx24xTxHxW, "
            f"got {tuple(video.shape)}"
        )
    if not video.is_floating_point():
        raise TypeError("MiniMax H3 learned upscale input must use a floating-point dtype")
    if min(video.shape) < 1:
        raise ValueError("MiniMax H3 learned upscale input axes must be non-empty")
    if (
        isinstance(target_h, bool)
        or isinstance(target_w, bool)
        or not isinstance(target_h, Integral)
        or not isinstance(target_w, Integral)
    ):
        raise TypeError("learned upscale target H/W must be integers")
    target_h = int(target_h)
    target_w = int(target_w)
    source_h, source_w = map(int, video.shape[-2:])
    if spatial_lattice != HALF_PIXEL_LATTICE and spatial_lattice not in H3_TRANSPORT_LATTICES:
        raise ValueError(f"Unsupported spatial lattice: {spatial_lattice!r}")
    if spatial_lattice in H3_TRANSPORT_LATTICES and any(v % 2 for v in (source_h, source_w, target_h, target_w)):
        raise ValueError("H3 patch-lattice transport requires even spatial axes")
    if target_h < source_h or target_w < source_w:
        raise ValueError("MiniMax H3 learned upscale target must not shrink either spatial axis")
    if precision not in _PRECISION_DTYPES:
        raise ValueError(f"Unsupported learned-upscaler precision: {precision!r}")
    if not isinstance(model_name, str) or not model_name or model_name.startswith("("):
        raise ValueError("Select a MiniMax H3 learned-upscaler checkpoint")
    if device not in {"cuda", "cpu"}:
        raise ValueError("learned-upscaler device must be cuda or cpu")

    if target_h == source_h and target_w == source_w:
        return video.clone()

    requested_device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    compute_dtype = _PRECISION_DTYPES[precision]
    result_device = video.device if output_device is None else torch.device(output_device)
    original_dtype = video.dtype
    temporal = int(video.shape[2])
    effective_scale = (
        ((target_h / source_h) + (target_w / source_w)) / 2.0
        if scale_embedding is None
        else float(scale_embedding)
    )
    if not math.isfinite(effective_scale) or effective_scale <= 0.0:
        raise ValueError("learned-upscaler scale embedding must be finite and positive")

    work = video.to(device=requested_device, dtype=compute_dtype, copy=True)
    model = load_model(model_name, requested_device, precision)
    norm_mean, norm_std = _make_norm_tensors(requested_device, compute_dtype)
    with torch.inference_mode():
        work.sub_(norm_mean).div_(norm_std)
        model_kwargs = {"scale": effective_scale, "target_size": (temporal, target_h, target_w)}
        if spatial_lattice in H3_TRANSPORT_LATTICES:
            model_kwargs["spatial_lattice"] = spatial_lattice
        output = model(work, **model_kwargs)
        del work
        output.mul_(norm_std).add_(norm_mean)

    expected_shape = (
        int(video.shape[0]),
        24,
        temporal,
        target_h,
        target_w,
    )
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"MiniMax H3 learned upscaler returned shape {tuple(output.shape)}; "
            f"expected {expected_shape}"
        )
    if not output.is_floating_point():
        raise TypeError("MiniMax H3 learned upscaler returned a non-floating tensor")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("MiniMax H3 learned upscaler returned NaN or Inf values")

    output = output.to(device=result_device, dtype=original_dtype)
    if requested_device.type == "cuda":
        if offload_after_upscale:
            model.to("cpu")
        if offload_after_upscale or clear_cuda_cache:
            torch.cuda.empty_cache()
    return output

# ==========================================
# ComfyUI node (new API)
# ==========================================
class UpscaleMode(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"
    MEGAPIXELS = "megapixels"

class UpscaleConfig(TypedDict):
    mode: UpscaleMode
    scale: float
    width: int
    height: int
    megapixels: float

class MinimaxH3LatentUpscaler3D(io.ComfyNode):
    """Minimax H3 latent upscaler with pixel-space alignment and aspect-ratio lock."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinimaxH3LatentUpscaler3D",
            display_name="Minimax H3 Latent Upscaler (3D)",
            category="video/MinimaxH3",
            search_aliases=["minimax", "h3", "latent", "upscale", "3d"],
            inputs=[
                io.AnyType.Input("latent", tooltip="Input latent (image or video)."),
                io.Combo.Input("model_name", options=scan_models(), tooltip="Minimax H3 upscale model."),

                io.DynamicCombo.Input(
                    "mode",
                    tooltip="How the target size is computed.",
                    options=[
                        io.DynamicCombo.Option(UpscaleMode.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05, tooltip="Upscale factor."),
                        ]),
                        io.DynamicCombo.Option(UpscaleMode.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1280, min=64, max=4096, step=8, tooltip="Target pixel width."),
                            io.Int.Input("height", default=704, min=64, max=4096, step=8, tooltip="Target pixel height.")
                        ]),
                        io.DynamicCombo.Option(UpscaleMode.MEGAPIXELS, [
                            io.Float.Input("megapixels", default=1.0, min=0.1, max=8.0, step=0.1, tooltip="Target megapixels (1024x1024 = 1MP).")
                        ])
                    ],
                ),

                io.Int.Input(
                    "align",
                    default=32,
                    min=1,
                    max=512,
                    step=1,
                    tooltip=(
                        "Pixel-space alignment. Both output axes are placed on a grid compatible "
                        "with this value and the H3 16x VAE grid."
                    ),
                ),
                io.Boolean.Input(
                    "keep_proportion",
                    default=True,
                    tooltip=(
                        "Preserve the source aspect ratio as closely as the requested alignment grid allows."
                    ),
                ),

                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda"),
                io.Combo.Input("precision", options=["fp32", "fp16", "bf16"], default="fp16"),
                io.Boolean.Input(
                    "offload_after_upscale",
                    default=False,
                    tooltip=(
                        "Move the cached learned upscaler to CPU after inference to free VRAM. "
                        "Leave off for faster repeated runs when VRAM is available."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output("latent", tooltip="Upscaled latent."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, model_name: str, mode: UpscaleConfig,
                align: int, keep_proportion: bool,
                device: str, precision: str, offload_after_upscale: bool = False) -> io.NodeOutput:

        if model_name.startswith('('):
            raise ValueError("Please place model files into the latent_upscale_models directory")

        selected_mode = mode["mode"]
        src = latent["samples"]
        orig_dtype = src.dtype
        was_4d = (src.dim() == 4)

        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        s = src.unsqueeze(2) if was_4d else src

        _batch, _channels, t, h_in, w_in = s.shape
        downsample = VAE_DOWNSAMPLE

        # 1. Theoretical target size in PIXEL space
        if selected_mode == UpscaleMode.SCALE_BY:
            scale_val = mode["scale"]
            w_pixel_target = w_in * downsample * scale_val
            h_pixel_target = h_in * downsample * scale_val
            effective_scale = scale_val
        elif selected_mode == UpscaleMode.TARGET_DIMENSIONS:
            w_pixel_target = float(mode["width"])
            h_pixel_target = float(mode["height"])
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        elif selected_mode == UpscaleMode.MEGAPIXELS:
            mp = mode["megapixels"]
            target_pixels = mp * 1024 * 1024
            aspect_ratio = w_in / h_in
            h_pixel_target = (target_pixels / aspect_ratio) ** 0.5
            w_pixel_target = h_pixel_target * aspect_ratio
            effective_scale = (w_pixel_target / (w_in * downsample) + h_pixel_target / (h_in * downsample)) / 2.0
        else:
            raise ValueError(f"Unsupported mode: {selected_mode}")

        # 2. Use one common grid so neither axis can be alignment-valid and then
        #    become invalid again when snapped to the 16x H3 VAE grid.
        w_pixel_final, h_pixel_final = _aligned_pixel_size(
            w_pixel_target,
            h_pixel_target,
            w_in,
            h_in,
            align,
            bool(keep_proportion),
        )

        # 3. Back to LATENT space (the helper already guarantees VAE divisibility).
        w_out = max(1, int(w_pixel_final // downsample))
        h_out = max(1, int(h_pixel_final // downsample))

        if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
            raise ValueError("This model only supports upscaling (effective scale >= 1.0).")

        if w_out == w_in and h_out == h_in:
            return io.NodeOutput(latent)

        print(f"[MinimaxH3-3D] Latent {w_in}x{h_in} -> {w_out}x{h_out} | "
              f"Pixels {w_out * downsample}x{h_out * downsample} | scale={effective_scale:.3f}")

        # 4. Inference. Keep the full temporal sequence intact: GroupNorm and the
        #    stacked 3D/temporal convolutions make naive temporal chunking non-equivalent.
        out = upscale_clean_video_exact(
            s,
            model_name=model_name,
            target_h=h_out,
            target_w=w_out,
            device=str(dev),
            precision=precision,
            offload_after_upscale=offload_after_upscale,
            output_device="cpu",
            clear_cuda_cache=True,
            scale_embedding=effective_scale,
        )

        if was_4d:
            out = out.squeeze(2)

        out = out.to(dtype=orig_dtype)

        return io.NodeOutput({"samples": out})

# ==========================================
# New-API extension registration
# ==========================================
if USE_NEW_API:
    class MinimaxH3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [MinimaxH3LatentUpscaler3D]

    async def comfy_entrypoint() -> MinimaxH3Extension:
        return MinimaxH3Extension()

# ==========================================
# Registration (both APIs)
# ==========================================
# Legacy mappings: ALWAYS defined so any loader version can pick up the node.
NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscaler3D": MinimaxH3LatentUpscaler3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscaler3D": "Minimax H3 Latent Upscaler (3D)",
}

# New-style extension entrypoint (only when the new API exists).
if USE_NEW_API:
    class MinimaxH3Extension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [MinimaxH3LatentUpscaler3D]

    async def comfy_entrypoint() -> MinimaxH3Extension:
        return MinimaxH3Extension()
