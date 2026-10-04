"""Versioned clean-video learned-upscaler provider for sampler-internal handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from .h3_patch_lattice import H3_PATCH_LATTICE, HALF_PIXEL_LATTICE

H3_LATENT_UPSCALER_API_VERSION = 1
H3_LATENT_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"
PREFERRED_H3_LATENT_UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_bf16.safetensors"


def _lbh_module():
    from . import minimax_h3_latent_upscaler_3d as lbh

    return lbh


@dataclass(frozen=True, slots=True)
class H3LatentUpscalerProvider:
    """Immutable provider configuration; checkpoint/model cache remains package-owned."""

    model_name: str
    device: str = "cuda"
    precision: str = "bf16"
    offload_after_upscale: bool = False

    api_version: ClassVar[int] = H3_LATENT_UPSCALER_API_VERSION
    kind: ClassVar[str] = H3_LATENT_UPSCALER_KIND
    h3_patch_lattice_api: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name, str)
            or not self.model_name
            or self.model_name.startswith("(")
        ):
            raise ValueError("Select a MiniMax H3 learned-upscaler checkpoint")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("learned-upscaler device must be cuda or cpu")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("learned-upscaler precision must be fp32, fp16, or bf16")
        if not isinstance(self.offload_after_upscale, bool):
            raise TypeError("offload_after_upscale must be boolean")

    @property
    def inference_device(self) -> str:
        """Return the configured device; provider mode never silently falls back."""
        return self.device

    def upscale_clean_video(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
        spatial_lattice: str = HALF_PIXEL_LATTICE,
    ) -> torch.Tensor:
        """Upscale one clean Bx24xTxHxW latent to exact target latent H/W."""
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "MiniMax H3 learned-upscaler provider is configured for CUDA, "
                "but CUDA is unavailable; select CPU explicitly instead"
            )
        kwargs = (
            {}
            if spatial_lattice == HALF_PIXEL_LATTICE
            else {"spatial_lattice": spatial_lattice}
        )
        return _lbh_module().upscale_clean_video_exact(
            video,
            model_name=self.model_name,
            target_h=target_h,
            target_w=target_w,
            device=self.device,
            precision=self.precision,
            offload_after_upscale=self.offload_after_upscale,
            **kwargs,
        )

    def upscale_clean_video_h3_patch_lattice(self, video, *, target_h, target_w):
        """Transport a physical H3 source carrier before decoder convolutions."""
        return self.upscale_clean_video(
            video,
            target_h=target_h,
            target_w=target_w,
            spatial_lattice=H3_PATCH_LATTICE,
        )


class MinimaxH3LatentUpscaler3DProvider:
    """Create a side-input provider for sampler-internal learned handoffs."""

    @classmethod
    def INPUT_TYPES(cls):
        lbh = _lbh_module()
        models = lbh.scan_models()
        model_spec = (
            (models, {"default": PREFERRED_H3_LATENT_UPSCALER_MODEL})
            if PREFERRED_H3_LATENT_UPSCALER_MODEL in models
            else (models,)
        )
        return {
            "required": {
                "model_name": model_spec,
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                "offload_after_upscale": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Move the cached learned upscaler to CPU after each handoff. Leave off "
                            "for repeated chunks/runs when VRAM is available."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("H3_LATENT_UPSCALER",)
    RETURN_NAMES = ("learned_upscaler",)
    FUNCTION = "build"
    CATEGORY = "video/MinimaxH3"
    DESCRIPTION = (
        "Checkpoint/device configuration for one exact-target, clean-video learned latent transform. "
        "Connect as a side input to compatible progressive handoff nodes; this node performs no "
        "sampling and never receives H3 audio."
    )

    def build(self, model_name, device, precision, offload_after_upscale=False):
        return (
            H3LatentUpscalerProvider(
                model_name=model_name,
                device=device,
                precision=precision,
                offload_after_upscale=bool(offload_after_upscale),
            ),
        )


NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DProvider": MinimaxH3LatentUpscaler3DProvider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DProvider": "MiniMax H3 Latent Upscaler Provider (3D) [Experimental]",
}
