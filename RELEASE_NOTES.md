# Unreleased: RoPE-box transport lattice

- Add `h3_rope_box_half_pixel_lattice_v1` as a second H3 transport lattice,
  selected through `upscale_clean_video_h3_patch_lattice(...,
  spatial_lattice=...)` and advertised by `h3_transport_lattices`.
- The new lattice reads H3 patch coordinates as interval starts inside the
  centered RoPE frame box. It matches the trained half-pixel map whenever source
  and target aspect ratios agree, and differs from
  `h3_dense_patch_center_lattice_v2` by one constant sub-cell translation.
- `h3_patch_lattice_api` stays `2` and the default lattice is unchanged, so
  existing callers keep their behavior. Which lattice matches the model's
  cross-resolution placement is an empirical question for the consumer.

# MiniMax H3 Latent Upscaler-Plus v0.2.2

Provide continuous H3 encoder-to-decoder feature transport for the coordinated
Flow handoff, preserving ordinary learned-upscaling behavior.

## Continuous physical lattice

- Resample the complete dense encoder field before decoder convolutions on
  coordinates whose adjacent-cell means match native H3 patch centers.
- Advertise `h3_patch_lattice_api=2` / `h3_dense_patch_center_lattice_v2` so Flow
  uses the same map for its source-prefix carrier.
- Avoid the reproduced edge-duplication mechanism from independent even/odd
  spatial resampling.
- Preserve ordinary half-pixel upscaling, checkpoint parameter keys,
  normalization, complete temporal network execution and cache/offload behavior.

Flow's target-grid continuation profile bypasses the checkpoint at the masked
handoff, while all-generated first chunks still use learned progressive
transfer. The map tests prove the operator correction; they do not qualify
heterogeneous trained-checkpoint continuation as visually fixed.

## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.9](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.9) | [#89](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/89), [#93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93) |
| Sol-H3 | [v0.1.8](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.8) | [#37](https://github.com/xmarre/ComfyUI-Sol-H3/pull/37) |
| VDN-H3-Plus | [v1.5.7](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.7) | [#33](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/33), [#34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34), [#35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35), [#36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), [#37](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/37) |
| H3 Continuum-Plus | [v3.4.5](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.5) | [#37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), [#38](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/38) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | [#16](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/pull/16) |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
It remains an upstream review item, with upstream workflow approval and merge
controlled by Comfy-Org maintainers. For INT8 fused MLP runtime adapters,
retain that ComfyUI Patcher PR overlay until the repair is available upstream.
The independent Core #16720 optimization is not included in this release set.

---

# Unreleased: continuous H3 transfer

The selected physical handoff path resamples dense encoder cells continuously,
preserving H3 patch-center coordinates without separating even/odd spatial
phases. This removes a reproduced edge-duplication mechanism. The capability is
`h3_patch_lattice_api=2` and requires the matching Flow companion overlay.
Ordinary upscaling retains its existing interpolation. Trained-checkpoint
continuity remains under runtime qualification.

# MiniMax H3 Latent Upscaler v0.2.1

v0.2.1 aligns the sampler-internal learned-upscaler provider defaults with the coordinated Flow-Aligned Regenerate progressive workflow.

## Provider defaults

**MiniMax H3 Latent Upscaler Provider (3D) [Experimental]** now defaults to:

```text
model_name            = minimax_h3_latent_upscaler_3d_bf16.safetensors  # when installed
device                = cuda
precision             = bf16
offload_after_upscale = false
```

Checkpoint discovery remains authoritative. The preferred bf16 checkpoint is selected only when that exact file is present; the provider does not invent a missing model entry. Other installed checkpoints, CPU execution, and fp32/fp16 precision remain selectable.

The immutable `H3LatentUpscalerProvider` direct-construction default also changes from fp16 to bf16 so direct consumers and the ComfyUI node do not silently diverge.

## Compatibility

The provider API version, exact-target clean-video contract, cache/offload behavior, and H3 NFE semantics are unchanged. This release changes defaults only; explicit existing workflow values continue to be honored.

---

# MiniMax H3 Latent Upscaler v0.2.0

v0.2.0 adds a versioned sampler-internal learned handoff provider for coordinated MiniMax H3 progressive generation while preserving the existing standalone and integrated-refine workflows.

## Learned handoff provider

- Adds the immutable API-v1 `H3_LATENT_UPSCALER` provider for sampler-internal clean-video spatial transfer.
- Adds an exact-target 3D helper for `B×24×T×H×W` video tensors.
- Preserves package-owned checkpoint discovery, model caching, precision/device policy, and optional `offload_after_upscale` behavior.
- Rejects invalid API versions, channel/temporal changes, spatial shrink, and unavailable configured devices rather than silently falling back.
- The provider never accepts audio and never invokes H3 sampling.

## Coordinated Flow-Aligned Regenerate validation

Decoded-media validation was completed in the coordinated `MiniMax-H3-Flow-Aligned-Regenerate` progressive Target Input path.

- Around a `1152×864` (~0.995 MP) target, replacing aggressive bicubic handoff transfer with `learned_3d` fixed the majority of the observed body/spatial handoff artifacts in the tested difficult prompt.
- `source_scale=0.70` resolved to `800×608 → 1152×864` and was judged excellent.
- `source_scale=0.65` resolved to `736×576 → 1152×864` and began losing reference likeness / tonal stability, so it is not promoted.
- A final `source_scale=0.70` gate resolved to `832×640 → 1184×896` (~1.061 MP). The generated action differed, but decoded quality was again judged very good.
- BF16 CUDA learned inference took about 0.60 s and 0.77 s for the two physical chunks in that final run and added zero H3 NFEs.

These results validate the coordinated progressive boundary for the tested prompt. They are not a universal claim that learned transfer is better for every consumer, prompt, scale, or model configuration.

## Compatibility

Existing node behavior remains unchanged unless the new provider is explicitly wired into a consumer. `offload_after_upscale=False` remains the default. The learned handoff provider is additive and does not replace the existing standalone latent-upscale or integrated upscale+refine paths.

## Validation

The release keeps the native ComfyUI compatibility matrix, Ruff/format checks, `compileall`, native source-contract tests, refinement regressions, alignment tests, temporal execution tests, and cache/offload coverage. The provider integration is additionally covered by exact-target shape/channel/temporal/device contract tests.

---

# MiniMax H3 Latent Upscaler v0.1.1

v0.1.1 is a backward-compatible maintenance release that consolidates the post-v0.1.0 upstream review, CI hardening, alignment fixes, learned-model offload controls, and the final sequence-aware H3 Continuum handoff fix.

## Selective LBH upstream sync

LBH's 2026-08-21 upstream changes were reviewed individually rather than merged wholesale.

Adopted with redesign:

- **Dual-axis output alignment** now uses a common pixel grid compatible with both the requested `align` value and H3's 16× VAE grid: `lcm(align, 16)`.
- **`keep_proportion=True` remains supported.** The node chooses a nearby valid aligned W/H pair instead of solving alignment by silently distorting the source aspect ratio.
- **Optional learned-model offload** is available through `offload_after_upscale`, but remains `False` by default to avoid unnecessary CPU↔GPU transfers on repeated/high-VRAM workflows.
- **Cached model restore** moves an optionally offloaded learned upscaler back to the requested device on the next use.

Deliberately not adopted:

- fixed 16-frame temporal chunking with only `temporal_kernel // 2` overlap, because repeated Conv3d/TemporalConv layers plus GroupNorm make it non-equivalent to whole-sequence execution and allow chunk-boundary changes;
- forced CPU offload after every upscale;
- out-of-place normalization/denormalization changes that do not improve arithmetic precision but do add full-latent temporary allocations;
- removal of `keep_proportion`.

## Sequence-aware Continuum offload fix

The sequence-aware `MiniMax H3 Latent Upscaler + Refine (3D)` path now correctly accepts and propagates the inherited `offload_after_upscale` setting.

For list-valued H3 Continuum execution the node now:

1. resolves the per-chunk learned-upscaler model/device/precision settings;
2. performs the learned upscale;
3. optionally offloads only that learned-upscaler cache entry before sampler 2;
4. preserves the exact post-refine continuation carry into the next chunk;
5. leaves the default behavior unchanged when `offload_after_upscale=False`.

This closes the gap where the base integrated node exposed the control but the sequence-aware list execution path did not accept/propagate it correctly.

## Alignment and conditioning reliability

The 3D learned upscale path guarantees that both final output axes satisfy the common requested/H3 VAE alignment grid. The integrated refiner continues to preserve H3 target-conditioning semantics:

- target `minimax_keyframes` follow the enlarged target grid;
- independent `minimax_refs` keep their own latent/RoPE geometry;
- Native-Masked video/audio masks remain paired with the correct Continuum chunk;
- sequence refinement carries the actual post-refine tail rather than a stale low-resolution prefix.

## CI and compatibility hardening

The repository no longer relies on the old isolated copied-test workaround. GitHub Actions now tests against reviewed native ComfyUI source revisions with the repository root imported directly.

The matrix validates:

- Python 3.10, 3.11, 3.12 and 3.13;
- multiple reviewed ComfyUI revisions;
- Ruff on the maintained integration/test surfaces;
- `compileall`;
- native ComfyUI source-contract tests;
- the full refinement regression suite;
- dual-axis/common-grid alignment;
- whole-sequence temporal execution;
- cached-model restore/offload behavior;
- sequence-aware per-chunk offload propagation.

## Documentation corrections

The README now documents the selective-upstream policy, exact alignment semantics and offload tradeoffs. The installation command also correctly clones this `xmarre` fork rather than the LBH upstream repository.

## Upgrade notes

No workflow migration is required. Existing workflows keep the same default runtime behavior. `offload_after_upscale` remains disabled unless explicitly enabled.
