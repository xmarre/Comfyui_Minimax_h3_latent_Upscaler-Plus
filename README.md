<p align="center">
  <a href="./README.md"><strong>English</strong></a> ·
  <a href="./README_zh.md">中文</a>
</p>

# ComfyUI MiniMax H3 Latent Upscaler-Plus

Learned spatial upscaling of MiniMax H3 video latents for ComfyUI, with an optional integrated low-sigma H3 refinement pass.

> **Plus fork:** this is the `xmarre`-maintained fork of [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler). It uses the same learned network and checkpoints and adds H3-aware refinement, H3 Continuum interop, a sampler-internal handoff provider and several correctness fixes. Some behavior intentionally differs from upstream; see [Differences from upstream](#differences-from-upstream).

A trained network enlarges the 24-channel H3 video latent directly, so a two-stage workflow can generate at a lower resolution, upscale in latent space and refine briefly at the target resolution without a VAE decode → pixel upscale → VAE encode round trip. Audio never passes through the upscaler.

The learned upscale saves time. It does not reduce the VRAM needed by a later H3 pass, which still runs the transformer on the full target-resolution grid.

**Examples:** [video comparison (MP4)](examples/Minimax_h3_latent_Upscaler_001.mp4)

![Image upscale comparison](examples/Minimax_h3_latent_Upscaler_002.jpg)

## Contents

- [Nodes](#nodes)
- [Installation](#installation)
- [Workflows](#workflows)
- [Behavior](#behavior)
- [Node reference](#node-reference)
- [Provider API](#provider-api)
- [Model and training data](#model-and-training-data)
- [Differences from upstream](#differences-from-upstream)
- [Testing](#testing)
- [Coordinated H3 release set](#coordinated-h3-release-set)
- [Acknowledgments](#acknowledgments)

## Nodes

All nodes are in the `video/MinimaxH3` category.

| Display name | Node ID | Purpose |
| --- | --- | --- |
| Minimax H3 Latent Upscaler (2D) | `MinimaxH3LatentUpscalerNode2D` | Learned upscale with a 2D backbone and temporal layers. Multiplier sizing only. |
| Minimax H3 Latent Upscaler (3D) | `MinimaxH3LatentUpscaler3D` | Fully 3D learned upscale. Multiplier, target-dimension or megapixel sizing with dual-axis alignment. |
| MiniMax H3 Latent Upscaler + Refine (3D) | `MinimaxH3LatentUpscaler3DRefineHandoff` | Learned 3D upscale followed by the H3 refinement sampling pass. Returns a decode-ready LATENT. Handles native joint AV latents and H3 Continuum chunk lists. |
| MiniMax H3 Latent Upscaler Provider (3D) [Experimental] | `MinimaxH3LatentUpscaler3DProvider` | Configuration object for sampler-internal learned handoffs, such as Flow-Aligned Regenerate's `learned_3d` transfer. Performs no sampling. |

The 2D and 3D nodes are plain `LATENT → LATENT` upscalers. They do not run H3.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus.git Comfyui_Minimax_h3_latent_Upscaler
```

Restart ComfyUI afterwards. The only dependency outside a standard ComfyUI install is `einops`, which ComfyUI already ships with.

### Checkpoints

Place checkpoints in `ComfyUI/models/latent_upscale_models/`. The folder is registered automatically and scanned non-recursively for `.safetensors` and `.pth` files.

Pre-trained checkpoints: [huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)

| File | Stored precision |
| --- | --- |
| `minimax_h3_latent_upscaler_3d_bf16.safetensors` | bf16 |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | fp16 |
| `minimax_h3_latent_upscaler_3d_fp32.pth` | fp32 |

These files share one 3D architecture and serve the 3D, Refine and Provider nodes. The Provider preselects the bf16 file when it is present. The `precision` widget sets the inference dtype independently of the stored precision.

The loaders detect block counts, channel width and temporal layers from the checkpoint. The 2D node needs a checkpoint with the 2D backbone layout; a 3D checkpoint fails in the 2D node with a missing-weights error.

## Workflows

### Standalone upscale

The 2D and 3D nodes take a plain 24-channel video latent (`B×24×T×H×W`, or `B×24×H×W` for a single frame). Split a joint H3 audio-video latent first and recombine it afterwards:

```text
joint H3 AV latent
  → LTXVSeparateAVLatent
      video_latent → Minimax H3 Latent Upscaler (3D) ─┐
      audio_latent ───────────────────────────────────┤
  → LTXVConcatAVLatent
  → second H3 pass of your choice, or VAE Decode
```

The workflows in [`workflow_templates/`](workflow_templates/) use this pattern for image-to-video and reference-to-video, followed by an external second pass with `BasicGuider` and `SamplerCustomAdvanced`.

### Upscale and refine in one node

**MiniMax H3 Latent Upscaler + Refine (3D)** replaces the split/upscale/concat/second-sampler chain:

```text
low-resolution H3 latent ─► MiniMax H3 Latent Upscaler + Refine (3D) ─► VAE Decode
                              + noise    (RandomNoise)
                              + sampler  (KSamplerSelect)
                              + sigmas   (partial-denoise schedule, sigmas[0] < 1)
                              + model + positive   (native workflows)
                                or refine_state    (H3 Continuum)
```

For a native workflow, connect the joint AV `latent`, `model` and `positive`. Leave `audio_latent` disconnected. `negative` is optional. Without it the node samples with positive-only guidance, which is the usual H3 setup. With it the node uses CFG at `cfg`.

Do not add an external `BasicGuider`, `DisableNoise` or `SamplerCustomAdvanced`. The node runs the sampling pass itself.

### H3 Continuum

Connect the parallel chunk-list outputs of **H3 Continuum Sampler V3.4** from [H3 Continuum-Plus](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus):

```text
H3 Continuum Sampler V3.4
  video_latents ──► Refine.latent
  audio_latents ──► Refine.audio_latent
  refine_state  ──► Refine.refine_state
```

Connecting `refine_state` makes Continuum capture, per chunk, a fresh MODEL clone carrying that chunk's Continuum options and the exact positive conditioning that sampler 1 received. Continuum also attaches each chunk's video/audio denoise masks to the latent outputs. A connected `refine_state` takes precedence: leftover `model`, `positive` or `negative` connections are ignored, and a malformed state raises an error instead of falling back to them.

The node consumes the whole chunk list in order. When a chunk begins with a fully protected continuation prefix (denoise mask 0 over whole time steps), that prefix is replaced by the final time steps of the previous chunk's *refined* output before sampler 2 runs, so consecutive chunks join on the refined content rather than on independently upscaled copies. Audio prefixes are carried the same way when `lock_audio` is off. Carrying requires identical video geometry across chunks.

Run Storage does not persist refinement state. If Continuum reuses stored chunks while `refine_state` is connected, it raises an error. Disable Run Storage or regenerate from chunk 1.

### Sampler-internal handoff (experimental)

**MiniMax H3 Latent Upscaler Provider (3D)** outputs an `H3_LATENT_UPSCALER` object. Connect it to a consumer that accepts one, such as the Target Input progressive handoff of [MiniMax-H3 Flow-Aligned Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate), and select that consumer's learned transfer mode (`learned_3d`). The consumer calls the provider once per handoff on its clean video estimate at an exact target latent size. The provider never receives audio and runs no H3 steps. Recommended handoff settings are documented by the consumer.

## Behavior

### Output size and alignment

The 3D and Refine nodes compute a pixel-space target from the selected mode:

- `scale by multiplier`: source size × `scale`.
- `target dimensions`: `width` × `height`.
- `megapixels`: `megapixels × 1024²` pixels at the source aspect ratio.

`align` is a pixel-space requirement. H3's VAE also needs a 16-pixel grid, so both axes are placed on `lcm(align, 16)`. The default `align=32` gives a 32-pixel grid. `align=24` gives a 48-pixel grid.

- `keep_proportion=False` rounds width and height to the grid independently.
- `keep_proportion=True` searches nearby grid pairs and picks the one that best preserves the source aspect ratio while staying close to the target. Both axes stay on the grid.

Only upscaling is supported. A target smaller than the source on either axis is rejected. If the result equals the source size, the latent is returned unchanged.

The 2D node scales the latent grid by `scale` with rounding and does no pixel-grid alignment.

### Refinement pass

The Refine node:

1. upscales only the video member with the learned 3D network;
2. optionally moves the learned model to CPU (`offload_after_upscale`);
3. rebuilds the joint H3 AV latent on the enlarged grid;
4. rebuilds the denoise mask on the enlarged grid (see [Audio and masks](#audio-and-masks));
5. resizes target-grid conditioning (see [Conditioning geometry](#conditioning-geometry));
6. generates fresh noise from `noise` for the enlarged AV grid;
7. runs the supplied `sampler` over `sigmas` with ComfyUI's standard guider path. ComfyUI applies the model's own noise scaling, so there is no manual pre-noising and no `DisableNoise` stage.

`sigmas[0]` must satisfy `0 <= sigmas[0] < 1`. Under H3's flow parameterization, a schedule that starts at 1.0 gives the upscaled latent zero weight, so such a schedule is rejected. Use a partial-denoise schedule. An empty `SIGMAS` returns the upscaled latent without sampling.

Sampler 2 runs on a clone of the MODEL whose `transformer_options["h3_refinement"]` marks the call as a refinement pass (API 1, `sigma_reference` = the model's `sigma_max`). Companion runtime patches read this contract, so they treat sampler 2 as a refinement pass rather than a fresh generation. The source MODEL is not modified.

Sampler 2 cost scales with the target token count. A 2× spatial upscale gives about four times as many video tokens per H3 step. Keep the refinement schedule short, and benchmark it on your hardware.

### Audio and masks

`lock_audio` (default on):

- **On**: audio noise is zero, the audio denoise mask is zero, and pass-1 audio is restored exactly after sampling.
- **Off**: audio receives normal noise and is refined with the video. An existing audio denoise mask is kept.

An existing video denoise mask is resized to the enlarged grid with nearest-neighbour sampling. Fully protected regions therefore stay protected, and a Native-Masked continuation prefix is not re-denoised. When no mask exists and `lock_audio` is off, the node samples without a mask.

### Conditioning geometry

- `minimax_keyframes` are target-grid conditions. They are resized to H3's internally padded even latent grid (H3 pads H/W to its 2×2 patch grid).
- `minimax_refs` are independent reference blocks with their own latent size and RoPE grid. They are left unchanged.
- Conditioning is copied, never mutated. Container types and extra fields are preserved.

The upscaled latent itself keeps the exact learned output shape. H3 crops its padding back internally.

### Model cache, devices and offload

- Loaded networks are cached per `(checkpoint, device, precision)` and reused across runs.
- `offload_after_upscale` (3D, Refine and Provider; default off) moves the cached network to CPU after use. The next use moves it back. On the Refine node, offload happens after the upscale and before sampler 2, which is where reclaiming VRAM matters most. Leave it off when VRAM allows, because every run then pays the transfer.
- The 2D, 3D and Refine nodes fall back to CPU when CUDA is unavailable. The Provider raises an error instead, because it never changes device silently.
- The 3D network always processes the full temporal sequence in one pass. It is never split into temporal chunks.
- Attention layers in the learned network are disabled at inference.

## Node reference

### Minimax H3 Latent Upscaler (2D)

| Input | Type | Default | Notes |
| --- | --- | --- | --- |
| `latent` | LATENT | | 24-channel video latent |
| `model_name` | combo | | 2D-backbone checkpoint |
| `scale` | FLOAT | 2.0 | 1.0–4.0 |
| `device` | combo | `cuda` | `cuda`, `cpu` |
| `precision` | combo | `fp32` | `fp32`, `fp16`, `bf16` |

Output: `LATENT`.

### Minimax H3 Latent Upscaler (3D)

| Input | Type | Default | Notes |
| --- | --- | --- | --- |
| `latent` | LATENT | | 24-channel video latent, 5D or 4D |
| `model_name` | combo | | 3D checkpoint |
| `mode` | combo | `scale by multiplier` | reveals `scale` (1.0–4.0), `width`/`height` (64–4096) or `megapixels` (0.1–8.0) |
| `align` | INT | 32 | pixel alignment, combined as `lcm(align, 16)` |
| `keep_proportion` | BOOLEAN | true | |
| `device` | combo | `cuda` | `cuda`, `cpu` |
| `precision` | combo | `fp16` | `fp32`, `fp16`, `bf16` |
| `offload_after_upscale` | BOOLEAN | false | |

Defaults for size inputs: `scale` 2.0, `width` 1280, `height` 704, `megapixels` 1.0. Output: `LATENT`.

### MiniMax H3 Latent Upscaler + Refine (3D)

Takes the 3D node's `model_name`, `mode`, `scale`, `width`, `height`, `megapixels`, `align`, `keep_proportion`, `device`, `precision` (default `fp16`) and `offload_after_upscale`. All size inputs are shown, and only those used by the selected `mode` take effect. Additional inputs:

| Input | Type | Required | Notes |
| --- | --- | --- | --- |
| `latent` | LATENT | yes | native joint AV latent, or the video latent for split input |
| `noise` | NOISE | yes | e.g. `RandomNoise` |
| `sampler` | SAMPLER | yes | e.g. `KSamplerSelect` |
| `sigmas` | SIGMAS | yes | partial-denoise schedule, `sigmas[0] < 1` |
| `lock_audio` | BOOLEAN | yes | default true |
| `cfg` | FLOAT | yes | default 1.0; used only when `negative` is connected on the native path |
| `audio_latent` | LATENT | split input only | required when `latent` holds only video; must be disconnected for a joint AV latent |
| `refine_state` | H3_CONTINUUM_REFINE_STATE | Continuum | takes precedence over `model`/`positive`/`negative` |
| `model` | MODEL | native | required without `refine_state` |
| `positive` | CONDITIONING | native | required without `refine_state` |
| `negative` | CONDITIONING | no | native path only; enables CFG |

Output: `LATENT` list, final and decode-ready. A single latent input yields a one-item list.

### MiniMax H3 Latent Upscaler Provider (3D) [Experimental]

| Input | Type | Default |
| --- | --- | --- |
| `model_name` | combo | `minimax_h3_latent_upscaler_3d_bf16.safetensors` when present |
| `device` | combo | `cuda` |
| `precision` | combo | `bf16` |
| `offload_after_upscale` | BOOLEAN | false |

Output: `H3_LATENT_UPSCALER`.

## Provider API

The `H3_LATENT_UPSCALER` value is an immutable `H3LatentUpscalerProvider` (`nodes/minimax_h3_handoff_provider.py`). It holds configuration only. Checkpoint loading and the model cache stay inside this package.

| Attribute | Value |
| --- | --- |
| `kind` | `"minimax_h3_learned_latent_upscaler"` |
| `api_version` | `1` |
| `h3_patch_lattice_api` | `2` |
| `h3_transport_lattices` | `("h3_dense_patch_center_lattice_v2", "h3_rope_box_half_pixel_lattice_v1")` |
| `model_name`, `device`, `precision`, `offload_after_upscale` | node inputs |

- `upscale_clean_video(video, *, target_h, target_w)` takes a clean floating-point `B×24×T×H×W` video latent and returns it at exactly `target_h × target_w` with batch, channels, time and dtype preserved. Shrinking either axis is rejected, and non-finite results raise an error.
- `upscale_clean_video_h3_patch_lattice(video, *, target_h, target_w, spatial_lattice="h3_dense_patch_center_lattice_v2")` runs the same network but resamples the dense encoder features on an H3 spatial-RoPE lattice before the decoder. It requires even spatial axes and exists for consumers that must match H3's patch-coordinate map at a handoff. The ordinary node and provider paths keep the trained half-pixel interpolation. Background: [`docs/TRANSFER_LATTICE_20261004.md`](docs/TRANSFER_LATTICE_20261004.md).
  - `h3_dense_patch_center_lattice_v2` (default) reads each H3 patch coordinate as the patch center: adjacent dense cells average to it.
  - `h3_rope_box_half_pixel_lattice_v1` reads each patch coordinate as the patch start inside H3's centered, `endpoint=False` RoPE frame box and places dense cells at half-pixel centers in that box. For equal source/target aspect ratios it equals the trained half-pixel map. It differs from v2 by one constant translation of `1 - target_step / source_step` source cells (about 0.4 source cells for 32×44→54×72).

## Model and training data

- **Input:** 24-channel H3 video latent, normalized with the training per-channel statistics before inference and denormalized afterwards.
- **3D architecture (default configuration; the loader reads the actual values from the checkpoint):** `in_channels=24`, 12 encoder and 12 decoder residual blocks with 512 channels, a temporal convolution (kernel 5) after every second residual block, and a learned scale embedding.
- **Resampling:** trilinear between encoder and decoder in the 3D network, bilinear in the 2D network. The time axis is preserved and only H×W are scaled.

The upstream checkpoints were trained on about 80,000 paired samples (low-resolution latent and high-resolution target):

| Modality | Pairs | Share |
| --- | --- | --- |
| Video clips | ~70,000 | ~87.5% |
| 2K images | ~8,000 | ~10% |

| Scale | Share |
| --- | --- |
| 2× | 40% |
| 1.5×, 2.5×, 3×, 4× | 10% each |
| arbitrary 1.0×–4.0× | 10% |

## Differences from upstream

Upstream changes are reviewed and adopted selectively. The current intentional differences:

- **No temporal chunking.** The 3D network runs once over the full sequence. Its stacked 3D/temporal convolutions and GroupNorm statistics span time, so chunked execution with partial overlap is not equivalent to full-sequence execution and can introduce boundary differences.
- **Offload is opt-in.** `offload_after_upscale` defaults to off instead of unloading after every run, which avoids repeated transfers on repeated or high-VRAM runs.
- **Dual-axis alignment keeps `keep_proportion`.** Both axes are aligned on `lcm(align, 16)` while aspect-ratio lock is retained.
- **Normalization stays in place** on a private copy. Out-of-place arithmetic in the same dtype would add full-size temporaries without changing numerical precision.
- **H3-specific additions** that upstream does not have: the Refine node, Continuum interop and the handoff Provider.

## Testing

GitHub Actions runs on pushes to `main` and on pull requests, against several pinned ComfyUI source revisions with Python 3.12, and on one revision with Python 3.10, 3.11 and 3.13. Each job runs Ruff and `compileall`, native ComfyUI source-contract tests, and the regression suite.

The suite covers native and split AV validation, learned-upscaler delegation, exact output geometry and alignment, keyframe and reference conditioning, denoise-mask reconstruction, Continuum `refine_state` resolution and precedence, sequential chunk-prefix carry, guider and sampler invocation, the partial-denoise guard, locked-audio restoration, full-sequence execution, cache device restore, offload placement, the provider contract and the H3 patch-lattice transform.

The tests do not load trained weights. Output quality, speed and peak VRAM depend on the workload and hardware and have to be checked in real workflows.

## Coordinated H3 release set

This package is released together with the other H3 components. Per-version details are in [RELEASE_NOTES.md](RELEASE_NOTES.md).

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.10](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.10) | [#96](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/96) |
| Sol-H3 | [v0.1.9](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.9) | [#39](https://github.com/xmarre/ComfyUI-Sol-H3/pull/39) |
| VDN-H3-Plus | [v1.5.8](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.8) | [#38](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/38) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | [#39](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/40) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
For INT8 fused MLP runtime adapters, retain that ComfyUI Patcher PR overlay until
the repair is available upstream. The independent Core #16720 optimization is
not included in this release set.

## Acknowledgments

- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler): the original nodes, network and pre-trained checkpoints.
- [Ttl/ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale): the neural latent-upscaling approach.
- LTX 2.3 Spatial Upscaler (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`): architectural reference for the network.
- [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) and ComfyUI's MiniMax H3 sampling code were studied for the refinement integration, which was implemented independently. This package depends on neither Tr1dae's nor Mamad8's upscaler packages.
